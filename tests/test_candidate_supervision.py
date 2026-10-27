"""No dataset training: purity labels, gradients and unchanged inference API."""
import gc
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from lib.region_fusion import RegionRelationFusion
from Network_Relation import Network
from test_region_network import inputs, fusion_args, permute
from tools.semantic_prior.candidate_supervision import (
    candidate_targets, candidate_reliability_loss, image_balanced_mean)
from tools.semantic_prior.training_core import deep_loss


def test_targets():
    masks = torch.zeros(1, 5, 8, 8)
    gt = torch.zeros(1, 1, 8, 8, requires_grad=True)
    with torch.no_grad():
        gt[:, :, :4, :4] = 1
    masks[:, 0, :4, :4] = 1   # complete target
    masks[:, 1, :1, :1] = 1   # small, correct part
    masks[:, 2, 4:, 4:] = 1   # context/background; remains a relation node
    masks[:, 3, :4, :] = 1    # half foreground
    masks[:, 4] = 1           # invalid padding with a nonzero payload
    valid = torch.tensor([[True, True, True, True, False]])
    labels = candidate_targets(masks, valid, gt)
    torch.testing.assert_close(labels['purity'], torch.tensor([[1., 1., 0., .5, 0.]]))
    assert labels['foreground_recall'][0, 1] == 1 / 16
    assert not labels['purity'].requires_grad
    logits = torch.zeros(1, 5, requires_grad=True)
    loss = candidate_reliability_loss(logits, masks, valid, gt)
    loss.backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] < 0
    assert logits.grad[0, 2] > 0 and logits.grad[0, 3] == logits.grad[0, 4] == 0
    assert gt.grad is None
    # Duplicating nodes inside one image does not give it extra batch weight.
    values = torch.tensor([[2., 2., 2.], [6., 0., 0.]])
    validity = torch.tensor([[True, True, True], [True, False, False]])
    assert image_balanced_mean(values, validity) == 4
    for n in (0, 3):
        logits = torch.zeros(2, n, requires_grad=True)
        loss = candidate_reliability_loss(logits, torch.zeros(2, n, 8, 8),
                                          torch.zeros(2, n, dtype=torch.bool), torch.zeros(2, 1, 8, 8))
        assert loss == 0 and torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None and not logits.grad.any()
    try:
        candidate_targets(masks, valid, gt[:, :, :4])
    except ValueError:
        pass
    else:
        raise AssertionError('Misaligned post-transform supervision accepted')


def test_fusion_auxiliary():
    batch = inputs()
    module = RegionRelationFusion(8, token_dim=12)
    module.use_node_geometry = False
    features = torch.randn(1, 8, 8, 8, requires_grad=True)
    output, aux = module(features, **fusion_args(batch), return_debug=True)
    assert torch.equal(output, features)  # supervision works even at alpha=0
    loss = candidate_reliability_loss(aux['direct_logits'], batch['region_masks'],
                                      aux['effective_valid'], torch.ones(1, 1, 64, 64))
    loss.backward()
    assert module.direct_gate.weight.grad.abs().sum() > 0
    assert module.node_encoder[0].weight.grad.abs().sum() > 0
    assert features.grad.abs().sum() > 0
    # Purity supervision has no direct gradient into edge/message parameters.
    assert module.edge_encoder[0].weight.grad is None
    assert module.relation_gate.weight.grad is None
    assert aux['effective_valid'].all() and aux['edge_weights'].sum() > 0
    with torch.no_grad():
        _, swapped = module(features, **fusion_args(permute(batch, [2, 0, 1])), return_debug=True)
        torch.testing.assert_close(swapped['direct_logits'], aux['direct_logits'][:, [2, 0, 1]])
        # Removing j->i leaves direct support unchanged and can alter only the
        # recipient's spatial support (node geometry is disabled in both arms).
        without = dict(batch, relation_valid=batch['relation_valid'].clone())
        without['relation_valid'][0, 0, 1] = False
        _, removed = module(features, **fusion_args(without), return_debug=True)
        torch.testing.assert_close(aux['direct_map'], removed['direct_map'], rtol=0, atol=0)
        delta = aux['relation_map'] - removed['relation_map']
        outside = (aux['coverage'][:, :1] == 0).expand_as(delta)
        assert not delta[outside].any()


def test_network_api():
    batch = inputs()
    model = Network(imagenet_pretrained=False).eval()
    with torch.inference_mode():
        default = model(**batch)
        outputs, aux = model(**batch, return_aux=True)
        assert len(default) == len(outputs) == 4
        for a, b in zip(default, outputs):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert aux['direct_logits'].shape == batch['region_valid'].shape
        assert 'gt' not in batch
    # One combined-loss optimizer step on synthetic data, NOT a COD result.
    model.train()
    train_batch = {k: v.repeat((2,) + (1,) * (v.ndim - 1)) for k, v in batch.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = model.region_fusion.direct_gate.weight.detach().clone()
    target = torch.ones(2, 1, 64, 64)
    outputs, aux = model(**train_batch, return_aux=True)
    loss = deep_loss(outputs, target) + .1 * candidate_reliability_loss(
        aux['direct_logits'], train_batch['region_masks'], aux['effective_valid'], target)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert not torch.equal(before, model.region_fusion.direct_gate.weight)
    del model
    gc.collect()


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.manual_seed(22)
    test_targets()
    test_fusion_auxiliary()
    test_network_api()
    print('CANDIDATE_SUPERVISION_TESTS_PASSED', flush=True)
