"""Quality-head gradient isolation, exact observer control and free COD adapters."""
import copy
import gc
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from test_region_network import inputs, fusion_args, permute
from lib.region_fusion import RegionRelationFusion
from Network_Relation import Network
from tools.semantic_prior.candidate_supervision import candidate_reliability_loss
from tools.semantic_prior.training_core import deep_loss, optimizer_parameters, seed_all


def test_fusion():
    batch = inputs()
    module = RegionRelationFusion(8, 12)
    module.use_node_geometry = False
    module.alpha_direct.data.fill_(.01)
    module.alpha_relation.data.fill_(.01)
    features = torch.randn(1, 8, 8, 8, requires_grad=True)
    reference = module(features, **fusion_args(batch))
    rng = torch.get_rng_state().clone()
    module.enable_quality_head(22, stop_gradient=True, feedback=True)
    assert torch.equal(rng, torch.get_rng_state())
    result, aux = module(features, **fusion_args(batch), return_debug=True)
    assert torch.equal(result, reference)  # all quality adapters start at zero
    for stop in (True, False):
        module.zero_grad(set_to_none=True)
        features.grad = None
        module.quality_stop_gradient = stop
        _, aux = module(features, **fusion_args(batch), return_debug=True)
        loss = candidate_reliability_loss(aux['quality_logits'], batch['region_masks'],
                                          aux['effective_valid'], torch.ones(1, 1, 64, 64))
        loss.backward()
        assert module.quality_head[0].weight.grad.abs().sum() > 0
        assert (features.grad is None) if stop else (features.grad.abs().sum() > 0)
        assert module.direct_gate.weight.grad is None
        assert module.node_encoder[0].weight.grad is None
        assert module.edge_encoder[0].weight.grad is None
    module.quality_stop_gradient = True
    module.zero_grad(set_to_none=True)
    output, aux = module(features, **fusion_args(batch), return_debug=True)
    output.square().mean().backward()
    assert all(p.grad is None for p in module.quality_head.parameters())
    assert module.quality_to_direct.weight.grad.abs().sum() > 0
    assert module.quality_to_edges.weight.grad.abs().sum() > 0
    with torch.no_grad():
        changed_roles = dict(batch, role_hint=torch.flip(batch['role_hint'], [-1]))
        _, alt = module(features, **fusion_args(changed_roles), return_debug=True)
        torch.testing.assert_close(alt['quality_logits'], aux['quality_logits'], rtol=0, atol=0)
        _, swapped = module(features, **fusion_args(permute(batch, [2, 0, 1])), return_debug=True)
        torch.testing.assert_close(swapped['quality_logits'], aux['quality_logits'][:, [2, 0, 1]])
        module.quality_to_direct.weight.fill_(2.)
        module.quality_to_relation.weight.fill_(-1.)
        module.quality_to_edges.weight.fill_(.5)
        active = module(features, **fusion_args(batch))
        module.use_quality_feedback = False
        assert not torch.equal(active, module(features, **fusion_args(batch)))
        module.use_quality_feedback = True
        empty = dict(batch, region_valid=torch.zeros_like(batch['region_valid']))
        assert torch.equal(module(features, **fusion_args(empty)), features)
        no_pairs = dict(batch, relation_valid=torch.zeros_like(batch['relation_valid']))
        assert not module(features, **fusion_args(no_pairs), return_debug=True)[1]['relation_map'].any()


def test_exact_observer_training():
    seed_all(22)
    model = Network(imagenet_pretrained=False)
    model.region_fusion.enable_quality_head(23, stop_gradient=True, feedback=False)
    model.region_fusion.alpha_direct.data.fill_(.01)
    model.region_fusion.alpha_relation.data.fill_(.01)
    observer = copy.deepcopy(model)
    main_a, head_a = optimizer_parameters(model)
    main_b, head_b = optimizer_parameters(observer)
    assert not {id(p) for p in main_b} & {id(p) for p in head_b}
    opt_a = torch.optim.AdamW(main_a, lr=1e-4, weight_decay=1e-4)
    opt_b = torch.optim.AdamW(main_b, lr=1e-4, weight_decay=1e-4)
    opt_q = torch.optim.AdamW(head_b, lr=1e-4, weight_decay=1e-4)
    initial_head = [p.detach().clone() for p in head_b]
    batch = inputs()
    batch = {k: v.repeat((2,) + (1,) * (v.ndim - 1)) for k, v in batch.items()}
    gt = torch.ones(2, 1, 64, 64)
    model.train(); observer.train()
    for step in range(2):
        for network, opt, main, auxiliary in ((model, opt_a, main_a, False), (observer, opt_b, main_b, True)):
            seed_all(100 + step)
            opt.zero_grad(set_to_none=True)
            opt_q.zero_grad(set_to_none=True)
            outputs, debug = network(**batch, return_aux=True)
            loss = deep_loss(outputs, gt)
            if auxiliary:
                loss = loss + .1 * candidate_reliability_loss(debug['quality_logits'], batch['region_masks'],
                                                             debug['effective_valid'], gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(main, 1.)
            opt.step()
            if auxiliary:
                torch.nn.utils.clip_grad_norm_(head_b, 1.)
                opt_q.step()
        for name, value in model.state_dict().items():
            if '.quality_head.' not in name:
                assert torch.equal(value, observer.state_dict()[name]), name
    assert any(not torch.equal(a, b) for a, b in zip(initial_head, head_b))
    del model, observer
    gc.collect()


if __name__ == '__main__':
    torch.set_num_threads(4)
    seed_all(22)
    test_fusion()
    test_exact_observer_training()
    print('QUALITY_DECOUPLING_TESTS_PASSED')
