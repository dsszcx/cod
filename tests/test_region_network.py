"""CPU forward/backward checks only: no optimizer, dataset training or metrics."""
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.region_relations import collate_region_samples
from lib.region_fusion import RegionRelationFusion
from Network_PVTv2 import Network as RGBNetwork
from Network_Relation import Network


def inputs(size=64):
    masks = np.zeros((3, size, size), np.uint8)
    masks[0, 8:48, 8:48] = 1
    masks[1, 12:16, 16:20] = 1
    masks[2, 30:60, 40:60] = 1
    sample = {"masks": masks, "rgb": np.random.default_rng(12).normal(size=(3, size, size)).astype(np.float32),
              "role_hint": np.eye(4, dtype=np.float32)[[0, 1, 2]], "role_hint_valid": np.ones(3, bool)}
    return collate_region_samples([sample])


def fusion_args(batch):
    return {k: v for k, v in batch.items() if k != "rgb"}


def permute(batch, order):
    result = dict(batch)
    for key in ("region_masks", "region_valid", "role_hint", "role_hint_valid"):
        result[key] = batch[key][:, order]
    for key in ("relation_geom", "relation_valid"):
        result[key] = batch[key][:, order][:, :, order]
    return result


def test_fusion():
    batch = inputs()
    features = torch.randn(1, 8, 8, 8, requires_grad=True)
    module = RegionRelationFusion(8, token_dim=12)
    out = module(features, **fusion_args(batch))
    assert torch.equal(out, features)  # zero residual coefficients
    out.square().mean().backward()
    for alpha in (module.alpha_direct, module.alpha_relation):
        assert alpha.grad is not None and torch.isfinite(alpha.grad) and alpha.grad.abs() > 0
    # Zero-initialized alpha blocks internal first-step gradients by design.
    assert module.edge_encoder[0].weight.grad.abs().sum() == 0
    module.zero_grad(set_to_none=True)
    with torch.no_grad():
        module.alpha_direct.fill_(.2)
        module.alpha_relation.fill_(.3)
    out, debug = module(features, **fusion_args(batch), return_debug=True)
    assert torch.isfinite(out).all() and not torch.equal(out, features)
    out.square().mean().backward()
    for name in ("node_encoder.0.weight", "edge_encoder.0.weight", "values.0.weight",
                 "message.0.weight", "direct_project.weight", "relation_project.weight",
                 "direct_gate.weight", "relation_gate.weight"):
        grad = dict(module.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    assert features.grad is not None and torch.isfinite(features.grad).all()
    permuted = permute(batch, [2, 0, 1])
    torch.testing.assert_close(module(features, **fusion_args(permuted)), out, rtol=1e-5, atol=1e-6)
    no_pairs = dict(batch, relation_valid=torch.zeros_like(batch["relation_valid"]))
    _, no_pair_debug = module(features, **fusion_args(no_pairs), return_debug=True)
    assert not no_pair_debug["relation_map"].any()
    assert no_pair_debug["direct_map"].abs().sum() > 0
    empty = dict(batch, region_valid=torch.zeros_like(batch["region_valid"]))
    no_prior, dbg = module(features, **fusion_args(empty), return_debug=True)
    assert torch.equal(no_prior, features)
    assert not dbg["direct_map"].any() and not dbg["relation_map"].any() and not dbg["edge_weights"].any()
    # Unknown roles do not disable visually supported nodes/relations.
    unknown = dict(batch, role_hint_valid=torch.zeros_like(batch["role_hint_valid"]))
    _, dbg = module(features, **fusion_args(unknown), return_debug=True)
    assert dbg["effective_valid"].all() and dbg["edge_weights"].sum() > 0
    # A single image pixel survives area pooling as fractional coverage.
    tiny = dict(batch, region_masks=torch.zeros_like(batch["region_masks"]))
    tiny["region_masks"][0, 0, 1, 1] = 1
    _, dbg = module(features, **fusion_args(tiny), return_debug=True)
    assert dbg["coverage"][0, 0].sum() > 0 and dbg["direct_map"].abs().sum() > 0
    assert not dbg["relation_map"].any()
    # Empty N=0 is legal in addition to the loader's all-invalid padded slot.
    zero_nodes = permute(batch, [])
    assert torch.equal(module(features, **fusion_args(zero_nodes)), features)
    # Adding invalid slots, even with nonzero finite payload, cannot affect output.
    padded = {}
    n = 3
    for key, v in batch.items():
        if key == "rgb":
            padded[key] = v
        elif key in ("relation_geom", "relation_valid"):
            shape = list(v.shape); shape[1] = shape[2] = 5
            padded[key] = torch.zeros(shape, dtype=v.dtype)
            padded[key][:, :n, :n] = v
        else:
            shape = list(v.shape); shape[1] = 5
            padded[key] = torch.ones(shape, dtype=v.dtype)
            padded[key][:, :n] = v
    padded["region_valid"][:, n:] = False
    torch.testing.assert_close(module(features, **fusion_args(padded)), out, rtol=1e-5, atol=1e-6)


def test_full_network():
    batch = inputs()
    baseline = RGBNetwork(channel=64, imagenet_pretrained=False, use_ca=False).eval()
    model = Network(channel=64, imagenet_pretrained=False, use_ca=False).eval()
    assert model.load_rgb_state_dict(baseline.state_dict()) == len(baseline.state_dict())
    assert len(baseline.state_dict()) == 740
    try:
        model.load_rgb_state_dict({})
    except ValueError:
        pass
    else:
        raise AssertionError("Incomplete baseline accepted")
    wrong = dict(baseline.state_dict())
    wrong["backbone.patch_embed1.proj.weight"] = wrong["backbone.patch_embed1.proj.weight"][:, :1]
    try:
        model.load_rgb_state_dict(wrong)
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong baseline shape accepted")
    with torch.no_grad():
        ref = baseline(batch["rgb"])
        initial = model(**batch)
        assert len(initial) == 4 and all(t.shape == (1, 1, 64, 64) for t in initial)
        for a, b in zip(ref, initial):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        model.region_fusion.alpha_direct.fill_(.2)
        model.region_fusion.alpha_relation.fill_(.3)
        empty = dict(batch, region_valid=torch.zeros_like(batch["region_valid"]))
        for a, b in zip(ref, model(**empty)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        active = model(**batch)
        assert all(torch.isfinite(t).all() for t in active)
        for a, b in zip(active, model(**permute(batch, [2, 0, 1]))):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    # One synthetic backward, in eval mode to keep BN statistics unchanged.
    # No optimizer/step, loss supervision or training data are involved.
    model.zero_grad(set_to_none=True)
    outputs = model(**batch)
    sum(t.square().mean() for t in outputs).backward()
    for name in ("region_fusion.edge_encoder.0.weight", "backbone.patch_embed1.proj.weight", "NCD.conv5.weight"):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name


def main():
    torch.set_num_threads(4)
    torch.manual_seed(22)
    test_fusion()
    print("FUSION_TESTS_PASSED", flush=True)
    test_full_network()
    print("REGION_NETWORK_TESTS_PASSED: forward, backward, permutation, zero priors, RGB equivalence, strict 740-key loading", flush=True)


if __name__ == "__main__":
    main()
