"""Opt-in calibration: finite gradients, empty identity and gate suppression."""
from pathlib import Path
import sys
import torch
import gc

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from lib.region_fusion import RegionRelationFusion
from test_region_network import inputs, fusion_args, permute
from tools.semantic_prior.training_core import build_model


def main():
    torch.set_num_threads(4);torch.manual_seed(22)
    batch=inputs();args=fusion_args(batch)
    features=(torch.randn(1,8,8,8)*40).requires_grad_()
    module=RegionRelationFusion(8,12)
    module.use_node_geometry=False
    with torch.no_grad():
        module.alpha_direct.fill_(.01);module.alpha_relation.fill_(.01)
    _,raw=module(features,**args,return_debug=True)
    module.projection_normalization='feature_rms'
    output,debug=module(features,**args,return_debug=True)
    assert torch.isfinite(output).all()
    assert debug['direct_map'].norm()>raw['direct_map'].norm()*3
    output.square().mean().backward()
    for key in ('node_encoder.0.weight','edge_encoder.0.weight','direct_project.weight','relation_project.weight'):
        grad=dict(module.named_parameters())[key].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,key
    torch.testing.assert_close(module(features,**fusion_args(permute(batch,[2,0,1]))),output,rtol=1e-5,atol=1e-5)
    empty=dict(args,region_valid=torch.zeros_like(args['region_valid']))
    assert torch.equal(module(features,**empty),features)
    no_pairs=dict(args,relation_valid=torch.zeros_like(args['relation_valid']))
    assert not module(features,**no_pairs,return_debug=True)[1]['relation_map'].any()
    with torch.no_grad():
        module.direct_gate.weight.zero_();module.direct_gate.bias.fill_(20)
    high=module(features,**args,return_debug=True)[1]['direct_map'].norm()
    with torch.no_grad():module.direct_gate.bias.fill_(-20)
    low=module(features,**args,return_debug=True)[1]['direct_map'].norm()
    assert low<high*1e-6  # RMS normalization must not divide away learned gates
    fingerprints=[]
    for arm in ('direct','relation'):
        model,meta=build_model(arm,22,str(ROOT/'pretrained/pvt_v2_b2.pth'),
                               {'alpha_init':.01,'projection_normalization':'feature_rms'})
        assert abs(model.region_fusion.alpha_direct.item()-.01)<1e-8
        assert model.region_fusion.projection_normalization=='feature_rms'
        assert abs(model.region_fusion.alpha_relation.item()-(.01 if arm=='relation' else 0.))<1e-8
        fingerprints.append((meta['rgb_initial_sha256'],meta['region_initial_sha256']))
        del model;gc.collect()
    assert fingerprints[0]==fingerprints[1]
    print('FUSION_CALIBRATION_TESTS_PASSED')


if __name__=='__main__':main()
