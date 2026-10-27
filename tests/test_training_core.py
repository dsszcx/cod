"""Synthetic optimizer regression; distinct from the pilot training results."""
from pathlib import Path
import sys
import gc

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from tools.semantic_prior.training_core import build_model, structure_loss, deep_loss, forward_arm, RELATION_PREFIXES
from MyTrain import hybrid_e_loss
from tests.test_region_network import inputs


def main():
    torch.set_num_threads(4)
    torch.manual_seed(9)
    logits=torch.randn(2,1,64,64,requires_grad=True)
    for mask in (torch.zeros_like(logits),torch.ones_like(logits),(torch.rand_like(logits)>.7).float()):
        torch.testing.assert_close(structure_loss(logits,mask),hybrid_e_loss(logits,mask,"fixed"),rtol=1e-6,atol=1e-6)
    checkpoint=ROOT/'pretrained/pvt_v2_b2.pth'
    batch=inputs()
    batch={k:v.repeat((2,)+(1,)*(v.ndim-1)) for k,v in batch.items()}
    rgb_digest,region_digest=None,None
    for arm in ('rgb','direct','relation'):
        model,meta=build_model(arm,22,str(checkpoint))
        if rgb_digest is None: rgb_digest=meta['rgb_initial_sha256']
        assert meta['rgb_initial_sha256']==rgb_digest
        if arm!='rgb':
            if region_digest is None: region_digest=meta['region_initial_sha256']
            assert meta['region_initial_sha256']==region_digest
            assert model.region_fusion.use_node_geometry is False
        frozen={n:p.clone() for n,p in model.named_parameters() if not p.requires_grad}
        model.train()
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        outputs=forward_arm(model,batch,arm)
        loss=deep_loss(outputs,torch.zeros(2,1,64,64))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        opt.step()
        assert all(torch.equal(dict(model.named_parameters())[n],p) for n,p in frozen.items())
        if arm=='direct':
            assert frozen and model.region_fusion.alpha_relation.item()==0
            assert all(not p.requires_grad for n,p in model.region_fusion.named_parameters() if n.startswith(RELATION_PREFIXES))
        print('CORE_ARM_TEST_PASSED',arm,float(loss.detach()),flush=True)
        del model,opt,outputs,loss
        gc.collect()
    print('TRAINING_CORE_TESTS_PASSED: identical init, fixed loss, train-mode BN, optimizer step and exact freezing',flush=True)


if __name__=='__main__': main()
