"""CPU first-batch backward only: expose zero-alpha and global-clipping coupling."""
import gc
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import resolve
from tools.semantic_prior.training_core import build_model,deep_loss,forward_arm,seed_all
from dataloader_relation import RelationDataset,collate_relation_batch


def main():
    torch.set_num_threads(4)
    manifest=resolve('outputs/RELATION_CAM_EXTENDED_20261026_V1/pilot_train.csv')
    target=manifest.parent/'INITIAL_GRADIENT_DIAGNOSIS.json'
    if target.exists():raise FileExistsError(str(target))
    results=[];reference_before=reference_after=None
    for label,arm,alpha,norm in (('rgb','rgb',0.,'raw'),('direct_zero','direct',0.,'raw'),
                                ('relation_zero','relation',0.,'raw'),('relation_scaled_small','relation',.01,'feature_rms')):
        ds=RelationDataset(manifest,'train',arm,352,20261022)
        generator=torch.Generator().manual_seed(20261022)
        loader=DataLoader(ds,batch_size=2,shuffle=True,generator=generator,num_workers=0,collate_fn=collate_relation_batch)
        model,_=build_model(arm,20261022,'/srv/cod-workspace/pretrained/pvt_v2_b2.pth',{'alpha_init':alpha,'projection_normalization':norm})
        seed_all(20261022);model.train();batch=next(iter(loader))
        loss=deep_loss(forward_arm(model,batch['inputs'],arm),batch['gt']);loss.backward()
        param=dict(model.named_parameters())['backbone.patch_embed1.proj.weight']
        before=param.grad.detach().clone()
        region_squared=sum(float(p.grad.detach().square().sum()) for n,p in model.named_parameters() if n.startswith('region_fusion.') and p.grad is not None)
        alpha_grads={n:float(p.grad) if p.grad is not None else None for n,p in model.named_parameters() if n.startswith('region_fusion.alpha')}
        total=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.))
        after=param.grad.detach().clone()
        if arm=='rgb':reference_before,reference_after=before,after
        results.append({'label':label,'arm':arm,'loss':float(loss.detach()),'image_ids':batch['image_ids'],
            'unclipped_total_gradient_l2':total,'unclipped_region_gradient_l2':region_squared**.5,
            'global_clip_multiplier':min(1.,1./(total+1e-6)),
            'patch_gradient_max_abs_difference_vs_rgb_before_clip':float((before-reference_before).abs().max()),
            'patch_gradient_max_abs_difference_vs_rgb_after_clip':float((after-reference_after).abs().max()),
            'alpha_gradients':alpha_grads,'optimizer_steps':0})
        del model,loader,ds,loss,batch,param,before,after;gc.collect()
    C.write_json(str(target),{'device':'cpu','results':results,'training':False,'test_evaluated':False,
        'warning':'CPU first-batch structural diagnostic, not GPU training effect quantification. Additional alpha gradients can change shared global clipping.'})
    print('INITIAL_GRADIENT_DIAGNOSIS_DONE',results,flush=True)


if __name__=='__main__':main()
