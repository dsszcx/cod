"""Best-checkpoint residual magnitude and learned branch parameter audit."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import resolve
from tools.semantic_prior.training_core import build_model
from dataloader_relation import RelationDataset,collate_relation_batch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/relation_pilot10_20261026.yaml')
    args=ap.parse_args();cfg=C.load_config(str(resolve(args.config)));root=resolve(cfg['paths']['out_dir'])
    target=root/'FUSION_AUDIT.json'
    if target.exists(): raise FileExistsError(str(target))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    reports={}
    for arm in ('direct','relation'):
        state=torch.load(root/arm/'best.pth',map_location='cpu',weights_only=True)
        model,_=build_model(arm,cfg['seed'],str(resolve(cfg['paths']['imagenet'])),cfg.get('fusion_options'))
        initial={k:v.clone() for k,v in model.region_fusion.state_dict().items()}
        model.load_state_dict(state['model'],strict=True)
        changed={k:float((v-initial[k]).abs().max()) for k,v in model.region_fusion.state_dict().items()}
        model.cuda().eval()
        samples=[]
        def hook(module,inputs,output):
            _,debug=module.forward(*inputs,return_debug=True)
            denominator=inputs[0].float().norm().clamp_min(1e-12)
            samples.append({'direct_relative_l2':float((module.alpha_direct*debug['direct_map']).float().norm()/denominator),
                            'relation_relative_l2':float((module.alpha_relation*debug['relation_map']).float().norm()/denominator)})
        handle=model.region_fusion.register_forward_hook(hook)
        dataset=RelationDataset(resolve(cfg['paths']['manifests'])/'pilot_val.csv','val',arm,352,cfg['seed'])
        with torch.inference_mode():
            for i in range(len(dataset)):
                batch=collate_relation_batch([dataset[i]])
                model(**{k:v.cuda() for k,v in batch['inputs'].items()})
        handle.remove()
        reports[arm]={'best_epoch':state['epoch'],'changed_parameter_max_abs':changed,
                      'mean_direct_relative_l2':float(np.mean([s['direct_relative_l2'] for s in samples])),
                      'mean_relation_relative_l2':float(np.mean([s['relation_relative_l2'] for s in samples])),
                      'max_direct_relative_l2':float(max(s['direct_relative_l2'] for s in samples)),
                      'max_relation_relative_l2':float(max(s['relation_relative_l2'] for s in samples))}
        del model,state;torch.cuda.empty_cache()
        print('FUSION_AUDIT',arm,{k:v for k,v in reports[arm].items() if k!='changed_parameter_max_abs'},flush=True)
    C.write_json(str(target),reports)
    print('TRAINED_FUSION_AUDIT_DONE',flush=True)


if __name__=='__main__':main()
