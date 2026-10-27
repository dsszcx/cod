"""Predeclare the complete 2-seed x 7-condition pilot, not result-driven selection."""
from pathlib import Path
import sys
import yaml

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import resolve


def main():
    root=resolve('outputs/CAM_FUSION_SWEEP_20261026_V1')
    root.mkdir(exist_ok=False)
    base={'version':'cam_fusion_sweep_20261026_v1','data_scope':'cam_only',
        'expected_train_images':128,'expected_val_images':32,'epochs':20,'input_size':352,'batch_size':2,
        'learning_rate':1e-4,'weight_decay':1e-4,'workers':0,'amp':False,'node_geometry':False,
        'deterministic_algorithms':True,'collect_fusion_stats':True,
        'deadline_utc':'2026-10-26T07:05:51+00:00'}
    jobs=[]
    for seed in (20261022,20261023):
        for name,alpha,norm,arms in (
            ('raw_zero',0.,'raw',['rgb','direct','relation']),
            ('raw_small',.01,'raw',['direct','relation']),
            ('scaled_small',.01,'feature_rms',['direct','relation'])):
            run=root/('s%d_%s'%(seed,name))
            cfg={**base,'seed':seed,'arms':arms,'fusion_options':{'alpha_init':alpha,'projection_normalization':norm},
                 'paths':{'manifests':'outputs/RELATION_CAM_EXTENDED_20261026_V1',
                          'imagenet':'/srv/cod-workspace/pretrained/pvt_v2_b2.pth','out_dir':str(run)}}
            path=root/('s%d_%s.yaml'%(seed,name))
            path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
            jobs.append({'seed':seed,'condition':name,'config':str(path),'out_dir':str(run),'arms':arms})
    C.write_json(str(root/'PLAN.json'),{'created_utc':C.utc_now(),'jobs':jobs,'n_training_runs':14,
        'train_images':128,'val_images':32,'epochs':20,'selection':'predeclared all 7 conditions for both seeds',
        'test_evaluated':False,'metric':'native-size raw-sigmoid macro MAE; best and final epoch reported',
        'time_budget_policy':'stop starting new jobs at configured deadline; preserve any partial results'})
    print('CAM_SWEEP_PLAN_WRITTEN',flush=True)


if __name__=='__main__':main()
