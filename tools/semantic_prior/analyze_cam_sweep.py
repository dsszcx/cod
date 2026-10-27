"""Inference dependency checks and portable raw-probability export for all arms."""
import gc
import argparse
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image
import torch

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import resolve
from tools.semantic_prior.training_core import build_model
from tools.semantic_prior.train_pilot import evaluate
from dataloader_relation import RelationDataset


class Probe:
    def __init__(self,model,mode):self.model,self.mode=model,mode
    def eval(self):self.model.eval()
    def __call__(self,**inp):
        inp=dict(inp)
        if self.mode=='no_priors':
            for key in ('region_valid','role_hint_valid','relation_valid'):inp[key]=torch.zeros_like(inp[key])
        elif self.mode=='no_role_hints':inp['role_hint_valid']=torch.zeros_like(inp['role_hint_valid'])
        elif self.mode=='zero_pair_geometry':inp['relation_geom']=torch.zeros_like(inp['relation_geom'])
        return self.model(**inp)


def probability_changes(full,changed,targets):
    all_changes=[];fg_changes=[];bg_changes=[];flips=[]
    for a,b in zip(full,changed):
        if a['image_id']!=b['image_id']:raise ValueError('Probe image ordering changed')
        diff=np.abs(a['prob']-b['prob']);fg=targets[a['image_id'].replace('/','__')]>=.5
        all_changes.append(diff)
        if fg.any():fg_changes.append(float(diff[fg].mean()))
        if (~fg).any():bg_changes.append(float(diff[~fg].mean()))
        flips.append(float(((a['prob']>=.5)!=(b['prob']>=.5)).mean()))
    return {'mean_absolute_probability_change':float(np.mean([d.mean() for d in all_changes])),
            'max_absolute_probability_change':float(max(d.max() for d in all_changes)),
            'mean_change_on_gt_foreground':float(np.mean(fg_changes)) if fg_changes else None,
            'mean_change_on_gt_background':float(np.mean(bg_changes)) if bg_changes else None,
            'threshold_05_flip_fraction':float(np.mean(flips))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--export-only',action='store_true',help='CPU export completed jobs while training continues; no inference')
    args=ap.parse_args()
    root=resolve('outputs/CAM_FUSION_SWEEP_20261026_V1')
    plan=C.read_json(str(root/'PLAN.json'))
    export=root/'metric_inputs';export.mkdir(exist_ok=True)
    manifests=resolve('outputs/RELATION_CAM_EXTENDED_20261026_V1')
    rows=C.read_csv(str(manifests/'pilot_val.csv'))
    targets={}
    for row in rows:
        key=row['image_id'].replace('/','__')
        with Image.open(row['gt_path']) as im:target=np.asarray(im.convert('L'),np.float32)
        if target.max()>1:target/=255.
        targets[key]=target
        destination=export/'RGB'/(key+Path(row['image_path']).suffix)
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():shutil.copy2(row['image_path'],destination)
    np.savez_compressed(export/'targets.npz',**targets)
    C.write_json(str(export/'VAL_INDEX.json'),[{**r,'metric_key':r['image_id'].replace('/','__')} for r in rows])
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    records=[];missing=[];reference={};reference_orders={};reference_source=None
    for job in plan['jobs']:
        cfg=C.load_config(job['config']);run=Path(job['out_dir'])
        summary=C.read_json(str(run/'SUMMARY.json'))
        if summary is None or summary.get('status')!='DONE':
            missing.append(job);continue
        source_info=C.read_json(str(run/'PROVENANCE.json'))
        fingerprint={k:source_info[k] for k in ('imagenet_sha256','manifest_sha256','code_sha256')}
        if reference_source is None:reference_source=fingerprint
        if fingerprint!=reference_source:raise ValueError('Cross-condition source or code changed')
        for record in summary['arms']:
            arm=record['arm'];tag='s%d_%s_%s'%(job['seed'],job['condition'],arm)
            epoch=record['best_epoch']
            per_image=C.read_json(str(run/arm/('epoch_%02d.json'%epoch)))['per_image']
            probs={r['metric_key']:np.load(run/arm/('val_epoch_%02d'%epoch)/(r['image_id'].split('/')[-1]+'.npy')) for r in C.read_json(str(export/'VAL_INDEX.json'))}
            np.savez_compressed(export/(tag+'.npz'),**probs)
            initialization=C.read_json(str(run/arm/'INITIALIZATION.json'))
            key=job['seed']
            if key in reference and initialization['rgb_initial_sha256']!=reference[key]:raise ValueError('Cross-condition RGB init mismatch')
            reference[key]=initialization['rgb_initial_sha256']
            for history in record['history']:
                epoch_record=C.read_json(str(run/arm/('epoch_%02d.json'%history['epoch'])))
                order_key=(key,history['epoch'])
                if order_key in reference_orders and reference_orders[order_key]!=epoch_record['sample_order']:
                    raise ValueError('Cross-condition sample-order mismatch')
                reference_orders[order_key]=epoch_record['sample_order']
            info={'seed':key,'condition':job['condition'],'arm':arm,'tag':tag,'best_epoch':epoch,
                  'best_mae':record['best_val_mae'],'last_mae':record['history'][-1]['val_native_mae'],
                  'optimizer_steps':record['optimizer_steps'],'elapsed_s':record['elapsed_s'],
                  'frozen_parameters_unchanged':record['frozen_parameters_unchanged'],
                  'history':record['history'],'per_image':per_image,'fusion_options':cfg['fusion_options']}
            diagnostic_path=run/arm/'INFERENCE_DIAGNOSTICS.json'
            if arm!='rgb' and not diagnostic_path.exists() and not args.export_only:
                model,_=build_model(arm,key,cfg['paths']['imagenet'],cfg['fusion_options'])
                state=torch.load(run/arm/'best.pth',map_location='cpu',weights_only=True)
                model.load_state_dict(state['model'],strict=True);model.cuda().eval()
                ds=RelationDataset(manifests/'pilot_val.csv','val',arm,cfg['input_size'],key)
                feature_stats=[]
                def feature_hook(module,arguments,output):
                    _,debug=module.forward(*arguments,return_debug=True)
                    denom=arguments[0].float().norm().clamp_min(1e-12)
                    feature_stats.append({'direct_relative_l2':float((module.alpha_direct*debug['direct_map']).float().norm()/denom),
                                          'relation_relative_l2':float((module.alpha_relation*debug['relation_map']).float().norm()/denom)})
                handle=model.region_fusion.register_forward_hook(feature_hook)
                full_mae,full=evaluate(Probe(model,'full'),ds,arm,torch.device('cuda:0'))
                handle.remove()
                stored_change=max(float(np.abs(d['prob']-probs[d['image_id'].replace('/','__')]).max()) for d in full)
                diagnostics={'full_mae':full_mae,'stored_mae':record['best_val_mae'],'max_abs_full_vs_stored':stored_change,'probes':{},
                    'checkpoint_alphas':[float(model.region_fusion.alpha_direct.detach()),float(model.region_fusion.alpha_relation.detach())],
                    'validation_residual_mean':{k:float(np.mean([s[k] for s in feature_stats])) for k in feature_stats[0]}}
                for mode in ('no_priors','no_role_hints','zero_pair_geometry'):
                    mae,changed=evaluate(Probe(model,mode),ds,arm,torch.device('cuda:0'))
                    diagnostics['probes'][mode]={'mae':mae,'mae_minus_full':mae-full_mae,
                        **probability_changes(full,changed,targets)}
                if arm=='relation':
                    with torch.no_grad():model.region_fusion.alpha_relation.zero_()
                    mae,changed=evaluate(Probe(model,'full'),ds,arm,torch.device('cuda:0'))
                    diagnostics['probes']['no_relation']={'mae':mae,'mae_minus_full':mae-full_mae,
                        **probability_changes(full,changed,targets)}
                C.write_json(str(diagnostic_path),diagnostics)
                del model,state;gc.collect();torch.cuda.empty_cache()
            info['inference_diagnostics']=C.read_json(str(diagnostic_path)) if arm!='rgb' else None
            records.append(info)
            print('SWEEP_ANALYZED',tag,'best',info['best_mae'],'last',info['last_mae'],flush=True)
    C.write_json(str(root/'SWEEP_ANALYSIS.json'),{'records':records,'missing_or_unfinished_jobs':missing,
        'all_14_runs_done':len(records)==14,'all_frozen_parameters_unchanged':all(r['frozen_parameters_unchanged'] for r in records),
        'inference_analysis_completed':not args.export_only,
        'cross_condition_rgb_initialization_equal_per_seed':True,'test_evaluated':False,
        'cross_condition_sample_orders_equal_per_seed':True,'cross_condition_sources_and_code_equal':True,
        'diagnostic_warning':'Same-checkpoint perturbations are inference dependency checks, not retraining causal ablations',
        'scope':'128 CAM-only training / 32 development validation; checkpoint selected on this same validation'})
    print('CAM_SWEEP_ANALYSIS_DONE',len(records),flush=True)


if __name__=='__main__':main()
