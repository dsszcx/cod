"""Post-training same-validation diagnostic, no retraining/threshold selection."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import resolve
from tools.semantic_prior.training_core import build_model, forward_arm
from tools.semantic_prior.train_pilot import evaluate
from dataloader_relation import RelationDataset


class DiagnosticModel:
    def __init__(self, model, mode):
        self.model,self.mode=model,mode
    def eval(self):
        self.model.eval()
    def __call__(self, **inputs):
        if self.mode=="no_priors":
            inputs=dict(inputs,region_valid=torch.zeros_like(inputs["region_valid"]),
                        role_hint_valid=torch.zeros_like(inputs["role_hint_valid"]),
                        relation_valid=torch.zeros_like(inputs["relation_valid"]))
        return self.model(**inputs)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/relation_smoke_20261026.yaml')
    args=ap.parse_args()
    cfg=C.load_config(str(resolve(args.config)))
    root=resolve(cfg['paths']['out_dir'])
    summary=C.read_json(str(root/'SUMMARY.json'))
    if summary['status']!='DONE': raise ValueError('Training did not finish')
    target=root/'ANALYSIS_V2.json'
    if target.exists(): raise FileExistsError(str(target))
    torch.set_num_threads(4)
    if cfg.get('deterministic_algorithms',False):
        torch.use_deterministic_algorithms(True,warn_only=True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0')
    comparisons,per_image,diagnostics=[],{},{}
    for arm in cfg['arms']:
        record=next(a for a in summary['arms'] if a['arm']==arm)
        epoch=record['best_epoch']
        details=C.read_json(str(root/arm/('epoch_%02d.json'%epoch)))['per_image']
        per_image[arm]={d['image_id']:d for d in details}
        positives=[d['mae'] for d in details if not d['gt_empty']]
        negatives=[d['mae'] for d in details if d['gt_empty']]
        comparisons.append({'arm':arm,'best_epoch':epoch,'val_mae':record['best_val_mae'],
            'nonempty_gt_n':len(positives),'nonempty_gt_mae':float(np.mean(positives)) if positives else None,
            'empty_gt_n':len(negatives),'empty_gt_mae':float(np.mean(negatives)) if negatives else None,
            'last_alphas':record['history'][-1]['alphas'],'optimizer_steps':record['optimizer_steps']})
        if arm=='rgb': continue
        state=torch.load(root/arm/'best.pth',map_location='cpu',weights_only=True)
        model,_=build_model(arm,cfg['seed'],str(resolve(cfg['paths']['imagenet'])),cfg.get('fusion_options'))
        model.load_state_dict(state['model'],strict=True)
        model.to(device).eval()
        ds=RelationDataset(resolve(cfg['paths']['manifests'])/'pilot_val.csv','val',arm,
                           cfg['input_size'],cfg['seed'])
        full_mae,full_details=evaluate(DiagnosticModel(model,'full'),ds,arm,device)
        mae,zero_details=evaluate(DiagnosticModel(model,'no_priors'),ds,arm,device)
        changes=[np.abs(a['prob']-b['prob']) for a,b in zip(full_details,zero_details)]
        diagnostics[arm]={'no_priors_val_mae':mae,'full_val_mae':full_mae,
                          'stored_best_val_mae':record['best_val_mae'],
                          'full_reevaluation_difference':full_mae-record['best_val_mae'],
                          'change_when_priors_disabled':mae-full_mae,
                          'mean_absolute_probability_change':float(np.mean([d.mean() for d in changes])),
                          'max_absolute_probability_change':float(max(d.max() for d in changes)),
                          'checkpoint_alphas':[float(model.region_fusion.alpha_direct.detach()),float(model.region_fusion.alpha_relation.detach())]}
        if arm=='relation':
            with torch.no_grad(): model.region_fusion.alpha_relation.zero_()
            no_relation,no_rel_details=evaluate(DiagnosticModel(model,'full'),ds,arm,device)
            diagnostics[arm]['relation_branch_disabled_val_mae']=no_relation
            diagnostics[arm]['mean_absolute_probability_change_no_relation']=float(np.mean([np.abs(a['prob']-b['prob']).mean() for a,b in zip(full_details,no_rel_details)]))
        del model,state
        torch.cuda.empty_cache()
        print('DIAGNOSTIC_DONE',arm,diagnostics[arm],flush=True)
    if len({tuple(sorted(p)) for p in per_image.values()})!=1: raise ValueError('Different validation IDs')
    deltas=[]
    for name in sorted(per_image['rgb']):
        a,b,c=[per_image[arm][name]['mae'] for arm in cfg['arms']]
        deltas.append({'image_id':name,'gt_empty':per_image['rgb'][name]['gt_empty'],
                       'rgb_mae':a,'direct_mae':b,'relation_mae':c,
                       'relation_minus_direct':c-b,'relation_minus_rgb':c-a})
    deltas.sort(key=lambda r:r['relation_minus_direct'])
    report={'comparison':comparisons,'same_model_inference_diagnostics':diagnostics,
            'per_image_deltas':deltas,'n_val':summary['n_val'],'training_repeated':False,'test_set_used':False,
            'data_scope':cfg.get('data_scope','historical_mixed'),
            'epochs':cfg['epochs'],'inference_settings_matched_to_training':True,
            'claim':'Same pilot validation used for checkpoint choice. No independent performance claim.'}
    C.write_json(str(target),report)
    C.write_csv(str(root/'PER_IMAGE_DELTAS_V2.csv'),deltas)
    lines=['# %d/%d图三组短训练结果'%(summary['n_train'],summary['n_val']),'',
           '仅用于检验训练流程，不是独立测试或稳定收益证据。每组%d epoch，验证集也用于选checkpoint。'%cfg['epochs'],'',
           '| 组 | 最优epoch | 验证MAE | 非空GT MAE | 空GT MAE | 更新步数 |',
           '|---|---:|---:|---:|---:|---:|']
    for r in comparisons:
        fmt=lambda v:'—' if v is None else '%.6f'%v
        lines.append('|%s|%s|%s|%s|%s|%s|'%(r['arm'],r['best_epoch'],fmt(r['val_mae']),fmt(r['nonempty_gt_mae']),fmt(r['empty_gt_mae']),r['optimizer_steps']))
    lines+=['','## 已训练同一模型关闭先验的诊断','',str(diagnostics),'',
            '这些诊断不重新训练，不能替代独立消融或证明分支因果贡献。','',
            '## 与direct相比变差最多的案例','']
    lines += ['- %s：MAE差%+.6f'%(r['image_id'],r['relation_minus_direct']) for r in deltas[-3:][::-1]]
    (root/'RESULTS_V2.md').write_text('\n'.join(lines),encoding='utf-8')
    print('RELATION_PILOT_ANALYSIS_DONE',flush=True)


if __name__=='__main__':main()
