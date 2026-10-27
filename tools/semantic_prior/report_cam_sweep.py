"""Local CPU metrics and scientific figures from exported native probabilities."""
from collections import defaultdict
import argparse
from importlib.metadata import version
from pathlib import Path
import sys
import numpy as np
from PIL import Image

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import resolve
from tools.semantic_prior.pilot_metrics import raw_metrics, PRIMARY_METRICS, require_primary_metrics

LABELS=('RGB','raw_zero direct','raw_zero relation','raw_small direct','raw_small relation','scaled_small direct','scaled_small relation')


def label(record):return 'RGB' if record['arm']=='rgb' else record['condition']+' '+record['arm']


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='outputs/CAM_FUSION_SWEEP_20261026_V1')
    root=resolve(ap.parse_args().root)
    analysis=C.read_json(str(root/'SWEEP_ANALYSIS.json'))
    inputs=root/'metric_inputs'
    targets=np.load(inputs/'targets.npz',allow_pickle=False)
    index=C.read_json(str(inputs/'VAL_INDEX.json'))
    summary=[];per_image=[];preds={}
    for record in analysis['records']:
        archive=np.load(inputs/(record['tag']+'.npz'),allow_pickle=False)
        if set(archive.files)!=set(targets.files):raise ValueError('Metric denominator mismatch')
        metrics=[]
        for key in targets.files:
            values=raw_metrics(archive[key],targets[key]);metrics.append(values)
            per_image.append({'tag':record['tag'],'key':key,**values})
        averaged={k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
        require_primary_metrics(averaged)
        if abs(averaged['mae']-record['best_mae'])>1e-7:raise ValueError('Exported probabilities disagree with training MAE')
        summary.append({'seed':record['seed'],'label':label(record),'best_epoch':record['best_epoch'],
            'last_mae':record['last_mae'],'n_val':len(metrics),**averaged})
        if record['seed']==20261022:preds[label(record)]={k:archive[k] for k in archive.files}
        archive.close()
        print('RAW_METRICS_DONE',record['tag'],averaged,flush=True)
    C.write_csv(str(root/'RAW_METRICS_PER_RUN.csv'),summary)
    C.write_csv(str(root/'RAW_METRICS_PER_IMAGE.csv'),per_image)
    grouped=defaultdict(list)
    for row in summary:grouped[row['label']].append(row)
    lines=['# CAM-only 128/32：两种子、七条件、20epoch','',
        '已导入%d/14次完成的训练；%s。'%(len(analysis['records']),'训练与推理诊断均完成' if analysis.get('all_14_runs_done') and analysis.get('inference_analysis_completed') else '阶段性产物，尚非最终报告'),'',
        '开发验证集同时用于选择best checkpoint；不是正式测试，所有结构指标显式禁用逐图min-max。',
        '下表同时列出两个种子的MAE，不能把同一32张验证图当作64张独立样本。','',
        '| 条件 | seed22 MAE↓ | seed23 MAE↓ | mean MAE↓ | mean last MAE↓ | S↑ | mean E↑ | wF↑ | Dice@0.5↑ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    aggregates=[]
    for name in LABELS:
        rows=sorted(grouped.get(name,[]),key=lambda r:r['seed'])
        if not rows:continue
        values={k:float(np.mean([r[k] for r in rows])) for k in ('mae','last_mae','sm','mean_e','wfm','dice_05')}
        aggregates.append({'label':name,'n_seeds':len(rows),**values})
        numbers=[('%.6f'%r['mae']) for r in rows]
        while len(numbers)<2:numbers.append('—')
        lines.append('|%s|%s|%s|%.6f|%.6f|%.4f|%.4f|%.4f|%.4f|'%(name,*numbers,values['mae'],values['last_mae'],values['sm'],values['mean_e'],values['wfm'],values['dice_05']))
    lines+=['','## 同一checkpoint去掉先验（不是重训练消融）','',
        '| seed | 条件 | 完整MAE | 无先验MAE | 无先验-完整 | 平均概率绝对变化 |','|---|---|---:|---:|---:|---:|']
    for record in analysis['records']:
        diagnostic=record['inference_diagnostics']
        if diagnostic is None:continue
        probe=diagnostic['probes']['no_priors']
        lines.append('|%s|%s|%.6f|%.6f|%+.6f|%.3g|'%(record['seed'],label(record),diagnostic['full_mae'],probe['mae'],probe['mae_minus_full'],probe['mean_absolute_probability_change']))
    lines+=['','## 解释限制','',
        '- 两种子仅检验初始化敏感性，不代表完整统计显著性；best来自同一开发验证集，最后一轮另列。',
        '- 关闭先验、角色、几何或关系分支是分布扰动与依赖诊断，不能替代相应训练对照。',
        '- 新的CAM-only数字不能与含NonCAM的旧混合MAE直接比较；候选GT oracle也不是模型结果。',
        '- raw概率评估口径与默认逐图min-max的旧指标或部分文献口径不同，不作跨协议数值比较。',
        '- 本轮不评估NonCAM拒识能力，也未使用官方测试集。']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    C.write_json(str(root/'RAW_METRICS_SUMMARY.json'),{'per_run':summary,'mean_per_condition':aggregates,
        'primary_metrics':list(PRIMARY_METRICS),'checkpoint_selection_metric':'mae',
        'pysodmetrics_version':version('pysodmetrics'),'scipy_version':version('scipy'),'numpy_version':np.__version__,
        'normalize':False,'probability_source':'native-size saved best checkpoint raw sigmoid float32',
        'gt_structural_threshold':.5,'test_evaluated':False,'n_unique_validation_images':len(targets.files)})
    if len(analysis['records'])==14:
        figures(root,analysis['records'],index,targets,preds)
    targets.close()
    print('CAM_SWEEP_LOCAL_REPORT_DONE',flush=True)


def figures(root,records,index,targets,preds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zip(LABELS,('black','#7699c0','#205080','#dc9c65','#b9561b','#68ae87','#116b43')))
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True,sharey=True)
    for ax,seed in zip(axes,(20261022,20261023)):
        for record in records:
            if record['seed']!=seed:continue
            name=label(record);history=record['history']
            ax.plot([h['epoch'] for h in history],[h['val_native_mae'] for h in history],label=name,
                    color=colors[name],linestyle='--' if record['arm']=='direct' else '-',linewidth=1.3)
        ax.set_title('Seed %d: SAME 32 CAM validation images'%seed)
        ax.set_xlabel('Epoch');ax.set_ylabel('Native raw-probability MAE');ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.savefig(root/'LEARNING_CURVES.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True,sharey=True)
    for ax,seed in zip(axes,(20261022,20261023)):
        for record in records:
            if record['seed']!=seed or record['arm']!='relation':continue
            stats=[h['first_batch_fusion_stats'] for h in record['history']]
            y=[s.get('relation_relative_l2',0) for s in stats]
            y=[v if v>0 else np.nan for v in y]
            ax.semilogy(range(1,len(y)+1),y,color=colors[label(record)],label=record['condition'])
        ax.set_title('Seed %d: relation residual / RGB feature L2'%seed)
        ax.set_xlabel('Epoch (first batch only)');ax.set_ylabel('Relative L2 norm');ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.savefig(root/'RELATION_RESIDUAL_SCALE.png',dpi=150);plt.close(fig)
    if 'scaled_small relation' not in preds or 'RGB' not in preds:return
    delta={k:float(np.abs(preds['scaled_small relation'][k]-targets[k]).mean()-np.abs(preds['RGB'][k]-targets[k]).mean()) for k in targets.files}
    ordered=sorted(delta,key=delta.get);chosen=list(dict.fromkeys(ordered[:2]+ordered[-2:][::-1]))
    columns=('RGB','raw_zero relation','scaled_small direct','scaled_small relation')
    fig,axes=plt.subplots(len(chosen),6,figsize=(18,3.2*len(chosen)),constrained_layout=True,squeeze=False)
    by_key={r['metric_key']:r for r in index}
    for i,key in enumerate(chosen):
        candidates=list((root/'metric_inputs'/'RGB').glob(key+'.*'))
        if len(candidates)!=1:raise ValueError('Case RGB missing/ambiguous')
        with Image.open(candidates[0]) as im:rgb=np.asarray(im.convert('RGB'))
        images=[rgb,targets[key]]+[preds[name][key] for name in columns]
        for j,array in enumerate(images):
            if j==0:axes[i,j].imshow(array)
            else:axes[i,j].imshow(array,cmap='gray',vmin=0,vmax=1)
            title=('Original','GT',*columns)[j]
            if j>=2:title+='\nMAE %.4f'%np.abs(array-targets[key]).mean()
            axes[i,j].set_title(title,fontsize=9);axes[i,j].axis('off')
        axes[i,0].set_title('Original\n'+by_key[key]['image_id'].split('/')[-1],fontsize=7)
    fig.suptitle('Seed22: two lowest and two highest MAE deltas (scaled relation - RGB); fixed probabilities [0,1]',fontsize=12)
    fig.savefig(root/'CASE_REVIEW.png',dpi=140);plt.close(fig)


if __name__=='__main__':main()
