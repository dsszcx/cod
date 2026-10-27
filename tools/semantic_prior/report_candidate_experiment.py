"""Local native-probability metrics and transparent candidate experiment report."""
import argparse
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import resolve
from tools.semantic_prior.pilot_metrics import raw_metrics, PRIMARY_METRICS, require_primary_metrics

LABELS = ('RGB', 'control direct', 'purity direct', 'control relation', 'purity relation')


def label(record):
    return 'RGB' if record['arm'] == 'rgb' else record['condition'] + ' ' + record['arm']


def case_figure(root, rgb_root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    inputs = root / 'metric_inputs'
    columns = ('control_rgb', 'control_relation', 'purity_relation')
    with np.load(inputs / 'targets.npz', allow_pickle=False) as targets:
        archives = [np.load(inputs / ('s20261022_' + name + '.npz'), allow_pickle=False) for name in columns]
        try:
            deltas = {key: float(np.abs(archives[2][key] - targets[key]).mean()
                                 - np.abs(archives[1][key] - targets[key]).mean()) for key in targets.files}
            ordered = sorted(deltas, key=deltas.get)
            chosen = list(dict.fromkeys(ordered[:2] + ordered[-2:][::-1]))
            fig, axes = plt.subplots(len(chosen), 5, figsize=(16, 3.2 * len(chosen)),
                                     constrained_layout=True, squeeze=False)
            for i, key in enumerate(chosen):
                paths = list(rgb_root.glob(key.replace('/', '__') + '.*'))
                if len(paths) != 1:
                    raise ValueError('Missing or ambiguous RGB for ' + key)
                with Image.open(paths[0]) as im:
                    rgb = np.asarray(im.convert('RGB'))
                arrays = [rgb, targets[key]] + [archive[key] for archive in archives]
                titles = ['Original\n' + key.split('/')[-1], 'GT', 'RGB model', 'Control relation', 'Purity relation']
                for j, array in enumerate(arrays):
                    if j == 0:
                        axes[i, j].imshow(array)
                    else:
                        axes[i, j].imshow(array, cmap='gray', vmin=0, vmax=1)
                    title = titles[j]
                    if j >= 2:
                        title += '\nMAE %.4f' % np.abs(array - targets[key]).mean()
                    axes[i, j].set_title(title, fontsize=7 if j == 0 else 9)
                    axes[i, j].axis('off')
            fig.suptitle('Seed22 best checkpoints: two smallest and two largest per-image MAE changes\n'
                         '(purity relation - control relation); raw probabilities fixed to [0,1]', fontsize=11)
            fig.savefig(root / 'CASE_REVIEW.png', dpi=140)
            plt.close(fig)
            C.write_json(str(root / 'CASE_SELECTION.json'), {
                'selection': 'post-hoc diagnostic; seed22 two lowest/two highest MAE changes, not training selection',
                'images': [{'image_id': key, 'mae_purity_minus_control': deltas[key]} for key in chosen]})
        finally:
            for archive in archives:
                archive.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--rgb-root', help='Optional existing RGB export; no new dataset access required')
    args = ap.parse_args()
    root = resolve(args.root)
    evaluation = C.read_json(str(root / 'EVALUATION.json'))
    plan = C.read_json(str(root / 'PLAN.json'))
    if evaluation['status'] != 'DONE' or len(evaluation['reports']) != plan['n_training_runs']:
        raise ValueError('A complete fixed plan is required')
    inputs = root / 'metric_inputs'
    runs, per_image = [], []
    with np.load(inputs / 'targets.npz', allow_pickle=False) as targets:
        for record in evaluation['reports']:
            with np.load(inputs / (record['name'] + '.npz'), allow_pickle=False) as archive:
                if set(archive.files) != set(targets.files):
                    raise ValueError('Validation denominator differs')
                rows = []
                for key in targets.files:
                    values = raw_metrics(archive[key], targets[key])
                    rows.append(values)
                    per_image.append({'name': record['name'], 'image_id': key, **values})
            averaged = {k: float(np.mean([m[k] for m in rows])) for k in rows[0]}
            require_primary_metrics(averaged)
            if abs(averaged['mae'] - record['mae']['full']) > 1e-7:
                raise ValueError('Native metrics differ from training/evaluation')
            runs.append({'seed': record['seed'], 'condition': record['condition'], 'arm': record['arm'],
                         'label': label(record), 'best_epoch': record['best_epoch'],
                         'last_mae': record['last_mae'], **averaged})
            print('CANDIDATE_RAW_METRICS_DONE', record['name'], averaged, flush=True)
        n_val = len(targets.files)
    C.write_csv(str(root / 'RAW_METRICS_PER_RUN.csv'), runs)
    C.write_csv(str(root / 'RAW_METRICS_PER_IMAGE.csv'), per_image)
    grouped = defaultdict(list)
    for row in runs:
        grouped[row['label']].append(row)
    averages = []
    lines = ['# 候选前景可靠性辅助监督：固定短程对照', '',
             '128 train / %d val，两种子，每组%d轮；所有分支重新训练，未使用官方测试集。' % (n_val, plan['epochs']), '',
             'control为原训练目标；purity额外以0.1权重监督直连门控的前景纯度。两者关系路径均不删除环境节点。', '',
             '## 分割结果', '',
             '| 条件 | seed22 MAE↓ | seed23 MAE↓ | mean best MAE↓ | mean last MAE↓ | S↑ | mean E↑ | wF↑ | Dice@0.5↑ |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in LABELS:
        rows = sorted(grouped[name], key=lambda r: r['seed'])
        if len(rows) != 2:
            raise ValueError('Both seeds must be present per condition')
        values = {k: float(np.mean([r[k] for r in rows])) for k in ('mae', 'last_mae', 'sm', 'mean_e', 'wfm', 'dice_05')}
        averages.append({'label': name, **values})
        lines.append('|%s|%.6f|%.6f|%.6f|%.6f|%.4f|%.4f|%.4f|%.4f|' %
                     (name, rows[0]['mae'], rows[1]['mae'], values['mae'], values['last_mae'],
                      values['sm'], values['mean_e'], values['wfm'], values['dice_05']))
    lines += ['', '## 直连门控与候选前景纯度（best checkpoint）', '',
              '这里检验门控是否更像前景含量估计，不等同于候选的因果辅助价值。MAE/MSE先按图内节点平均，再按有节点图平均。', '',
              '| seed | 条件 | 门控-纯度MAE↓ | MSE↓ | 高纯度门控均值 | 低纯度门控均值 | 有候选图/节点 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in evaluation['reports']:
        if r['arm'] == 'rgb':
            continue
        g = r['gate_calibration']
        def fmt(value):
            return '—' if value is None else '%.5f' % value
        lines.append('|%s|%s|%s|%s|%s|%s|%d/%d|' % (r['seed'], label(r), fmt(g['image_macro_mae']),
                     fmt(g['image_macro_mse']), fmt(g['foreground_gate_mean']), fmt(g['background_gate_mean']),
                     g['n_images'], g['n_nodes']))
    lines += ['', '## 同checkpoint依赖诊断', '',
              '| seed | 条件 | 无先验MAE增量 | 无关系MAE增量 |', '|---|---|---:|---:|']
    for r in evaluation['reports']:
        if r['arm'] == 'rgb':
            continue
        full = r['mae']['full']
        relation = '%+.7f' % (r['mae']['no_relation'] - full) if 'no_relation' in r['mae'] else '—'
        lines.append('|%s|%s|%+.7f|%s|' % (r['seed'], label(r), r['mae']['no_priors'] - full, relation))
    utility_path = root / 'RELATION_EDGE_UTILITY.json'
    if utility_path.exists():
        utility = C.read_json(str(utility_path))
        if utility['status'] != 'DONE':
            raise ValueError('Incomplete relation-edge diagnostic')
        lines += ['', '## 定向边功能诊断（不是语义关系标注）', '',
                  '每图按索引顺序最多取12条有效有向边，选择不读取GT。删除边后的MAE减去完整MAE为正，表示该边在这个模型中有帮助；删除也包含消息重归一化影响。', '',
                  '| 模型 | 检查边数 | 有帮助 (>1e-7) | 有害 (<-1e-7) | 近零 | 每图平均边效应，再按图平均 |',
                  '|---|---:|---:|---:|---:|---:|']
        for record in utility['reports']:
            per_image_delta = [np.mean([e['mae_removed_minus_full'] for e in image['edges']])
                               for image in record['images'] if image['edges']]
            macro = float(np.mean(per_image_delta)) if per_image_delta else 0.
            lines.append('|%s|%d|%d|%d|%d|%+.8f|' %
                         (record['name'], record['n_edges'], record['helpful_edges_gt_1e7'],
                          record['harmful_edges_lt_minus_1e7'], record['near_zero_edges'], macro))
    lines += ['', '## 解释限制', '',
              '- best由同一32张开发验证图选择，额外列last；两个种子不是64张独立验证图。',
              '- 本轮10轮与此前20轮不能直接混比；辅助目标拟合改善不自动代表COD改善。',
              '- 图像内多个候选/边共享图像和网络，不是独立统计样本。',
              '- 候选纯度不是主目标/辅助关系真值；环境仍可能通过关系路径提供帮助。',
              '- 依赖探针是固定模型的输入/分支扰动，不能代替重训练消融。',
              '- 原概率指标禁用逐图min-max；不与不同评估协议直接比较。']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    C.write_json(str(root / 'RAW_METRICS_SUMMARY.json'), {
        'per_run': runs, 'mean_per_condition': averages, 'normalize': False,
        'primary_metrics': list(PRIMARY_METRICS), 'checkpoint_selection_metric': 'mae',
        'n_unique_validation_images': n_val, 'test_evaluated': False,
        'pysodmetrics_version': version('pysodmetrics'), 'scipy_version': version('scipy'),
        'numpy_version': np.__version__})
    if args.rgb_root:
        case_figure(root, resolve(args.rgb_root))
    print('CANDIDATE_LOCAL_REPORT_DONE', flush=True)


if __name__ == '__main__':
    main()
