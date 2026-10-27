"""CPU-only four-metric report and exact detached-observer control audit."""
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

LABELS = ('control', 'observer', 'feedback', 'shared')


def audit_observer(root, evaluation):
    audits = []
    records = evaluation['reports']
    for seed in sorted({r['seed'] for r in records}):
        pair = [next(r for r in records if r['seed'] == seed and r['condition'] == name) for name in ('control', 'observer')]
        summaries = [C.read_json(str(root / ('s%d_%s' % (seed, name)) / 'SUMMARY.json'))['arms'][0]
                     for name in ('control', 'observer')]
        if summaries[0]['core_final_sha256'] != summaries[1]['core_final_sha256']:
            raise ValueError('Detached observer changed main-network final state')
        if len(summaries[0]['history']) != len(summaries[1]['history']):
            raise ValueError('Control and observer training lengths differ')
        for a, b in zip(summaries[0]['history'], summaries[1]['history']):
            for key in ('epoch', 'mean_segmentation_loss', 'val_native_mae', 'optimizer_steps'):
                if a[key] != b[key]:
                    raise ValueError('Detached observer changed main trajectory: ' + key)
        with np.load(root / 'metric_inputs' / (pair[0]['name'] + '.npz'), allow_pickle=False) as a, \
             np.load(root / 'metric_inputs' / (pair[1]['name'] + '.npz'), allow_pickle=False) as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[key], b[key]) for key in a.files):
                raise ValueError('Detached observer changed validation probabilities')
        audits.append({'seed': seed, 'same_core_final_state': True,
                       'same_segmentation_training_trajectory': True, 'same_full_probabilities': True})
    return audits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    args = ap.parse_args()
    root = resolve(args.root)
    plan = C.read_json(str(root / 'PLAN.json'))
    evaluation = C.read_json(str(root / 'EVALUATION.json'))
    if evaluation['status'] != 'DONE' or len(evaluation['reports']) != plan['n_training_runs']:
        raise ValueError('Incomplete fixed experiment')
    observer_audit = audit_observer(root, evaluation)
    runs, per_image, initializations = [], [], {}
    with np.load(root / 'metric_inputs' / 'targets.npz', allow_pickle=False) as targets:
        for record in evaluation['reports']:
            run = root / ('s%d_%s' % (record['seed'], record['condition']))
            initialization = C.read_json(str(run / 'relation' / 'INITIALIZATION.json'))
            common = (initialization['rgb_initial_sha256'], initialization['region_initial_sha256'],
                      initialization['quality_initial_sha256'])
            if initializations.setdefault(str(record['seed']), common) != common:
                raise ValueError('Unequal RGB/region/quality initialization')
            with np.load(root / 'metric_inputs' / (record['name'] + '.npz'), allow_pickle=False) as archive:
                if set(archive.files) != set(targets.files):
                    raise ValueError('Metric denominator changed')
                rows = []
                for key in targets.files:
                    metrics = raw_metrics(archive[key], targets[key])
                    rows.append(metrics)
                    per_image.append({'name': record['name'], 'image_id': key, **metrics})
            averaged = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
            require_primary_metrics(averaged)
            if abs(averaged['mae'] - record['mae']['full']) > 1e-7 or record['full_saved_max_abs_diff'] != 0:
                raise ValueError('Metrics/replay differ from saved probabilities')
            runs.append({'seed': record['seed'], 'condition': record['condition'],
                         'best_epoch': record['best_epoch'], 'last_mae': record['last_mae'],
                         'quality_mae': record['quality_calibration']['image_macro_mae'], **averaged})
            print('QUALITY_FOUR_METRICS_DONE', record['name'], {k: averaged[k] for k in PRIMARY_METRICS}, flush=True)
        n_val = len(targets.files)
    C.write_csv(str(root / 'RAW_METRICS_PER_RUN.csv'), runs)
    C.write_csv(str(root / 'RAW_METRICS_PER_IMAGE.csv'), per_image)
    grouped = defaultdict(list)
    for row in runs:
        grouped[row['condition']].append(row)
    averages = []
    lines = ['# 独立候选质量头：解耦与梯度隔离对照', '',
             '128 train / %d development val；两种子、四条件、各10epoch。未评估官方测试集。' % n_val, '',
             '- control：有同结构质量头但不训练、不反馈。',
             '- observer：质量头从detach后的视觉特征学习，仅观察，不改变COD。',
             '- feedback：在observer基础上，COD用可学习有符号适配器消费detach分数，可忽略它。',
             '- shared：无反馈，但质量损失允许影响共享图像特征。', '',
             '## 四项主指标与辅助结果', '',
             '| 条件 | seed22 MAE↓ | seed23 MAE↓ | mean MAE↓ | S↑ | mean E↑ | weighted F↑ | Dice@0.5↑ | mean last MAE↓ | 质量头纯度MAE↓ |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in LABELS:
        rows = sorted(grouped[name], key=lambda r: r['seed'])
        if len(rows) != 2:
            raise ValueError('Both seeds required')
        values = {k: float(np.mean([r[k] for r in rows])) for k in
                  ('mae', 'sm', 'mean_e', 'wfm', 'dice_05', 'last_mae', 'quality_mae')}
        averages.append({'condition': name, **values})
        lines.append('|%s|%.6f|%.6f|%.6f|%.4f|%.4f|%.4f|%.4f|%.6f|%.4f|' %
                     (name, rows[0]['mae'], rows[1]['mae'], values['mae'], values['sm'], values['mean_e'],
                      values['wfm'], values['dice_05'], values['last_mae'], values['quality_mae']))
    lines += ['', '## 质量估计与功能依赖分开检验', '',
              '| seed | 条件 | 质量纯度MAE↓ | 高纯度节点平均分 | 低纯度节点平均分 | 无质量反馈MAE增量 |',
              '|---|---|---:|---:|---:|---:|']
    for r in evaluation['reports']:
        q = r['quality_calibration']
        delta = ('%+.8f' % (r['mae']['no_quality_feedback'] - r['mae']['full'])
                 if 'no_quality_feedback' in r['mae'] else '—')
        lines.append('|%s|%s|%.5f|%.5f|%.5f|%s|' %
                     (r['seed'], r['condition'], q['image_macro_mae'],
                      q['foreground_gate_mean'], q['background_gate_mean'], delta))
    lines += ['', '## 精确观察者对照', '',
              '两个种子的observer均与control具有完全相同的COD最终参数/缓冲区、逐轮分割损失与验证MAE、完整验证概率图；辅助质量头单独优化与裁剪，未污染主网络。', '',
              '## 限制', '',
              '- 所有四指标来自同一个best-MAE checkpoint；另列last，不能为每个指标分别挑轮次。',
              '- 同一32张开发验证集用于选best；两种子和同图候选不构成独立大样本。',
              '- 质量目标是前景含量，不是辅助价值或语义关系标签；低前景节点仍保留关系通路。',
              '- shared只检验无反馈情况下的共享辅助梯度；本轮没有完整交叉测试共享梯度与反馈的交互。',
              '- 无质量反馈是同checkpoint扰动，不代替重新训练对照；不把微小变化夸大成收益。',
              '- 旧纯度直接监督门控实验仅作历史参考，不与不同训练协议混比。']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    C.write_json(str(root / 'RAW_METRICS_SUMMARY.json'), {
        'status': 'DONE', 'per_run': runs, 'mean_per_condition': averages,
        'observer_audit': observer_audit, 'primary_metrics': list(PRIMARY_METRICS),
        'normalize': False, 'checkpoint_selection_metric': 'mae',
        'n_unique_validation_images': n_val, 'test_evaluated': False,
        'versions': {p: version(p) for p in ('numpy', 'scipy', 'pysodmetrics', 'scikit-image', 'scikit-learn')}})
    print('QUALITY_DECOUPLING_REPORT_DONE', flush=True)


if __name__ == '__main__':
    main()
