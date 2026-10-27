"""Four-metric, dual-protocol report on development validation only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import sys

import numpy as np
from PIL import Image

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.official_sinetv2.metrics import evaluate, prediction_to_png
from tools.semantic_prior.pilot_metrics import raw_metrics, require_primary_metrics
from tools.semantic_prior.prepare_cam_only import category


def score_one(item):
    row, pred_root = item
    prob = np.load(Path(pred_root)/(row['image_id'].split('/')[-1]+'.npy'), allow_pickle=False)
    with Image.open(row['gt_path']) as image:
        gt = np.array(image.convert('L'))
    if not np.isin(gt, [0, 255]).all():
        raise ValueError('Nonbinary GT')
    raw = raw_metrics(prob, gt.astype(np.float32)/255)
    standard = evaluate(prediction_to_png(prob), gt)
    require_primary_metrics(raw)
    require_primary_metrics(standard)
    return dict(image_id=row['image_id'], dataset=category(row), raw=raw, png_standard=standard)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root
    cfg = C.load_config(str(root/'CONFIG.yaml'))
    summary = C.read_json(str(root/'training/SUMMARY.json'))
    if summary['status'] != 'DONE' or summary['n_train'] != 3652 or summary['n_val'] != 383 or summary['epochs'] != 1:
        raise ValueError('Incomplete acceptance run')
    arm = summary['arms'][0]
    pred_root = root/'training/rgb'/('val_epoch_%02d' % arm['best_epoch'])
    rows = C.read_csv(str(Path(cfg['paths']['manifests'])/'val.csv'))
    expected = {r['image_id'].split('/')[-1]+'.npy' for r in rows}
    if {p.name for p in pred_root.glob('*.npy')} != expected:
        raise ValueError('Incomplete prediction coverage')
    scored = []
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn')) as pool:
        for index, result in enumerate(pool.map(score_one, [(r, str(pred_root)) for r in rows], chunksize=4), 1):
            scored.append(result)
            if index % 100 == 0 or index == len(rows):
                print('RGB_FULL_DATA_METRICS', index, '/', len(rows), flush=True)
    metrics = {}
    for protocol in ['raw', 'png_standard']:
        groups = {}
        for name in ['all', 'CAMO', 'COD10K-CAM']:
            subset = [r for r in scored if name == 'all' or r['dataset'] == name]
            if not subset:
                raise ValueError('Empty metric group')
            groups[name] = dict(n_images=len(subset), **{
                k: float(np.mean([r[protocol][k] for r in subset])) for k in ['mae', 'sm', 'mean_e', 'wfm']})
        metrics[protocol] = groups
    if abs(metrics['raw']['all']['mae']-arm['best_val_mae']) > 1e-7:
        raise ValueError('Saved probabilities do not reproduce selected validation MAE')
    C.write_json(str(root/'METRICS.json'), dict(status='DONE', n_train=3652, n_val=383,
        epoch=1, optimizer_steps=arm['optimizer_steps'], test_set_used=False,
        claim='Engineering acceptance only; not a converged baseline or a SINet comparison',
        standard_protocol='Same float->uint8 quantization and metric functions as verified official SINet adapter',
        metrics=metrics))
    C.write_json(str(root/'PER_IMAGE.json'), scored)
    text = ['# 全数据 RGB 一轮工程验收', '',
            '训练 3652 张，开发验证 383 张；仅 1 epoch，不代表收敛，不与官方测试集/论文数字直接比较。', '',
            '| 协议 / 开发集 | N | S↑ | mean E↑ | weighted F↑ | MAE↓ |',
            '|---|---:|---:|---:|---:|---:|']
    for protocol, groups in metrics.items():
        for name, v in groups.items():
            text.append('| %s / %s | %d | %.6f | %.6f | %.6f | %.6f |' %
                        (protocol, name, v['n_images'], v['sm'], v['mean_e'], v['wfm'], v['mae']))
    text += ['', 'raw 为既有内部概率评估；png_standard 为已核验的官方兼容量化/归一化评估。两套协议不混排比较。',
             'PNG 协议在内存中执行相同 uint8 序列化量化；原始概率仍保存在服务器。',
             '没有生成 Qwen/SAM3 先验，没有使用官方 SINet 权重训练，没有推理官方测试集。',
             '下一步仍需预先固定正式训练轮次/学习率计划，并完成同训练预算的 RGB/直连/关系对照。']
    (root/'RESULTS.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    print('RGB_FULL_DATA_ACCEPTANCE_DONE', flush=True)


if __name__ == '__main__':
    main()
