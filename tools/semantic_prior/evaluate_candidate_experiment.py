"""Evaluate fixed development checkpoints; GT is only used after forward.

Exports native raw sigmoid maps for local PySODMetrics, plus gate calibration
and same-checkpoint prior/relationship dependency probes. No test data read.
"""
import argparse
import gc
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import resolve
from tools.semantic_prior.training_core import build_model, forward_arm
from tools.semantic_prior.candidate_supervision import candidate_targets
from dataloader_relation import RelationDataset, collate_relation_batch
from tools.semantic_prior.evaluation_runtime import configure_evaluation, prepare_export_directory


def calibration_summary(rows):
    nonempty = [r for r in rows if r['purity']]
    if not nonempty:
        return {'n_images': 0, 'n_nodes': 0}
    y = np.concatenate([r['purity'] for r in nonempty])
    p = np.concatenate([r['gates'] for r in nonempty])
    correlation = float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else None
    return {'n_images': len(nonempty), 'n_nodes': len(y),
            'image_macro_mae': float(np.mean([np.abs(np.asarray(r['purity']) - r['gates']).mean() for r in nonempty])),
            'image_macro_mse': float(np.mean([((np.asarray(r['purity']) - r['gates']) ** 2).mean() for r in nonempty])),
            'constant_half_image_macro_mae': float(np.mean([np.abs(np.asarray(r['purity']) - .5).mean() for r in nonempty])),
            'node_micro_pearson': correlation,
            'foreground_gate_mean': float(p[y >= .75].mean()) if (y >= .75).any() else None,
            'background_gate_mean': float(p[y <= .25].mean()) if (y <= .25).any() else None,
            'n_foreground_nodes': int((y >= .75).sum()), 'n_background_nodes': int((y <= .25).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    args = ap.parse_args()
    plan_path = resolve(args.plan)
    plan = C.read_json(str(plan_path))
    root = plan_path.parent
    target_path = root / 'EVALUATION.json'
    if target_path.exists():
        raise FileExistsError(str(target_path))
    export = prepare_export_directory(root / 'metric_inputs')
    reports, native_targets, consistency, runtime_settings = [], {}, {}, {}
    for job in plan['jobs']:
        cfg = C.load_config(job['config'])
        runtime_settings[Path(job['out_dir']).name] = configure_evaluation(cfg)
        run_root = Path(job['out_dir'])
        summary = C.read_json(str(run_root / 'SUMMARY.json'))
        if summary['status'] != 'DONE':
            raise ValueError('Only completed, fixed-length runs may be compared')
        for arm in job['arms']:
            arm_root = run_root / arm
            initialization = C.read_json(str(arm_root / 'INITIALIZATION.json'))
            key = str(job['seed'])
            reference = consistency.setdefault(key, {'rgb': initialization['rgb_initial_sha256'], 'orders': {}})
            if reference['rgb'] != initialization['rgb_initial_sha256']:
                raise ValueError('Cross-condition RGB initialization differs')
            if arm != 'rgb':
                reference.setdefault('region', initialization['region_initial_sha256'])
                if reference['region'] != initialization['region_initial_sha256']:
                    raise ValueError('Cross-condition region initialization differs')
            for epoch in range(1, cfg['epochs'] + 1):
                log = C.read_json(str(arm_root / ('epoch_%02d.json' % epoch)))
                order = log['sample_order']
                if reference['orders'].setdefault(str(epoch), order) != order:
                    raise ValueError('Cross-condition sample order differs')
            checkpoint = torch.load(arm_root / 'best.pth', map_location='cpu', weights_only=True)
            model, _ = build_model(arm, cfg['seed'], cfg['paths']['imagenet'], cfg.get('fusion_options'), cfg.get('quality_options'))
            model.load_state_dict(checkpoint['model'], strict=True)
            model.cuda().eval()
            dataset = RelationDataset(Path(cfg['paths']['manifests']) / 'pilot_val.csv',
                                      'val', arm, cfg['input_size'], cfg['seed'])
            modes = ['full'] if arm == 'rgb' else ['full', 'no_priors']
            if arm == 'relation':
                modes.append('no_relation')
            if cfg.get('quality_options', {}).get('feedback', False):
                modes.append('no_quality_feedback')
            per_image, calibration, probabilities, max_saved_diff = [], [], {}, 0.
            quality_calibration = []
            with torch.inference_mode():
                for index, row in enumerate(dataset.rows):
                    item = dataset[index]
                    batch = collate_relation_batch([item])
                    inp = {k: v.cuda() for k, v in batch['inputs'].items()}
                    gt_augmented = batch['gt'].cuda()
                    with Image.open(row['gt_path']) as im:
                        gt = np.asarray(im.convert('L'), np.float32)
                    if gt.max() > 1:
                        gt /= 255.
                    identity = row['image_id']
                    if identity in native_targets and not np.array_equal(native_targets[identity], gt):
                        raise ValueError('GT changed across conditions')
                    native_targets[identity] = gt
                    scores, full = {}, None
                    for mode in modes:
                        current = dict(inp)
                        if mode == 'no_priors':
                            for name in ('region_valid', 'role_hint_valid', 'relation_valid'):
                                current[name] = torch.zeros_like(current[name])
                        if mode == 'no_relation':
                            saved_alpha = model.region_fusion.alpha_relation.clone()
                            model.region_fusion.alpha_relation.zero_()
                        if mode == 'no_quality_feedback':
                            model.region_fusion.use_quality_feedback = False
                        if arm != 'rgb' and mode == 'full':
                            outputs, aux = model(**current, return_aux=True)
                            labels = candidate_targets(inp['region_masks'], aux['effective_valid'], gt_augmented)
                            valid = labels['valid']
                            calibration.append({'image_id': identity,
                                                'purity': labels['purity'][valid].cpu().tolist(),
                                                'gates': aux['direct_gates'][valid].cpu().tolist()})
                            if 'quality_scores' in aux:
                                quality_calibration.append({'image_id': identity,
                                    'purity': labels['purity'][valid].cpu().tolist(),
                                    'gates': aux['quality_scores'][valid].cpu().tolist()})
                        else:
                            outputs = forward_arm(model, current, arm)
                        if mode == 'no_relation':
                            model.region_fusion.alpha_relation.copy_(saved_alpha)
                        if mode == 'no_quality_feedback':
                            model.region_fusion.use_quality_feedback = True
                        prob = F.interpolate(outputs[-1].float(), size=item['native_hw'],
                                             mode='bilinear', align_corners=False).sigmoid()[0, 0].cpu().numpy()
                        scores[mode + '_mae'] = float(np.abs(prob - gt).mean())
                        if mode == 'full':
                            full = prob
                            probabilities[identity] = prob
                            saved = np.load(arm_root / ('val_epoch_%02d' % checkpoint['epoch']) /
                                            (identity.split('/')[-1] + '.npy'), allow_pickle=False)
                            max_saved_diff = max(max_saved_diff, float(np.abs(saved - prob).max()))
                        else:
                            scores[mode + '_probability_change'] = float(np.abs(prob - full).mean())
                    per_image.append({'image_id': identity, **scores})
            mae = {mode: float(np.mean([r[mode + '_mae'] for r in per_image])) for mode in modes}
            if abs(mae['full'] - checkpoint['val_native_mae']) > 1e-7 or max_saved_diff != 0:
                raise ValueError('Full re-evaluation differs from stored native probabilities: '
                                 'seed=%s condition=%s arm=%s mae_delta=%.12g max_pixel_diff=%.12g' %
                                 (job['seed'], job['condition'], arm,
                                  mae['full'] - checkpoint['val_native_mae'], max_saved_diff))
            name = 's%d_%s_%s' % (job['seed'], job['condition'], arm)
            np.savez_compressed(export / (name + '.npz'), **probabilities)
            report = {'name': name, 'seed': job['seed'], 'condition': job['condition'], 'arm': arm,
                      'candidate_reliability_weight': cfg['candidate_reliability_weight'],
                      'best_epoch': checkpoint['epoch'], 'mae': mae,
                      'last_mae': summary['arms'][job['arms'].index(arm)]['history'][-1]['val_native_mae'],
                      'full_saved_max_abs_diff': max_saved_diff,
                      'gate_calibration': calibration_summary(calibration),
                      'quality_options': cfg.get('quality_options'),
                      'quality_supervision_weight': cfg.get('quality_supervision_weight', 0.),
                      'quality_calibration': calibration_summary(quality_calibration),
                      'quality_per_image': quality_calibration,
                      'quality_adapter_weights': ({name: getattr(model.region_fusion, name).weight.detach().cpu().tolist()
                                                   for name in ('quality_to_direct', 'quality_to_relation', 'quality_to_edges')}
                                                  if cfg.get('quality_options') is not None else None),
                      'gate_per_image': calibration, 'per_image': per_image}
            reports.append(report)
            C.write_json(str(root / (name + '_DIAGNOSTICS.json')), report)
            print('CANDIDATE_EVALUATION_ARM_DONE', name, mae, report['gate_calibration'], flush=True)
            del model, checkpoint, outputs
            if arm != 'rgb':
                del aux
            gc.collect()
            torch.cuda.empty_cache()
    np.savez_compressed(export / 'targets.npz', **native_targets)
    C.write_json(str(target_path), {'status': 'DONE', 'reports': reports,
                                  'evaluation_runtime': runtime_settings,
                                  'evaluation_code_sha256': {p: C.sha256_file(str(resolve(p))) for p in
                                      ('tools/semantic_prior/evaluate_candidate_experiment.py',
                                       'tools/semantic_prior/evaluation_runtime.py')},
                                  'same_initialization_and_orders': True, 'test_evaluated': False,
                                  'gt_used_in_model_forward': False,
                                  'limitations': 'Same development validation selects best; inference probes are not retraining ablations'})
    print('CANDIDATE_EVALUATION_DONE', flush=True)


if __name__ == '__main__':
    main()
