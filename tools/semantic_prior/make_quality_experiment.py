"""Predeclare gradient-isolation and optional score-feedback comparisons."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import yaml

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import resolve

CONDITIONS = (
    ('control', 0., True, False),
    ('observer', .1, True, False),
    ('feedback', .1, True, True),
    ('shared', .1, False, False),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--manifests', required=True)
    ap.add_argument('--imagenet', required=True)
    args = ap.parse_args()
    root = resolve(args.out)
    root.mkdir(exist_ok=False, parents=True)
    base = {'version': 'quality_decoupling_v1', 'data_scope': 'cam_only',
            'expected_train_images': 128, 'expected_val_images': 32, 'epochs': 10,
            'input_size': 352, 'batch_size': 2, 'learning_rate': 1e-4, 'weight_decay': 1e-4,
            'workers': 0, 'amp': False, 'node_geometry': False,
            'deterministic_algorithms': True, 'collect_fusion_stats': True,
            'deadline_utc': (datetime.now(timezone.utc) + timedelta(minutes=50)).isoformat(),
            'candidate_reliability_weight': 0., 'arms': ['relation'],
            'fusion_options': {'alpha_init': .01, 'projection_normalization': 'feature_rms'}}
    jobs = []
    for seed in (20261022, 20261023):
        for condition, weight, stop_gradient, feedback in CONDITIONS:
            out = root / ('s%d_%s' % (seed, condition))
            cfg = {**base, 'seed': seed, 'quality_supervision_weight': weight,
                   'quality_options': {'stop_gradient': stop_gradient, 'feedback': feedback},
                   'paths': {'manifests': str(resolve(args.manifests)),
                             'imagenet': str(resolve(args.imagenet)), 'out_dir': str(out)}}
            config = root / (out.name + '.yaml')
            config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
            jobs.append({'seed': seed, 'condition': condition, 'arms': ['relation'],
                         'config': str(config), 'out_dir': str(out)})
    C.write_json(str(root / 'PLAN.json'), {
        'created_utc': C.utc_now(), 'jobs': jobs, 'n_training_runs': 8,
        'epochs': 10, 'n_train': 128, 'n_val': 32, 'optimizer_updates_total': 5120,
        'primary_metrics': ['mae', 'sm', 'mean_e', 'wfm'],
        'checkpoint_selection': 'same best-MAE checkpoint for all metrics; no post-hoc per-metric epochs',
        'quality_input': 'detached or shared pooled RGB features, global RGB context, mask area; NO weak roles or GT',
        'quality_target': 'post-augmentation train-GT foreground purity, no relationship labels',
        'quality_feedback': 'detached score, zero-init signed direct/recipient/edge adapters trained by COD',
        'optimizers': 'main AdamW excludes quality head; head AdamW separately clipped and stepped',
        'control_requirement': 'observer must reproduce control main training trajectory and predictions exactly',
        'selection': 'fixed four conditions, two seeds; no weight or adapter-strength search',
        'test_evaluated': False, 'qwen_or_sam_regenerated': False,
        'claim_limit': 'development mechanism test only; not stable generalization or semantic relation proof'})
    print('QUALITY_DECOUPLING_PLAN_WRITTEN', flush=True)


if __name__ == '__main__':
    main()
