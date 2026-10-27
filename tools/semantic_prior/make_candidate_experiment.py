"""Fixed 2-seed, 5-arm test of train-only candidate purity supervision."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import yaml

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--manifests', required=True)
    ap.add_argument('--imagenet', required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--minutes', type=int, default=50)
    args = ap.parse_args()
    if args.epochs < 1 or args.minutes < 1:
        raise ValueError('Positive epochs and deadline duration required')
    root = resolve(args.out)
    root.mkdir(exist_ok=False, parents=True)
    base = {'version': 'candidate_purity_v1', 'data_scope': 'cam_only',
            'expected_train_images': 128, 'expected_val_images': 32,
            'epochs': args.epochs, 'input_size': 352, 'batch_size': 2,
            'learning_rate': 1e-4, 'weight_decay': 1e-4, 'workers': 0,
            'amp': False, 'node_geometry': False, 'deterministic_algorithms': True,
            'collect_fusion_stats': True,
            'deadline_utc': (datetime.now(timezone.utc) + timedelta(minutes=args.minutes)).isoformat(),
            'fusion_options': {'alpha_init': .01, 'projection_normalization': 'feature_rms'}}
    jobs = []
    for seed in (20261022, 20261023):
        for condition, weight, arms in (('control', 0., ['rgb', 'direct', 'relation']),
                                        ('purity', .1, ['direct', 'relation'])):
            out = root / ('s%d_%s' % (seed, condition))
            cfg = {**base, 'seed': seed, 'arms': arms,
                   'candidate_reliability_weight': weight,
                   'paths': {'manifests': str(resolve(args.manifests)),
                             'imagenet': str(resolve(args.imagenet)), 'out_dir': str(out)}}
            path = root / ('s%d_%s.yaml' % (seed, condition))
            path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
            jobs.append({'seed': seed, 'condition': condition, 'config': str(path),
                         'out_dir': str(out), 'arms': arms, 'auxiliary_weight': weight})
    C.write_json(str(root / 'PLAN.json'), {
        'created_utc': C.utc_now(), 'jobs': jobs, 'n_training_runs': 10,
        'epochs': args.epochs, 'n_train': 128, 'n_val': 32,
        'test_evaluated': False, 'qwen_or_sam_regenerated': False,
        'hypothesis': 'Purity supervision improves direct reliability without removing context relation paths',
        'loss': 'four-output structure loss + 0.1 * image-balanced node soft BCE',
        'node_target': 'foreground pixels in candidate / candidate pixels, after same spatial augmentation',
        'relation_supervision': 'none; binary COD GT does not label context utility or same-instance relations',
        'selection': 'all conditions fixed before this experiment; no loss-weight search',
        'evaluation': 'same 32 development images: best and last MAE, raw probability metrics and gate calibration',
        'claim_limit': 'short controlled development experiment, not official test or stable generalization proof'})
    print('CANDIDATE_EXPERIMENT_PLAN_WRITTEN', flush=True)


if __name__ == '__main__':
    main()
