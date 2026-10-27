"""A single full-data engineering epoch, NOT a converged benchmark run."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import yaml

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--manifests', required=True, type=Path)
    p.add_argument('--imagenet', required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = dict(version='rgb_full_data_acceptance_v1', seed=20261022,
               data_scope='cam_only', manifest_files={'train': 'train.csv', 'val': 'val.csv'},
               expected_train_images=3652, expected_val_images=383,
               arms=['rgb'], epochs=1, input_size=352, batch_size=8,
               learning_rate=.0001, weight_decay=.0001, workers=0, amp=False,
               node_geometry=False, deterministic_algorithms=True,
               deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
               paths=dict(manifests=str(args.manifests.resolve()),
                          imagenet=str(args.imagenet.resolve()),
                          out_dir=str((args.out/'training').resolve())))
    (args.out/'CONFIG.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    C.write_json(str(args.out/'PLAN.json'), dict(
        purpose='Full RGB data/optimizer/validation acceptance; no efficacy claim',
        n_train=3652, n_val=383, epochs=1, batch_size=8, optimizer_updates=457,
        primary_metrics=['mae', 'sm', 'mean_e', 'wfm'],
        protocols=['raw sigmoid development metrics', 'official-compatible PNG protocol'],
        test_set_used=False, official_sinet_weights_used_for_training=False,
        priors_required=False, qwen_or_sam_regenerated=False,
        checkpoint_selection='fixed single epoch; existing raw val-MAE selector unchanged',
        full_training_complete=False))
    print('RGB_FULL_DATA_ACCEPTANCE_PLAN_WRITTEN', flush=True)


if __name__ == '__main__':
    main()
