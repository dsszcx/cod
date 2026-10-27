"""Small synthetic report regression, no real training data or server needed."""
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.e0e4 import common as C


def main():
    parent = ROOT / '.test_tmp'
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='candidate_report_', dir=parent) as directory:
        root = Path(directory)
        inputs = root / 'metric_inputs'
        inputs.mkdir()
        (inputs / 'RGB').mkdir()
        gt = np.zeros((16, 16), np.float32)
        gt[4:12, 4:12] = 1
        targets = {'TrainVal/a': gt, 'TrainVal/b': 1 - gt}
        np.savez_compressed(inputs / 'targets.npz', **targets)
        for key in targets:
            Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(inputs / 'RGB' / (key.replace('/', '__') + '.png'))
        reports = []
        for seed in (20261022, 20261023):
            for condition, arms in (('control', ('rgb', 'direct', 'relation')), ('purity', ('direct', 'relation'))):
                for arm in arms:
                    name = 's%d_%s_%s' % (seed, condition, arm)
                    probs = {k: (v * .7 + .1).astype(np.float32) for k, v in targets.items()}
                    mae = float(np.mean([np.abs(probs[k] - targets[k]).mean() for k in targets]))
                    np.savez_compressed(inputs / (name + '.npz'), **probs)
                    reports.append({'name': name, 'seed': seed, 'condition': condition, 'arm': arm,
                                    'best_epoch': 1, 'last_mae': mae, 'mae': {'full': mae, 'no_priors': mae},
                                    'gate_calibration': {'n_images': 2, 'n_nodes': 4,
                                                         'image_macro_mae': .2, 'image_macro_mse': .05,
                                                         'foreground_gate_mean': .8, 'background_gate_mean': .2}})
        C.write_json(str(root / 'PLAN.json'), {'n_training_runs': 10, 'epochs': 10})
        C.write_json(str(root / 'EVALUATION.json'), {'status': 'DONE', 'reports': reports})
        C.write_json(str(root / 'RELATION_EDGE_UTILITY.json'), {'status': 'DONE', 'reports': [
            {'name': 'synthetic_relation', 'n_edges': 3, 'helpful_edges_gt_1e7': 1,
             'harmful_edges_lt_minus_1e7': 1, 'near_zero_edges': 1,
             'images': [{'edges': [{'mae_removed_minus_full': 2e-7},
                                   {'mae_removed_minus_full': -3e-7},
                                   {'mae_removed_minus_full': 0.}]}, {'edges': []}]}]})
        subprocess.run([sys.executable, str(ROOT / 'tools/semantic_prior/report_candidate_experiment.py'),
                        '--root', str(root), '--rgb-root', str(inputs / 'RGB')], check=True, stdout=subprocess.DEVNULL)
        result = C.read_json(str(root / 'RAW_METRICS_SUMMARY.json'))
        assert len(result['per_run']) == 10 and len(result['mean_per_condition']) == 5
        assert result['n_unique_validation_images'] == 2 and result['normalize'] is False
        assert (root / 'RESULTS.md').is_file()
        assert 'synthetic_relation' in (root / 'RESULTS.md').read_text(encoding='utf-8')
        assert (root / 'CASE_REVIEW.png').is_file() and (root / 'CASE_SELECTION.json').is_file()
    print('CANDIDATE_REPORT_TESTS_PASSED')


if __name__ == '__main__':
    main()
