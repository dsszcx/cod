"""Formula/edge checks, NOT a claim of official-weight/MATLAB reproduction."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

path = Path(__file__).resolve().parents[1]/'tools/official_sinetv2/metrics.py'
spec = importlib.util.spec_from_file_location('official_metrics', path)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class TestOfficialProtocol(unittest.TestCase):
    def test_constant_normalization_unchanged(self):
        for value in [0, 127, 255]:
            p = np.full((8, 9), value, np.uint8)
            np.testing.assert_allclose(M.mapminmax(p), value/255)

    def test_e_strict_threshold_matches_literal_matlab_formula(self):
        rng = np.random.default_rng(27)
        for gt in [rng.random((11, 12)) > .7, np.zeros((11, 12), bool), np.ones((11, 12), bool)]:
            pred = M.mapminmax(rng.integers(0, 256, gt.shape, dtype=np.uint8))
            literal = []
            for threshold in M.THRESHOLDS:
                b = (pred > threshold).astype(float)
                if not gt.any():
                    enhanced = 1-b
                elif gt.all():
                    enhanced = b
                else:
                    bp, bg = b-b.mean(), gt-gt.mean()
                    align = 2*bp*bg/(bp*bp+bg*bg+M.EPS)
                    enhanced = (align+1)**2/4
                literal.append(enhanced.sum()/(b.size-1+M.EPS))
            np.testing.assert_allclose(M.e_curve(pred, gt), literal, atol=1e-14)

    def test_perfect_map_and_four_metric_contract(self):
        gt = np.zeros((20, 22), np.uint8)
        gt[5:15, 6:16] = 255
        result = M.evaluate(gt, gt)
        self.assertEqual(set(result), {'mae', 'sm', 'mean_e', 'wfm'})
        self.assertEqual(result['mae'], 0)
        self.assertAlmostEqual(result['sm'], 1)
        self.assertAlmostEqual(result['wfm'], 1)
        self.assertGreater(result['mean_e'], .99)

    def test_weighted_f_matches_independent_pysod_backend(self):
        from py_sod_metrics import WeightedFmeasure
        rng = np.random.default_rng(77)
        gt = rng.random((21, 23)) > .65
        p = rng.random(gt.shape)
        self.assertAlmostEqual(M.weighted_f(p, gt), WeightedFmeasure().cal_wfm(p, gt), places=13)

    def test_structure_matlab_one_based_half_rounding(self):
        # 0-based centroid 4.5: numpy round(4.5)+1=5; MATLAB round(5.5)=6.
        from py_sod_metrics import Smeasure
        rng = np.random.default_rng(8)
        gt = np.zeros((12, 14), bool)
        gt[3:7, 4:8] = True
        p = rng.random(gt.shape)
        # Correct one-based centroid is (6.5, 5.5), giving (7, 6).
        cx, cy = 7, 6
        backend = Smeasure()
        region = sum(weight*backend.ssim(p[r], gt[r]) for weight, r in [
            (cx*cy/p.size, (slice(0, cy), slice(0, cx))),
            ((14-cx)*cy/p.size, (slice(0, cy), slice(cx, 14))),
            (cx*(12-cy)/p.size, (slice(cy, 12), slice(0, cx))),
            ((14-cx)*(12-cy)/p.size, (slice(cy, 12), slice(cx, 14)))])
        expected = max(0, .5*backend.object(p, gt)+.5*region)
        self.assertAlmostEqual(M.structure(p, gt), expected, places=14)

    def test_empty_gt_and_size_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            M.evaluate(np.zeros((4, 5), np.uint8), np.zeros((4, 5), np.uint8))
        with self.assertRaises(ValueError):
            M.evaluate(np.zeros((4, 5), np.uint8), np.ones((5, 5), np.uint8)*255)


if __name__ == '__main__':
    unittest.main()
