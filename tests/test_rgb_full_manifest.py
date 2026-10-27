"""RGB must not require a single prior; region conditions still must."""
from pathlib import Path
import sys
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.semantic_prior.train_pilot import manifest_paths_for, require_prior_files
from tools.official_sinetv2.metrics import prediction_to_png


def main():
    root = Path('manifests')
    assert [p.name for p in manifest_paths_for({}, root)] == ['pilot_train.csv', 'pilot_val.csv']
    assert [p.name for p in manifest_paths_for({'manifest_files': {'train':'train.csv', 'val':'val.csv'}}, root)] == ['train.csv', 'val.csv']
    rows = [[{'image_id':'synthetic/no_prior'}]]
    require_prior_files(rows, ['rgb'])
    for arm in ['direct', 'relation']:
        try:
            require_prior_files(rows, [arm])
        except ValueError:
            pass
        else:
            raise AssertionError('A missing region prior was accepted')
    with tempfile.TemporaryDirectory() as temporary:
        package = Path(temporary)/'present.npz'
        np.savez(package, valid=np.zeros(0))
        require_prior_files([[{'image_id':'synthetic/present', 'region_package':str(package)}]], ['rgb','relation'])
    rng = np.random.default_rng(41)
    for p in [rng.random((22, 25), dtype=np.float32), np.ones((8, 8), np.float32)*.4]:
        old = (p-p.min())/(p.max()-p.min()+1e-8)
        span = old.max()-old.min()
        expected = (((old-old.min())*(255/(span if span else 1))).clip(0,255)+.5).astype(np.uint8)
        assert np.array_equal(prediction_to_png(p), expected)
    print('RGB_FULL_MANIFEST_TESTS_PASSED', flush=True)


if __name__ == '__main__':
    main()
