"""Independent benchmark RGB input must exactly match training validation."""
from pathlib import Path
import sys
import tempfile
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dataloader_relation import RelationDataset
from tools.e0e4 import common as C
from tools.semantic_prior.audit_rgb_baseline import rgb_input,replay_differences


def main():
    (ROOT/'.test_tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'.test_tmp') as temp:
        root = Path(temp)
        rng = np.random.default_rng(21)
        p,g = root/'image.png',root/'mask.png'
        Image.fromarray(rng.integers(0,256,(47,63,3),dtype=np.uint8)).save(p)
        Image.fromarray(np.ones((47,63),np.uint8)*255).save(g)
        row = dict(image_id='TrainVal/camourflage_fixture',split='val',image_path=str(p),gt_path=str(g),image_sha256=C.sha256_file(str(p)))
        C.write_csv(str(root/'val.csv'),[row])
        old = RelationDataset(root/'val.csv','val','rgb',size=32)[0]['inputs']['rgb']
        new,shape = rgb_input(p,32)
        assert shape == (47,63) and np.array_equal(old,new[0].numpy())
    metrics = dict(mae=.1,sm=.8,mean_e=.9,wfm=.7)
    a = [dict(image_id='x',raw=metrics.copy(),png_standard=metrics.copy())]
    b = [dict(image_id='x',raw=metrics.copy(),png_standard=metrics.copy())]
    assert all(v==0 for v in replay_differences(a,b).values())
    b[0]['raw']['mae'] += 1e-8
    try:
        replay_differences(a,b)
    except ValueError:
        pass
    else:
        raise AssertionError('Replay accepted changed validation metrics')
    print('RGB_BASELINE_AUDIT_TESTS_PASSED',flush=True)


if __name__ == '__main__':
    main()
