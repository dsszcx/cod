"""CPU fixture test of the exact parallel validation path used by training."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.e0e4 import common as C
from dataloader_relation import RelationDataset
from tools.semantic_prior.train_rgb_full import validate


class TinyRGB(torch.nn.Module):
    def forward(self, rgb):
        return [rgb[:, :1]]*4


def main():
    (ROOT/'.test_tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'.test_tmp') as folder:
        root = Path(folder)
        rgb = np.tile(np.arange(32, dtype=np.uint8)*8, (32, 1))
        image = root/'rgb.png'
        Image.fromarray(np.stack([rgb]*3, axis=-1)).save(image)
        gt = np.zeros((32, 32), np.uint8)
        gt[8:24, 8:24] = 255
        target = root/'gt.png'
        Image.fromarray(gt).save(target)
        rows = [dict(image_id='TrainVal/'+name, image_path=str(image),
                     image_sha256=C.sha256_file(str(image)), gt_path=str(target), split='val')
                for name in ('camourflage_fixture', 'COD10K-CAM-1-fixture')]
        manifest = root/'val.csv'
        C.write_csv(str(manifest), rows)
        dataset = RelationDataset(manifest, 'val', 'rgb', size=32)
        with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn')) as pool:
            metrics, per_image = validate(TinyRGB(), dataset, pool, torch.device('cpu'))
        assert len(per_image) == 2
        for protocol in ('raw', 'png_standard'):
            assert metrics[protocol]['all']['n_images'] == 2
            for group in ('all', 'CAMO', 'COD10K-CAM'):
                assert all(np.isfinite(metrics[protocol][group][k]) for k in ('mae','sm','mean_e','wfm'))
    print('FULL_RGB_PARALLEL_FOUR_METRICS_PASSED', flush=True)


if __name__ == '__main__':
    main()
