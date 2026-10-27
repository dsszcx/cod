"""MyTest.py - inference for the 3-channel PVTv2-B2 baseline.

Usage:
  python MyTest.py --pth_path model_pth/PVT/KKK_best_41.pth --datasets CAMO

Layer-1 fixes (2026-10).  Every one of them used to make a *number* wrong
without printing a warning, which is why the 3ch baseline could not be put in
the same table as the SAM3 rows:

  * ``--datasets``   the dataset list was hard-coded to ``['CAMO']`` even though
    ``--data_root`` advertised multi-dataset support.
  * ``--minmax``     the per-image contrast stretch was unconditional, while
    ``MyTest_SAM3.py`` has always had it OFF by default: the baseline row was
    measured on stretched maps, the SAM3 row on raw sigmoid.  Both scripts now
    default to raw and the protocol is printed + recorded in ``run.json``.
  * ``--save_name``  the output folder was derived from the checkpoint's parent
    directory, so KKK_best_10.pth and KKK_best_50.pth silently overwrote each
    other's maps.
  * the checkpoint is channel-checked (``assert_ckpt_channels(expected=3)``) and
    loaded with a key report; a 7/8ch file used to load with ``strict=False``
    and leave the backbone randomly initialised.
  * ``cv2.imwrite``'s return value is checked: a failed write used to print
    "saving prediction" anyway and surface later as a short prediction folder.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from Network_PVTv2 import Network
from check_ckpt_channels import assert_ckpt_channels, load_with_report
from dataloader import test_dataset
from utils import save_map, sha256_file, write_run_manifest


def build_parser():
    parser = argparse.ArgumentParser(
        description="Inference for the 3-channel PVTv2-B2 COD baseline")
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, required=True,
                        help='path to trained model checkpoint (.pth)')
    parser.add_argument('--data_root', type=str, default='./Dataset/TestDataset/',
                        help='root path to test datasets')
    parser.add_argument('--datasets', type=str, default='CAMO',
                        help="comma-separated test dataset names, e.g. CAMO,COD10K")
    parser.add_argument('--save_root', type=str, default='./results/',
                        help='path to save prediction maps')
    parser.add_argument('--save_name', type=str, default=None,
                        help='subfolder under --save_root (default: checkpoint folder name)')
    parser.add_argument('--model', type=str, default='PVTv2-B2',
                        choices=['PVTv2-B1', 'PVTv2-B2', 'PVTv2-B3', 'PVTv2-B4', 'PVTv2-B5'],
                        help='kept for CLI compatibility; the code path is PVTv2-B2')
    parser.add_argument('--gpu_id', type=str, default='0', help='visible GPU id')
    parser.add_argument('--minmax', action='store_true',
                        help='apply the historical per-image min-max stretch '
                             '(OFF by default = standard raw-sigmoid protocol)')
    parser.add_argument('--deterministic', action='store_true',
                        help='disable cudnn autotuning / use deterministic kernels')
    return parser


def main(argv=None):
    opt = build_parser().parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    cudnn.benchmark = not opt.deterministic
    if opt.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    # ---- Layer-0 guard: the 3ch network must not accept a 7/8ch file ----
    # imagenet_pretrained=False: the checkpoint below replaces every weight, so
    # downloading/loading pvt_v2_b2.pth first is pure waste (and fails offline).
    assert_ckpt_channels(opt.pth_path, expected=3, require_fusion=False)
    model = Network(channel=64, imagenet_pretrained=False).cuda()
    load_with_report(model, opt.pth_path, expected=3, require_fusion=False,
                     strict=True)
    model.eval()

    _run_name = opt.save_name or os.path.basename(os.path.dirname(opt.pth_path))
    txt_save_path = os.path.join(opt.save_root, _run_name)
    os.makedirs(txt_save_path, exist_ok=True)

    print('>>> config:', opt)
    print(">>> protocol: %s"
          % ('minmax (historical)' if opt.minmax else 'raw sigmoid'))
    print(">>> saving predictions to: %s" % os.path.abspath(txt_save_path))

    n_fail = 0
    per_dataset = {}
    for _data_name in [s.strip() for s in opt.datasets.split(',') if s.strip()]:
        data_path = os.path.join(opt.data_root, _data_name)
        if not os.path.isdir(os.path.join(data_path, 'Imgs')):
            print(">>> WARNING: test dataset not found, skip: %s" % data_path)
            continue
        map_save_path = os.path.join(txt_save_path, _data_name)
        os.makedirs(map_save_path, exist_ok=True)

        image_root = os.path.join(data_path, 'Imgs').replace(os.sep, '/') + '/'
        gt_root = os.path.join(data_path, 'GT').replace(os.sep, '/') + '/'
        test_loader = test_dataset(image_root, gt_root, opt.testsize)
        model.eval()

        for i in range(test_loader.size):
            _, image, gt, name, _ = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            image = image.cuda()

            glo, cam_out_2, cam_out_3, cam_out_4 = model(image)

            res = F.interpolate(cam_out_4, size=gt.shape, mode='bilinear',
                                align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            if opt.minmax:      # historical non-standard contrast stretch
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            _out = os.path.join(map_save_path, name)
            try:
                save_map(_out, res)
            except IOError as exc:
                n_fail += 1
                print(">>> [ERROR] %s" % exc)
                continue
            print('>>> saving prediction at: {}'.format(_out))

        per_dataset[_data_name] = test_loader.size
        print(">>> [%s] wrote %d map(s) -> %s"
              % (_data_name, test_loader.size, os.path.abspath(map_save_path)))

    if n_fail:
        raise SystemExit("ERROR: %d prediction map(s) could not be written"
                         % n_fail)

    write_run_manifest(os.path.join(txt_save_path, "run.json"), opt, extra={
        "script": "MyTest.py",
        "pth_sha256": sha256_file(opt.pth_path),
        "pred_protocol": "minmax_baked" if opt.minmax else "raw",
        "n_images": per_dataset,
        "testsize": opt.testsize,
    })
    return 0


if __name__ == '__main__':
    sys.exit(main())
