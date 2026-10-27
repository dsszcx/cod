"""MyTest_SAM3.py - Inference with SAM3 region-prior network.

Usage:
  python MyTest_SAM3.py --pth_path ./model_pth/SAM3/KKK_best_2.pth --prior_root ./prior_sam3/

Protocol note (Layer 0 fix): the prediction map is saved as the raw sigmoid x 255
(standard protocol).  Pass ``--minmax`` to re-apply the historical per-image
min-max stretch; prediction folders written before this switch have the stretch
baked in (use ``eval_metrics.py --minmax`` on those to reproduce old numbers).

Ablation switches (default = unchanged behaviour):
  --prior_mode zero|oracle|shuffle   ablation of the SAM3 prior branch
  --neg_flag zero                    feed an all-zero 8th channel instead of the
                                     COD10K "NonCAM" file-name label (mandatory
                                     for the 2276-image CAMO+COD10K-CAM table)
  --save_name NAME                   write to results_sam3/NAME/ instead of
                                     results_sam3/<ckpt folder>/ (never overwrite
                                     the baseline results with an ablation run)
  --strict_prior                     abort if any test image has no real prior
                                     (default: only report the coverage)

Layer-0 fixes (2026-10):
  * priors are resized AND binarized at 0.5 through the shared
    dataloader_sam3.prepare_prior_at, so the test-time prior distribution is
    identical to training (this file used to interpolate without binarizing).
  * neg_flag follows the COD10K "NonCAM" file-name rule exactly like the
    training loader (this file used to hard-code 0 -> 1974/4000 COD10K-test
    images were fed the wrong 8th channel).
  * prior coverage and the checkpoint channel count are always printed: a 7ch
    checkpoint or an all-zero prior can no longer pass silently.
  * ``--prior_mode oracle`` resized the GT to the *image* size while the
    network input was ``testsize`` -> a shape mismatch for every image whose
    native size differs.  The GT is now NEAREST-resampled to the network input.

NOTE: predictions produced after these fixes are NOT directly comparable with
      older runs (both the prior input and the neg_flag input changed).
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import cv2
from dataloader import test_dataset
from dataloader_sam3 import (check_prior_coverage, load_test_prior,
                             neg_flag_value)
from check_ckpt_channels import assert_ckpt_channels, load_with_report
from Network_SAM3 import Network


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testsize", type=int, default=352)
    parser.add_argument("--pth_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./Dataset/TestDataset/")
    parser.add_argument("--datasets", type=str, default="CAMO,COD10K",
                        help="comma-separated test dataset names")
    parser.add_argument("--save_root", type=str, default="./results_sam3/")
    parser.add_argument("--prior_root", type=str, default="./prior_sam3/")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--prior_mode", type=str, default="real",
                        choices=["real", "zero", "oracle", "shuffle"],
                        help="ablation: real=SAM3 priors (default), zero=no prior, "
                             "oracle=GT as prior, shuffle=previous image's prior")
    parser.add_argument("--neg_flag", type=str, default="label", choices=["label", "zero"],
                        help="8th input channel: 'label' (default) = 1 for COD10K "
                             "'NonCAM' images (the answer is read from the file "
                             "name); 'zero' = always 0, required for an honest "
                             "CAMO+COD10K-CAM table")
    parser.add_argument("--save_name", type=str, default=None,
                        help="subfolder under save_root (default: ckpt folder name)")
    parser.add_argument("--minmax", action="store_true",
                        help="apply the historical per-image min-max stretch (off by default)")
    parser.add_argument("--strict_prior", action="store_true",
                        help="abort if any image has no real prior npz (default: report only)")
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    cudnn.benchmark = True

    # ---- Layer-0 guard: a 7ch / fusion-less checkpoint must not load silently ----
    assert_ckpt_channels(opt.pth_path, expected=8)   # fails BEFORE the GPU is used
    model = Network(channel=64, imagenet_pretrained=False).cuda()
    load_with_report(model, opt.pth_path, expected=8)
    model.eval()

    _run_name = opt.save_name or os.path.basename(os.path.dirname(opt.pth_path))
    txt_save_path = os.path.join(opt.save_root, _run_name)
    os.makedirs(txt_save_path, exist_ok=True)
    print(">>> config:", opt)

    _no_prior_ds = []
    for _data_name in [s.strip() for s in opt.datasets.split(",") if s.strip()]:
        map_save_path = txt_save_path + "/{}/".format(_data_name)
        os.makedirs(map_save_path, exist_ok=True)
        data_path = os.path.join(opt.data_root, _data_name)
        if not os.path.isdir(os.path.join(data_path, "Imgs")):
            print(f">>> WARNING: test dataset not found, skip: {data_path}")
            continue
        # ---- Layer-0 guard: never silently evaluate an all-zero prior ----
        try:
            check_prior_coverage(opt.prior_root, "{}/Imgs/".format(data_path),
                                 split="test", strict=(opt.prior_mode == "real"))
        except FileNotFoundError as e:
            # 0% coverage with --prior_mode real: scoring this dataset would
            # produce a "SAM3 prior" number from all-zero priors.  Skip it loudly
            # and make the process exit non-zero instead of dying with a
            # traceback in the middle of a multi-dataset run.
            print(">>> [FATAL] " + str(e))
            print(">>> [FATAL] dataset '{}' SKIPPED: no real prior available. "
                  "Use --prior_mode zero for a no-prior arm, or generate the "
                  "priors first (region_selector.py).".format(_data_name))
            _no_prior_ds.append(_data_name)
            continue
        test_loader = test_dataset("{}/Imgs/".format(data_path), "{}/GT/".format(data_path), opt.testsize)

        n_prior_used, n_prior_missing, n_noncam_imgs = 0, 0, 0
        n_prior_empty = 0
        prev_name = None
        with torch.no_grad():
            for i in range(test_loader.size):
                _, image, gt, name, _ = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                image = image.cuda()
                hw = image.shape[-2:]
                if opt.prior_mode == "zero":
                    target = torch.zeros(1, *hw)
                    aux = torch.zeros(3, *hw)
                elif opt.prior_mode == "oracle":
                    # GT is at *native* resolution; the network input is `hw`.
                    # The old code fed the native-size mask -> shape mismatch
                    # for every image whose size != testsize.  NEAREST keeps the
                    # prior binary (no 0~1 boundary values).
                    _g = torch.from_numpy((gt > 127.5).astype(np.float32))
                    _g = _g.view(1, 1, *_g.shape)
                    target = F.interpolate(_g, size=tuple(hw), mode="nearest").view(1, *hw)
                    aux = torch.zeros(3, *hw)
                elif opt.prior_mode == "shuffle":
                    # deterministic negative control: previous image's prior
                    target, aux, _found = load_test_prior("test", prev_name or name,
                                                          opt.prior_root, hw)
                    n_prior_used += int(_found)
                    n_prior_missing += int(not _found)
                else:
                    target, aux, _found = load_test_prior("test", name,
                                                          opt.prior_root, hw)
                    n_prior_used += int(_found)
                    n_prior_missing += int(not _found)
                prev_name = name
                # file present but all-zero target mask == dead prior branch for
                # this image (CAMO test: 52/250 in prior_sam3_0814, 79/250 in
                # prior_sam3_sel).  Reported separately so coverage (100%) cannot
                # hide it.
                if (opt.prior_mode in ("real", "shuffle") and _found
                        and float(target.sum()) == 0.0):
                    n_prior_empty += 1
                target = target.unsqueeze(0).cuda()
                aux = aux.unsqueeze(0).cuda()

                # Layer-0: same 8th-channel rule as training (--neg_flag)
                _nv = neg_flag_value(name, opt.neg_flag)
                n_noncam_imgs += int(_nv > 0)
                neg_flag = torch.full((1, 1, image.shape[-2], image.shape[-1]),
                                      _nv, device=image.device)
                _, _, _, c4 = model(image, target, aux, neg_flag)
                res = F.interpolate(c4, size=gt.shape, mode="bilinear", align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                if opt.minmax:      # historical non-standard contrast stretch
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                cv2.imwrite(map_save_path + name, res * 255)
                print(">>> saved:", map_save_path + name)

        _nf_txt = ("neg_flag: all 0 (--neg_flag zero)" if opt.neg_flag == "zero"
                   else "neg_flag=1 on %d/%d images" % (n_noncam_imgs, test_loader.size))
        if opt.prior_mode in ("real", "shuffle"):
            print(">>> [{:s}] prior loaded {}/{} (missing {}, all-zero {}) | {}"
                  .format(_data_name, n_prior_used, test_loader.size,
                          n_prior_missing, n_prior_empty, _nf_txt))
            if n_prior_empty:
                print(">>> [{:s}] WARNING: {}/{} existing priors are ALL-ZERO: the "
                      "prior branch is dead for those images"
                      .format(_data_name, n_prior_empty, n_prior_used))
            if opt.strict_prior and n_prior_missing:
                raise SystemExit("ERROR: {} of {} images have no real prior under {} "
                                 "(--strict_prior)".format(n_prior_missing,
                                                           test_loader.size,
                                                           opt.prior_root))
        else:
            print(">>> [{:s}] prior_mode={} (npz not used) | {}"
                  .format(_data_name, opt.prior_mode, _nf_txt))

    if _no_prior_ds:
        print(">>> [FATAL] no real prior for dataset(s): {} -> exit 2"
              .format(", ".join(_no_prior_ds)))
        raise SystemExit(2)
