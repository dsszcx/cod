"""tools/e0e4/infer_e0.py - E0: frozen 3-channel PVTv2-B2 baseline -> M0 soft/binary.

Why not reuse `MyTest.py`
-------------------------
`MyTest.py` is CAMO-only, needs a GT folder to size its output, and its metrics
folder convention is "one folder per checkpoint".  E0 needs the opposite:

* **no GT anywhere** -- the output size comes from the RGB image itself;
* ``M0_soft.npy`` kept at full resolution (float32, sigmoid, no stretch) because
  ROI extraction and E4 both read it;
* ``M0_binary.png`` at threshold 0.5 for the ROI stage;
* ``predictions/E0/<dataset>/<stem>.png`` for `eval_strict.py`.

Protocol (matches ``eval_metrics.py`` "raw sigmoid")
---------------------------------------------------
``logits = model(x)[3]``  (4th return value, S_3_pred)
``soft   = sigmoid(bilinear(logits -> (H, W)))``   -- interpolate BEFORE sigmoid
``binary = soft >= 0.5``
No per-image min-max: the historical stretch made the 3ch row and the SAM3 row
incomparable (see `eval_metrics.py` docstring).

CAVEAT (pre-registered in the config)
-------------------------------------
``model_pth/OUR/log.log`` prints ``MAE=46.076...`` and ``Best Epoch: 0``: the
grader kept GT in 0..255 against a 0..1 prediction, so the ``best`` checkpoint it
names cannot be trusted.  ``Net_epoch_40.pth`` (training-complete) is used
instead and its sha256 is recorded next to every prediction.

Usage::

    python tools/e0e4/infer_e0.py --config configs/e0e4_v1.yaml \
        --ids outputs/E0_E4_ZEROTRAIN_V1/manifest_p0.txt --gpu_id 0
"""
import argparse
import os
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from tools.e0e4 import common as C  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(testsize):
    """Exactly the pipeline of ``dataloader.test_dataset`` (PIL resize -> tensor)."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((int(testsize), int(testsize))),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_image_tensor(image_path, testsize):
    """RGB image -> (1,3,S,S) tensor, plus the ORIGINAL (H, W)."""
    from PIL import Image

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        width, height = im.size
        tensor = build_transform(testsize)(im).unsqueeze(0)
    return tensor, (height, width)


def soft_from_logits(logits, orig_hw):
    """sigmoid(bilinear(logits -> original size)) as float32 numpy (H, W)."""
    import torch
    import torch.nn.functional as F

    logits = logits.float()
    if logits.dim() == 4:
        logits = logits[0]
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    if logits.dim() == 3 and logits.shape[0] != 1:
        logits = logits[:1]
    up = F.interpolate(logits.unsqueeze(0), size=(int(orig_hw[0]), int(orig_hw[1])),
                       mode="bilinear", align_corners=False)
    return torch.sigmoid(up).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def binarize(soft, threshold=0.5):
    return (np.asarray(soft, dtype=np.float32) >= float(threshold))


def load_model(ckpt_path, device, use_ca=False):
    """Strict load of the 3ch baseline.

    ``use_ca=False``: every 740-key checkpoint in model_pth/ predates the
    ChannelAttention inside GRA (only ``RS*.*_gra.ca.fc.*`` are missing), so the
    network is rebuilt without ``ca`` and strict-loads 740/740.  With
    ``use_ca=True`` the load would have to skip 18 keys and those layers would be
    random -- that is a different model, not the frozen baseline.
    """
    from check_ckpt_channels import assert_ckpt_channels, load_with_report
    from Network_PVTv2 import Network

    if not os.path.isfile(ckpt_path):
        raise C.StageError("E0 checkpoint not found: %s" % ckpt_path)
    assert_ckpt_channels(ckpt_path, expected=3, require_fusion=False)
    model = Network(channel=64, imagenet_pretrained=False,
                    use_ca=bool(use_ca)).to(device)
    load_with_report(model, ckpt_path, expected=3, require_fusion=False, strict=True)
    model.eval()
    return model


def run_stage(cfg, out_dir, ids, limit=None, overwrite=False, gpu_id="0"):
    """E0 over ``ids`` -> e0/<dataset>/<stem>/{M0_soft.npy,M0_binary.png,...}."""
    import torch
    from utils import save_map

    C.guard_config(cfg)
    if C.cfg_get(cfg, "e0.minmax", False):
        raise C.StageError("e0.minmax=true is not allowed: use the raw sigmoid protocol")

    ckpt = C.resolve_path(C.cfg_get(cfg, "paths.e0_ckpt"))
    testsize = int(C.cfg_get(cfg, "e0.input_size", 352))
    threshold = float(C.cfg_get(cfg, "e0.threshold", 0.5))
    use_ca = bool(C.cfg_get(cfg, "e0.use_ca", False))

    data_root = C.resolve_path(C.cfg_get(cfg, "paths.data_root"))
    datasets = C.cfg_get(cfg, "paths.datasets", ["CAMO", "COD10K"])
    rows, problems = C.manifest_rows(data_root, datasets)
    if problems:
        raise C.StageError("dataset problems: %s" % "; ".join(problems))
    selected = C.rows_for_ids(rows, ids)
    if limit:
        selected = selected[:int(limit)]

    root = os.path.abspath(out_dir)
    e0_root = os.path.join(root, "e0")
    pred_root = os.path.join(root, "predictions", "E0")
    C.guard_stage_dir(e0_root, overwrite=overwrite)
    C.guard_stage_dir(pred_root, overwrite=overwrite)
    C.ensure_dir(e0_root)
    C.ensure_dir(pred_root)

    device = "cuda:%s" % gpu_id if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(">>> [E0] WARNING: CUDA unavailable, running on CPU")
    model = load_model(ckpt, device, use_ca=use_ca)
    ckpt_sha = C.sha256_file(ckpt)
    print(">>> [E0] ckpt=%s sha256=%s size=%d threshold=%.2f use_ca=%s device=%s"
          % (ckpt, ckpt_sha, testsize, threshold, use_ca, device))

    index, t_start = [], time.time()
    with torch.inference_mode():
        for idx, row in enumerate(selected, 1):
            dataset, stem = row["dataset"], row["stem"]
            image_path, _gt = C.rel_id_to_paths(row, cfg)
            t0 = time.time()
            tensor, orig_hw = load_image_tensor(image_path, testsize)
            outputs = model(tensor.to(device))
            soft = soft_from_logits(outputs[3], orig_hw)
            binary = binarize(soft, threshold)

            img_dir = os.path.join(e0_root, dataset, stem)
            C.ensure_dir(img_dir)
            np.save(os.path.join(img_dir, "M0_soft.npy"), soft)
            save_map(os.path.join(img_dir, "M0_binary.png"), binary.astype(np.float32))
            save_map(os.path.join(img_dir, "M0_soft.png"), soft)
            pred_path = os.path.join(pred_root, dataset, stem + ".png")
            C.ensure_dir(os.path.dirname(pred_path))
            save_map(pred_path, binary.astype(np.float32))

            meta = {"image_id": row["image_id"], "group": row["group"],
                    "orig_hw": list(orig_hw), "input_size": testsize,
                    "threshold": threshold, "minmax": False, "use_ca": use_ca,
                    "foreground_fraction": round(float(binary.mean()), 6),
                    "soft_mean": round(float(soft.mean()), 6),
                    "soft_max": round(float(soft.max()), 6),
                    "ckpt": os.path.relpath(ckpt, C.ROOT).replace(os.sep, "/"),
                    "ckpt_sha256": ckpt_sha,
                    "config_version": cfg.get("_version"),
                    "elapsed_ms": int(round((time.time() - t0) * 1000))}
            C.write_json(os.path.join(img_dir, "meta.json"), meta)
            index.append({"image_id": row["image_id"], "group": row["group"],
                          "H": orig_hw[0], "W": orig_hw[1],
                          "fg_fraction": meta["foreground_fraction"]})
            C.progress(idx, len(selected), "%s fg=%.4f (%.0f ms)"
                       % (row["image_id"], binary.mean(), meta["elapsed_ms"]))

    C.write_csv(os.path.join(root, "E0_INDEX.csv"), index)
    C.write_json(os.path.join(root, "E0_RUN.json"), {
        "stage": "E0", "utc": C.utc_now(), "n_images": len(index),
        "ckpt": ckpt, "ckpt_sha256": ckpt_sha, "input_size": testsize,
        "threshold": threshold, "minmax": False, "use_ca": use_ca,
        "device": device, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "elapsed_s": round(time.time() - t_start, 1),
        "config": {"version": cfg.get("_version"), "sha256": cfg.get("_sha256")},
    })
    print(">>> E0 done: %d image(s) -> %s (%.1f s)"
          % (len(index), e0_root, time.time() - t_start))
    return index


def main(argv=None):
    ap = argparse.ArgumentParser(description="E0: 3ch baseline -> M0 soft/binary")
    ap.add_argument("--config", default=os.path.join(C.ROOT, "configs", "e0e4_v1.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu_id", default="0")
    ap.add_argument("--overwrite", action="store_true")
    opt = ap.parse_args(argv)
    try:
        cfg = C.load_config(opt.config)
        out_dir = C.resolve_path(opt.out or C.cfg_get(cfg, "paths.out_dir"))
        run_stage(cfg, out_dir, C.read_id_file(opt.ids), limit=opt.limit,
                  overwrite=opt.overwrite, gpu_id=opt.gpu_id)
    except C.StageError as exc:
        print(">>> BLOCKED: %s" % exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
