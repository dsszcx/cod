"""tools/compact_priors.py - shrink prior_sam3/*.npz to a training-friendly size.

WHY
---
`region_selector.py` used to write ``target_mask`` (1,H,W) + ``aux_masks``
(3,H,W) as **float32 at the original image resolution** with ``np.savez``
(uncompressed).  Measured on the COD10K+CAMO merge (2026-10-17):

    8275 files = 78 GB, largest single file 578 MB (images up to 3592x2386)

`dataloader_sam3.CamObjDatasetSAM3` then read that whole file for *every*
sample and resized it back down to ``trainsize``: measured 85 ms/sample median
and 1708 ms max, i.e. 78 GB of disk traffic per epoch (6.2 TB over an 80-epoch
run) competing with every other job on the box.

WHAT
----
Rewrite every npz at a fixed working resolution (default 448 = 1.27 x the 352
training crop) as **uint8 0/1** with ``savez_compressed``:

    target_mask (1,448,448) uint8     aux_masks (3,448,448) uint8
    target_tag, bbox (unchanged, no prior_size rewrite)

Down-sampling uses PIL ``BOX`` (area average) followed by "any coverage" (> 0),
which for a binary mask is exactly max-pooling: thin / low-recall target
structures are never dropped -- the conservative choice for a prior.

WHY NO DOWNSTREAM CHANGE IS NEEDED
----------------------------------
Every consumer already accepts uint8 0/1:

    dataloader_sam3._load_prior      -> .astype(np.float32) -> clip*255 -> PIL
    dataloader_sam3.load_test_prior  -> .float() -> prepare_prior_at()

so switching ``--prior_root prior_sam3`` -> ``--prior_root prior_sam3_c448`` is
the only change a training run needs.

USAGE
-----
    python tools/compact_priors.py --src prior_sam3 --dst prior_sam3_c448
    python tools/compact_priors.py --src prior_sam3 --dst prior_sam3_c448 \
        --verify 300

The source directory is never modified.  The tool is resume-safe (existing
output files are skipped unless ``--overwrite``).
"""
import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

MASK_KEYS = ("target_mask", "aux_masks")


def _to_chw(m):
    """Normalise a mask array to (C,H,W)."""
    m = np.asarray(m)
    if m.ndim == 2:
        m = m[None]
    elif m.ndim == 3 and m.shape[0] > 8:          # (H,W,C) -> (C,H,W)
        m = m.transpose(2, 0, 1)
    return m


def resize_binary_mask(m, size):
    """(C,H,W) binary-ish float -> (C,size,size) uint8 0/1.

    BOX (area) + "any coverage" keeps every pixel that had any coverage in the
    source region, i.e. max-pooling semantics for binary masks.
    """
    m = _to_chw(m)
    out = np.empty((m.shape[0], size, size), np.uint8)
    for c in range(m.shape[0]):
        img = Image.fromarray((m[c] > 0.5).astype(np.uint8) * 255)
        if img.size != (size, size):
            img = img.resize((size, size), Image.BOX)
        out[c] = (np.asarray(img) > 0).astype(np.uint8)
    return out


def compact_one(src_path, dst_path, size, overwrite=False):
    """Convert one npz.  Returns (state, bytes_in, bytes_out)."""
    if os.path.exists(dst_path) and not overwrite:
        return "skip", 0, 0
    with np.load(src_path, allow_pickle=True) as d:
        payload = {k: resize_binary_mask(d[k], size) for k in MASK_KEYS}
        for k in d.files:
            if k not in MASK_KEYS:
                payload[k] = d[k]
    tmp = dst_path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, dst_path)
    return "ok", os.path.getsize(src_path), os.path.getsize(dst_path)


def mask_at(m, size):
    """(C,H,W) -> (size,size) bool for the IoU comparison."""
    return resize_binary_mask(m, size)[0] > 0


def verify(src, dst, size, n, iou_size=352, seed=0):
    """Random-sample IoU between the original and the compacted prior.

    Both sides are resized to `iou_size` the way the loader sees them
    (`prepare_prior_at`: bilinear + binarise at 0.5); here BOX + >0.5.
    """
    files = sorted(f for f in os.listdir(src) if f.endswith(".npz"))
    random.Random(seed).shuffle(files)
    files = files[:n]
    ious, checked = [], 0
    for name in files:
        d_path = os.path.join(dst, name)
        if not os.path.exists(d_path):
            continue
        with np.load(os.path.join(src, name), allow_pickle=True) as ds:
            a = mask_at(ds["target_mask"], iou_size)
            aa = mask_at(ds["aux_masks"], iou_size)
        with np.load(d_path, allow_pickle=True) as dd:
            b = mask_at(dd["target_mask"], iou_size)
            bb = mask_at(dd["aux_masks"], iou_size)
        for x, y in ((a, b), (aa, bb)):
            inter = np.logical_and(x, y).sum()
            union = np.logical_or(x, y).sum()
            ious.append(1.0 if union == 0 else float(inter) / float(union))
        checked += 1
    if not ious:
        print(">>> verify: no compacted files found, skipped")
        return 1.0, 1.0, 0
    ious.sort()
    return float(np.mean(ious)), float(ious[int(0.05 * (len(ious) - 1))]), checked


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Compact prior_sam3/*.npz (uint8, fixed size, compressed).")
    ap.add_argument("--src", default="prior_sam3")
    ap.add_argument("--dst", default="prior_sam3_c448")
    ap.add_argument("--size", type=int, default=448,
                    help="working resolution stored in the npz (default 448)")
    ap.add_argument("--splits", default="train,test",
                    help="comma separated prefixes to convert (default train,test); "
                         "'*' converts every *.npz in --src")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", type=int, default=0,
                    help="after conversion, sample N files and report prior IoU")
    opt = ap.parse_args(argv)

    if not os.path.isdir(opt.src):
        print("ERROR: --src not found: %s" % opt.src)
        return 2
    os.makedirs(opt.dst, exist_ok=True)

    if opt.splits.strip() == "*":
        names = sorted(f for f in os.listdir(opt.src) if f.endswith(".npz"))
    else:
        prefixes = tuple(p.strip() + "_" for p in opt.splits.split(",") if p.strip())
        names = sorted(f for f in os.listdir(opt.src)
                       if f.endswith(".npz") and f.startswith(prefixes))
    if not names:
        print("ERROR: no matching *.npz in %s (splits=%s)" % (opt.src, opt.splits))
        return 2

    print(">>> compact: %d files, %s -> %s, size=%d, jobs=%d"
          % (len(names), opt.src, opt.dst, opt.size, opt.jobs))
    t0 = time.time()
    n_ok = n_skip = n_err = 0
    bytes_in = bytes_out = 0

    def work(name):
        try:
            return compact_one(os.path.join(opt.src, name),
                               os.path.join(opt.dst, name),
                               opt.size, opt.overwrite)
        except Exception as exc:                       # keep going, report at end
            return ("err", name, repr(exc))

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, opt.jobs)) as pool:
        for res in pool.map(work, names):
            done += 1
            if res[0] == "ok":
                n_ok += 1
                bytes_in += res[1]
                bytes_out += res[2]
            elif res[0] == "skip":
                n_skip += 1
            else:
                n_err += 1
                print(">>> ERROR %s: %s" % (res[1], res[2]))
            if done % 1000 == 0 or done == len(names):
                print("    %d/%d  ok=%d skip=%d err=%d"
                      % (done, len(names), n_ok, n_skip, n_err))

    print(">>> done in %.1fs: ok=%d skip=%d err=%d"
          % (time.time() - t0, n_ok, n_skip, n_err))
    if n_ok:
        print(">>> bytes: %.2f GB -> %.3f GB (%.0fx smaller)"
              % (bytes_in / 1e9, bytes_out / 1e9, bytes_in / max(bytes_out, 1)))

    # ---- size gate ----
    sizes = [os.path.getsize(os.path.join(opt.dst, f))
             for f in names if os.path.exists(os.path.join(opt.dst, f))]
    total_out = sum(sizes)
    biggest = max(sizes or [0])
    print(">>> dst total = %.2f GB, largest file = %.0f KB"
          % (total_out / 1e9, biggest / 1e3))
    bad = []
    if total_out > 2e9:
        bad.append("dst total %.2f GB > 2 GB" % (total_out / 1e9))
    if biggest > 2e5:
        bad.append("largest npz %.0f KB > 200 KB" % (biggest / 1e3))
    if n_err:
        bad.append("%d conversion error(s)" % n_err)

    # ---- IoU gate ----
    if opt.verify:
        mean_iou, p5_iou, checked = verify(opt.src, opt.dst, opt.size, opt.verify)
        print(">>> verify (%d files): mean IoU=%.4f  p5=%.4f"
              % (checked, mean_iou, p5_iou))
        if mean_iou < 0.98:
            bad.append("mean prior IoU %.4f < 0.98 (raise --size)" % mean_iou)
        if p5_iou < 0.95:
            bad.append("p5 prior IoU %.4f < 0.95 (raise --size)" % p5_iou)

    if bad:
        print(">>> GATE compact FAILED: " + "; ".join(bad))
        return 1
    print(">>> GATE compact PASSED (size=%d)" % opt.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
