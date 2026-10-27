"""Unit test for tools/compact_priors.py + camo prior storage conventions.

No GPU / SAM3 / Qwen needed: everything here is numpy + PIL.

Covers the 2026-10-17 regression where prior_sam3/ was 78 GB of float32 masks at
the ORIGINAL image resolution (largest single file 578 MB) and the training
loader had to resize all of it down to 352 for every sample.

    python tests/test_compact_priors.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import compact_priors as CP
import region_selector as RS


def blob(h, w, cy, cx, ry, rx):
    yy, xx = np.mgrid[0:h, 0:w]
    return ((((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0).astype(np.float32)


test_tmp_root = os.path.join(ROOT, ".test_tmp")
os.makedirs(test_tmp_root, exist_ok=True)
tmp = tempfile.mkdtemp(prefix="prior_compact_test_", dir=test_tmp_root)
src = os.path.join(tmp, "src")
dst = os.path.join(tmp, "dst")
os.makedirs(src)

SIZES = {"train_a_jpg": (2400, 3600), "train_b_jpg": (700, 1000),
         "test_c_jpg": (200, 200)}
for name, (h, w) in SIZES.items():
    t = blob(h, w, h * 0.45, w * 0.5, h * 0.18, w * 0.12)[None]
    t[0, h // 3: h // 3 + 4, w // 4: w // 2] = 1.0        # 4px thin bar
    aux = np.zeros((3, h, w), np.float32)
    aux[0, : h // 2, : w // 2] = 1.0
    aux[1, h // 2:, : w // 2] = blob(h // 2, w // 2, h * 0.2, w * 0.2, h * 0.1, w * 0.1)
    np.savez(os.path.join(src, name + ".npz"), target_mask=t, aux_masks=aux,
             target_tag=np.array(["fox"], dtype=object), bbox=np.array([1, 2, 3, 4]))

# --- 1) resize_binary_mask: shape / dtype / max-pool (thin bar survives) ---
small = CP.resize_binary_mask(np.pad(np.ones((1, 3, 300), np.float32), ((0, 0), (0, 500), (0, 0))),
                              448)
assert small.shape == (1, 448, 448) and small.dtype == np.uint8, small.shape
assert set(np.unique(small).tolist()) <= {0, 1}
row = np.pad(np.zeros((1, 700, 700), np.float32), ((0, 0), (0, 0), (0, 0)))
row[0, 300:302, 100:600] = 1.0          # 2px tall bar
assert CP.resize_binary_mask(row, 448)[0].sum() > 0, "2px bar lost"
print("1) resize_binary_mask ok: uint8 0/1, thin structures preserved")

# --- 2) compact_one: keys kept, opaque tag/bbox, size shrinks dramatically ---
dst_one = os.path.join(tmp, "one")
os.makedirs(dst_one, exist_ok=True)
st, b_in, b_out = CP.compact_one(os.path.join(src, "train_a_jpg.npz"),
                                 os.path.join(dst_one, "train_a_jpg.npz"), 448)
assert st == "ok", st
assert b_out < 0.02 * b_in, (b_in, b_out)
with np.load(os.path.join(dst_one, "train_a_jpg.npz"), allow_pickle=True) as d:
    assert d["target_mask"].dtype == np.uint8 and d["target_mask"].shape == (1, 448, 448)
    assert d["aux_masks"].dtype == np.uint8 and d["aux_masks"].shape == (3, 448, 448)
    assert d["target_tag"].tolist() == ["fox"]
    assert d["bbox"].tolist() == [1, 2, 3, 4]
print("2) compact_one ok: %.1f MB -> %.0f KB (%.0fx), tag/bbox preserved"
      % (b_in / 1e6, b_out / 1e3, b_in / max(b_out, 1)))

# --- 3) full CLI + IoU gate (mean >= 0.98, p5 >= 0.95) ---
rc = CP.main(["--src", src, "--dst", dst, "--size", "448", "--jobs", "2",
              "--verify", "3"])
assert rc == 0, "compact_priors gate failed (rc=%s)" % rc
mean_iou, p5_iou, n = CP.verify(src, dst, 448, 3)
assert mean_iou >= 0.98 and p5_iou >= 0.95, (mean_iou, p5_iou)
print("3) CLI + IoU gate ok: mean=%.4f p5=%.4f over %d files" % (mean_iou, p5_iou, n))

# --- 4) region_selector writes the same format ---
import region_selector as _rs
t_out, a_out = _rs._compact_prior(np.ones((1, 100, 120), np.float32),
                                  np.ones((3, 100, 120), np.float32), 448)
assert (t_out.shape, t_out.dtype) == ((1, 448, 448), np.uint8), (t_out.shape, t_out.dtype)
assert (a_out.shape, a_out.dtype) == ((3, 448, 448), np.uint8), (a_out.shape, a_out.dtype)
assert t_out.sum() == 448 * 448       # all-ones mask stays fully covered
t_nat, a_nat = _rs._compact_prior(np.ones((1, 10, 12), np.float32),
                                  np.ones((3, 10, 12), np.float32), 0)
assert t_nat.dtype == np.float32 and t_nat.shape == (1, 10, 12)   # --prior_size 0 = legacy
print("4) region_selector._compact_prior ok (448 -> uint8, 0 -> native float32)")

# --- 5) loader compatibility: uint8 0/1 -> clip*255 -> {0,255} ---
with np.load(os.path.join(dst, "train_b_jpg.npz"), allow_pickle=True) as d:
    m = d["target_mask"].astype(np.float32)
pil = np.uint8(np.rint(np.clip(m, 0.0, 1.0) * 255.0))[0]
assert set(np.unique(pil).tolist()) <= {0, 255}, np.unique(pil)
print("5) loader compatibility ok: uint8 0/1 -> {0,255} (no code change needed)")

shutil.rmtree(tmp, ignore_errors=True)
try:
    os.rmdir(test_tmp_root)
except OSError:
    pass
print("\nALL TESTS PASSED")
