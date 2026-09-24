"""Quick unit test for the region_selector optimizations (no Qwen/SAM3 needed)."""
import sys
import os

import numpy as np

sys.path.insert(0, r"c:\Users\me\Desktop\camouflaged-object-detection-improve-main")
import region_selector as RS


def make_mask(shape, box):
    m = np.zeros(shape, np.float32)
    y0, y1, x0, x1 = box
    m[y0:y1, x0:x1] = 1.0
    return m


H, W = 100, 100
# target at center; aux candidates: one overlapping (hard neg), one far (easy neg), one tiny
target = make_mask((H, W), (40, 60, 40, 60))
overlap = make_mask((H, W), (42, 62, 42, 62))     # IoU ~0.75 with target (hard)
far = make_mask((H, W), (5, 20, 5, 20))            # IoU 0
tiny = make_mask((H, W), (90, 95, 90, 95))         # tiny area
dup = make_mask((H, W), (40, 60, 40, 60))          # duplicate of target (IoU 1.0)
masks = np.stack([target, overlap, far, tiny, dup])
tags = ["fox", "rock", "sky", "leaf", "fox_dup"]
scores = np.array([0.9, 0.85, 0.7, 0.5, 0.88], np.float32)

# --- 1) _assemble IoU hard-negative mining ---
t, a, tt, at = RS._assemble(tags, masks, 0, max_aux=3)
print("[1] aux tags:", at, "| aux sums:", [float(m.sum()) for m in a])
# expect: overlap(400, hardest), far(225), tiny(25) ; dup excluded (IoU=1.0)
assert at[0] == "rock", f"expected hardest-negative 'rock' first, got {at}"
assert at[1] == "sky", f"expected 'sky' second, got {at}"
assert at[2] == "leaf", f"expected 'leaf' third, got {at}"
assert "fox_dup" not in at, "near-duplicate of target must be excluded"
assert [float(m.sum()) for m in a] == [400.0, 225.0, 25.0]
print("    PASS: IoU ranking + duplicate exclusion")

# padding path: only 1 valid non-dup candidate -> 2 zero-padded
t2, a2, tt2, at2 = RS._assemble(tags[:3], masks[:3], 0, max_aux=3)  # fox,rock,sky
assert at2[0] == "rock" and at2[1] == "sky" and at2[2] == "", f"pad expected, got {at2}"
print("    PASS: zero padding when fewer than max_aux")

# --- 2) heuristic_select returns 5-tuple with score ---
out = RS.heuristic_select(tags, masks, scores, max_aux=3)
assert len(out) == 5, f"expected 5-tuple, got {len(out)}"
target_m, aux_m, tag, aux_tags, sc = out
print("[2] heuristic target:", tag, "score:", sc, "| aux:", aux_tags)
assert tag == "fox" and abs(sc - 0.9) < 1e-5, f"target/score mismatch: {tag} {sc}"
print("    PASS: heuristic_select + score")

# --- 3) quality gate logic in process_split ---
# simulate: target area tiny -> gated to zero
regions_dir = r"c:\Users\me\Desktop\camouflaged-object-detection-improve-main\_rs_test"
os.makedirs(regions_dir, exist_ok=True)
out_dir = regions_dir + "/out"
np.savez(os.path.join(regions_dir, "train_test_1_jpg.npz"),
         tags=np.array(["bug"], dtype=object),
         masks=np.stack([tiny]), scores=np.array([0.9], np.float32),
         bbox=np.array([], np.float32))
np.savez(os.path.join(regions_dir, "train_test_2_jpg.npz"),
         tags=np.array(["bug"], dtype=object),
         masks=np.stack([target]), scores=np.array([0.4], np.float32),  # low conf
         bbox=np.array([], np.float32))
np.savez(os.path.join(regions_dir, "train_test_3_jpg.npz"),
         tags=np.array(["bug"], dtype=object),
         masks=np.stack([target]), scores=np.array([0.9], np.float32),  # good
         bbox=np.array([], np.float32))
RS.process_split(regions_dir, out_dir, "train", "heuristic",
                 resume=False, min_target_ratio=0.005, conf_thresh=0.6, max_aux=3)
d1 = np.load(os.path.join(out_dir, "train_test_1_jpg.npz"))
d2 = np.load(os.path.join(out_dir, "train_test_2_jpg.npz"))
d3 = np.load(os.path.join(out_dir, "train_test_3_jpg.npz"))
print("[3] tiny-area target sum:", float(d1["target_mask"].sum()),
      "| low-conf target sum:", float(d2["target_mask"].sum()),
      "| good target sum:", float(d3["target_mask"].sum()))
assert float(d1["target_mask"].sum()) == 0, "tiny area should be gated to zero"
assert float(d2["target_mask"].sum()) == 0, "low confidence should be gated to zero"
assert float(d3["target_mask"].sum()) == target.sum(), "good prior should survive"
print("    PASS: quality gate")

# --- 4) qwen failure fallback (simulated exception) ---
class FakeQwen:
    def generate_with_text(self, img, prompt, system_prompt=None):
        raise RuntimeError("network down")

t, a, tt, at, sc = RS.qwen_select("nonexistent.png", tags, masks, scores, FakeQwen(),
                                  max_aux=3, draw_dir=None)
print("[4] qwen failure -> heuristic fallback target:", tt, "score:", sc)
assert tt == "fox", "fallback should pick heuristic target"
print("    PASS: qwen failure fallback")

import shutil
shutil.rmtree(regions_dir, ignore_errors=True)
print("\nALL TESTS PASSED")
