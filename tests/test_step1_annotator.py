# -*- coding: utf-8 -*-
"""test_step1_annotator.py - Step1 修复回归测试（2026-10-15）
Run: python tests/test_step1_annotator.py   (无 GPU / 无 Qwen 权重也能跑)
A: fallback_boxes (whole image / quadrants when OpenCV finds nothing)
B: _normalize_items (kill the "'int' object has no attribute 'get'" crash)
C: NonCAM policy (skip train NonCAM, Gate1 denominator = CAM subset)
No GPU / no Qwen weights needed.
"""
import os
import sys
import glob
import tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import llm_annotator as la

FAIL = []
def ck(cond, msg, extra=""):
    print(("PASS  " if cond else "FAIL  ") + msg +
          (("   <%r>" % (extra,)) if (extra != "" and not cond) else ""))
    if not cond:
        FAIL.append(msg)

print("flags: _CV2_OK=%s _LLM_AVAILABLE=%s" % (la._CV2_OK, la._LLM_AVAILABLE))
print("--- B: parser normalisation ---")
ck(la._parse_multi_target("[1, 2]") == [], "B1 bare index list -> []")
ck(la._parse_multi_target('[{"tags":["bird"],"bbox":[1,2,3,4]}]') ==
   [{"tags": ["bird"], "bbox": [1, 2, 3, 4]}], "B2 dict list round-trip")
ck(la._parse_multi_target('{"tags": "frog", "bbox": [0,0,9,9]}') ==
   [{"tags": ["frog"], "bbox": [0, 0, 9, 9]}], "B3 single dict + str tags")
ck(la._parse_multi_target("[null, 3, [1,2]]") == [], "B4 null/int/nested -> []")
ck(la._parse_multi_target("") == [], "B5 empty text -> []")
ck(la._parse_multi_target("please see the image") == [], "B6 prose only -> []")
ck(la._parse_multi_target("I cannot see any camouflaged object in this image") == [],
   "B6a refusal sentence -> [] (was stored as a tag before)")
ck(la._parse_multi_target("brightness changes") == [], "B6b prose about brightness -> []")
r = la._parse_multi_target("fox")
ck(len(r) == 1 and r[0]["tags"] == ["fox"], "B6c single free-text noun still kept", r)
ck(la._parse_multi_target("fox [68,73,279,274], lizard [200,100,500,400]") ==
   [{"tags": ["fox"], "bbox": [68, 73, 279, 274]},
    {"tags": ["lizard"], "bbox": [200, 100, 500, 400]}],
   "B6d legacy regex path keeps BOTH boxes (old split ate the first ']')")
ck(la._parse_multi_target("1 [10,10,50,50], 2 [60,60,90,90]") == [],
   "B6e bare index tags ('1','2') dropped")
r = la._parse_multi_target("1) fox")
ck(len(r) == 1 and r[0]["tags"] == ["fox"], "B6f leftover '1)' numbering stripped", r)
ck(la._parse_prescreen_output("[1, 2]", [[0, 0, 10, 10]]) == [], "B7 prescreen '[1,2]' -> []")

r = la._parse_prescreen_output("1: white bird\n2: green leaf",
                               [[0, 0, 10, 10], [1, 1, 5, 5]])
ck(len(r) == 2 and r[0]["tags"] == ["white bird"] and r[0]["bbox"] == [0, 0, 10, 10]
   and r[1]["bbox"] == [1, 1, 5, 5], "B8 normal numbered lines preserved", r)
ck(la._parse_prescreen_output("1: background\n2: sky",
                              [[0, 0, 10, 10], [1, 1, 5, 5]]) == [],
   "B9 background-only/NonObject -> []")
r = la._parse_prescreen_output("1: camouflaged fish\n2: unknown", [[0, 0, 10, 10], [1, 1, 5, 5]])
ck(len(r) == 1 and r[0]["tags"] == ["camouflaged fish"], "B9b 'unknown' dropped, real name kept", r)
r = la._parse_prescreen_output("1: cat [12, 34, 56, 78]", [[0, 0, 10, 10]])
ck(len(r) == 1 and r[0]["tags"] == ["cat"] and r[0]["bbox"] == [0, 0, 10, 10],
   "B10 coords stripped, box taken from the index", r)
r = la._parse_prescreen_output('[1, {"tags": ["cat"], "bbox": "oops"}]', [[0, 0, 10, 10]])
ck(len(r) == 1 and r[0]["tags"] == ["cat"] and r[0]["bbox"] in ([0, 0, 0, 0], [0, 0, 1, 1]),
   "B11 mixed junk: dict kept, bad bbox -> no box ([0,0,0,0] -> _validate_bbox -> [0,0,1,1])", r)
r = la._parse_prescreen_output('{"tags": ["frog"], "bbox": [3,3,9,9]}', [[0, 0, 10, 10]])
ck(len(r) == 1 and r[0]["bbox"] == [3, 3, 9, 9], "B12 bare dict JSON accepted", r)

print("--- B4: crash regression through the real call chain ---")
def fake_ann(gen_ret):
    a = la.QwenVLAnnotator.__new__(la.QwenVLAnnotator)
    a.prescreen_generate = lambda *args, **kw: gen_ret
    a.generate = lambda *args, **kw: gen_ret
    return a

try:
    out = fake_ann([1, 2]).prescreen_generate_multi("whatever.jpg")
    ck(out == {"tags": [], "bbox": [0, 0, 0, 0]},
       "B13 CRASH REGRESSION: [1,2] -> empty dict, no AttributeError", out)
except Exception as e:
    ck(False, "B13 CRASH REGRESSION: [1,2] -> empty dict, no AttributeError", repr(e))
out = fake_ann([{"tags": ["fish"], "bbox": [1, 2, 3, 4]}]).prescreen_generate_multi("w.jpg")
ck(out == {"tags": ["fish"], "bbox": [1, 2, 3, 4]}, "B14 single-run passthrough unchanged", out)
out = fake_ann([3, 4]).generate_multi("w.jpg")
ck(out == {"tags": [], "bbox": [0, 0, 0, 0]}, "B15 generate_multi junk -> empty dict", out)
out = fake_ann(["cat", None]).prescreen_generate_multi("w.jpg")
ck(out == {"tags": [], "bbox": [0, 0, 0, 0]}, "B16 non-dict junk -> empty dict", out)


print("--- A: fallback_boxes / prescreen fallback path ---")
imgs = sorted(glob.glob(os.path.join(ROOT, "Dataset", "TrainValDataset", "Imgs", "*.jpg")))
img = imgs[0] if imgs else None
if img is None:
    print("SKIP  A: no local dataset image found")
else:
    import cv2
    h, w = cv2.imread(img).shape[:2]
    bx = la.fallback_boxes(img, max_cand=5, mode="quad")
    ck(len(bx) == 5, "A1 quad mode -> 5 boxes", bx)
    ck(bx[0] == [0, 0, w, h], "A2 whole-image box comes first (%dx%d)" % (w, h), bx[0])
    ck(all(len(b) == 4 and 0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h for b in bx),
       "A3 every box valid and inside the image", bx)
    ck(len(la.fallback_boxes(img, max_cand=3, mode="quad")) == 3, "A4 max_cand honoured")
    ck(la.fallback_boxes(img, max_cand=5, mode="whole") == [[0, 0, w, h]],
       "A5 whole mode -> single whole-image box")
    ck(la.fallback_boxes(img, max_cand=5, mode="off") == [], "A6 off mode -> []")
    ck(la.fallback_boxes(os.path.join(ROOT, "no_such_file.jpg"), mode="quad") == [],
       "A7 unreadable image -> [] (caller falls back to legacy)")
    ck(bx[1][2] > w // 2 and bx[1][0] < w // 2 and bx[2][2] > w // 2,
       "A8 quadrants overlap the centre line", bx[1:3])

    # A9: end-to-end prescreen with OpenCV forced empty + a fake Qwen
    orig_ocv = la.opencv_detect_candidates
    orig_call = la.QwenVLAnnotator._call_model
    state = {"calls": 0, "prompts": []}
    la.opencv_detect_candidates = lambda p, max_cand=5, **kw: []
    def fake_call(self, image_path, system_prompt, user_prompt, max_tokens=None):
        state["calls"] += 1
        state["prompts"].append(user_prompt)
        return "1: camouflaged fish\n2: water"       # 'water' is a BACKGROUND tag
    la.QwenVLAnnotator._call_model = fake_call
    try:
        a = la.QwenVLAnnotator.__new__(la.QwenVLAnnotator)
        tmp = tempfile.mkdtemp(prefix="step1test_")
        res = la.QwenVLAnnotator.prescreen_generate(a, img, max_cand=5, tmp_dir=tmp,
                                                    box_fallback="quad")
        ck(state["calls"] == 1 and "5 numbered" in state["prompts"][0],
           "A9 OpenCV empty -> quad fallback used, 1 Qwen call, prompt says 5 boxes",
           (state["calls"], state["prompts"][:1]))
        ck(len(res) == 1 and res[0]["tags"] == ["camouflaged fish"]
           and res[0]["bbox"] == [0, 0, w, h],
           "A10 fallback result parsed with the whole-image box", res)
        ck(os.path.exists(os.path.join(tmp, "numbered_boxes.jpg")),
           "A11 numbered boxes drawn")
        # A12: unusable reply -> one retry with a single whole-image box
        state["calls"] = 0; state["prompts"] = []
        def fake_call2(self, image_path, system_prompt, user_prompt, max_tokens=None):
            state["calls"] += 1
            state["prompts"].append(user_prompt)
            return "[1, 2]" if state["calls"] == 1 else "1: fish"
        la.QwenVLAnnotator._call_model = fake_call2
        res = la.QwenVLAnnotator.prescreen_generate(a, img, max_cand=5, tmp_dir=tmp,
                                                    box_fallback="quad")
        ck(state["calls"] == 2 and "1 numbered" in state["prompts"][1],
           "A12 junk reply -> exactly one retry, prompt says 1 box",
           (state["calls"], state["prompts"][1:]))
        ck(len(res) == 1 and res[0]["bbox"] == [0, 0, w, h],
           "A13 retry result carries the whole-image box", res)
        ck(os.path.exists(os.path.join(tmp, "numbered_whole.jpg")),
           "A14 retry redrew the image with the whole-image box")
        # A15: --box_fallback off must reproduce the old behaviour
        state["calls"] = 0
        def fake_call3(self, image_path, system_prompt, user_prompt, max_tokens=None):
            state["calls"] += 1
            return "fox [68,73,279,274]"
        la.QwenVLAnnotator._call_model = fake_call3
        res = la.QwenVLAnnotator.prescreen_generate(a, img, max_cand=5, tmp_dir=tmp,
                                                    box_fallback="off")
        ck(state["calls"] == 1 and len(res) == 1 and res[0]["tags"] == ["fox"],
           "A15 box_fallback=off -> legacy open-ended path (rollback switch)", res)
    finally:
        la.opencv_detect_candidates = orig_ocv
        la.QwenVLAnnotator._call_model = orig_call

print("--- C: NonCAM policy / Gate1 denominator ---")
tr = os.path.join(ROOT, "Dataset", "TrainValDataset", "Imgs")
ck(la.is_noncam_name("COD10K-NonCAM-1-Aquatic-1-BatFish-9.jpg") is True,
   "C0 NonCAM name detected")
ck(la.is_noncam_name("COD10K-CAM-1-Aquatic-1-BatFish-9.jpg") is False,
   "C0b CAM name not flagged")
ck(la.is_noncam_name("CAMO-1.jpg") is False, "C0c CAMO name not flagged")
from pathlib import Path as _P
ck(la.is_noncam_name(_P("COD10K-NonCAM-2.jpg")) is True, "C0d Path input accepted")
if os.path.isdir(tr):
    names = [f for f in os.listdir(tr)
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    ncam = [n for n in names if la.is_noncam_name(n)]
    ck(len(names) == 7000, "C1 train images == 7000", len(names))
    ck(len(ncam) == 2960, "C2 NonCAM == 2960", len(ncam))
    ck(len(names) - len(ncam) == 4040, "C3 CAM subset (Gate1 denominator) == 4040",
       len(names) - len(ncam))
    try:
        from dataloader_sam3 import is_noncam as rule
        diff = [n for n in names if bool(la.is_noncam_name(n)) != bool(rule(n))]
        ck(diff == [], "C4 llm_annotator rule == dataloader_sam3 rule on 7000 names",
           diff[:5])
    except Exception as e:
        print("SKIP  C4 dataloader_sam3 not importable here: %r" % (e,))
else:
    print("SKIP  C1-C3 no local train image dir")

print("--- C5-C8: batch_annotate NonCAM filtering + json purge (end-to-end, fake Qwen) ---")
import json as _json


def _run_batch(policy, split, files, stale=None):
    d = tempfile.mkdtemp(prefix="step1_c_")
    for n in files:
        with open(os.path.join(d, n), "wb") as fh:
            fh.write(b"\xff\xd8\xff\xe0")            # dummy image bytes
    out = os.path.join(d, "ann.json")
    if stale is not None:
        with open(out, "w", encoding="utf-8") as fh:
            _json.dump(stale, fh)
    seen = {}

    class FakeAnn:
        def __init__(self, model_path=None):
            pass

        def prescreen_generate_multi(self, *a, **k):
            return {"tags": ["fish"], "bbox": [1, 2, 3, 4]}

    def fake_loop(image_files, data_root, sp, annotations, out_path, annotator,
                  num_runs=1, mode="legacy", tmp_dir=None, box_fallback="quad"):
        seen["files"] = [p.name for p in image_files]
        seen["box_fallback"] = box_fallback
        for n in seen["files"]:
            annotations.setdefault(split, {})[n] = annotator.prescreen_generate_multi()
        with open(out_path, "w", encoding="utf-8") as fh:
            _json.dump(annotations, fh)

    o_ann, o_loop = la.QwenVLAnnotator, la._annotate_loop
    la.QwenVLAnnotator, la._annotate_loop = FakeAnn, fake_loop
    try:
        la.batch_annotate(d, out, split, "dummy-model", gpus=[0], mode="prescreen",
                          box_fallback="whole", noncam_policy=policy)
    finally:
        la.QwenVLAnnotator, la._annotate_loop = o_ann, o_loop
    with open(out, "r", encoding="utf-8") as fh:
        return seen, _json.load(fh)


_CAM = "COD10K-CAM-1-Aquatic-1-BatFish-1.jpg"
_NCAM = "COD10K-NonCAM-1-Aquatic-1-BatFish-2.jpg"
_CAMO = "CAMO-3.jpg"
_STALE = {"train": {_NCAM: {"tags": ["bogus"], "bbox": [0, 0, 1, 1]},
                    "old_cam.jpg": {"tags": ["x"], "bbox": [0, 0, 1, 1]}}}

seen, ann = _run_batch("skip_train", "train", [_CAM, _NCAM, _CAMO], _STALE)
ck(seen["files"] == sorted([_CAM, _CAMO]),
   "C5 skip_train: NonCAM file not annotated", seen["files"])
ck(seen["box_fallback"] == "whole", "C6 box_fallback threaded into the loop",
   seen["box_fallback"])
ck(_NCAM not in ann["train"] and "old_cam.jpg" in ann["train"],
   "C7 stale NonCAM json entry purged, other entries kept", sorted(ann["train"]))

seen, ann = _run_batch("annotate", "train", [_CAM, _NCAM, _CAMO], _STALE)
ck(seen["files"] == sorted([_CAM, _NCAM, _CAMO]) and _NCAM in ann["train"],
   "C8 noncam_policy=annotate -> rollback: NonCAM kept and not purged",
   (seen["files"], sorted(ann["train"])))

seen, ann = _run_batch("skip_all", "test", [_CAM, _NCAM, _CAMO])
ck(seen["files"] == sorted([_CAM, _CAMO]),
   "C9 skip_all also applies to the test split", seen["files"])

seen, ann = _run_batch("skip_train", "test", [_CAM, _NCAM, _CAMO])
ck(seen["files"] == sorted([_CAM, _NCAM, _CAMO]),
   "C10 skip_train leaves the test split untouched (MyTest_SAM3 checks it)", seen["files"])

print("=" * 60)
print("%d test(s) failed" % len(FAIL))
for m in FAIL:
    print("  FAILED:", m)
sys.exit(1 if FAIL else 0)
