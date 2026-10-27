"""test_e0e4.py - regression tests for the E0-E4 inference-only pipeline.

No GPU, no Qwen weights, no SAM3 install, no dataset: everything here is numpy /
cv2 / torch-CPU plus source-level guards.  Run from the repo root:

    python tests/test_e0e4.py

Covered:
  [1] frozen config values + guard switches
  [2] ROI extraction (area floor, ordering, padding, normalized box)
  [3] composite transform invertibility
  [4] Qwen reply parsing (OK / ABSTAIN / PARSE_ERROR / SCHEMA_ERROR, region ids)
  [5] prompt parity E2 vs E3 (same schema, one fixed sentence differs)
  [6] E4 fusion: matched, unmatched, no-box fallback, dilation clamp, 1:1 match
  [7] manifest determinism + P0/P1 disjointness
  [8] strict eval: 4-metric backend, bootstrap CI, inventory HOLD
  [9] preflight / source guards (no GT in inference, no filename context)
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.e0e4 import common as C            # noqa: E402
from tools.e0e4 import roi_from_mask as R     # noqa: E402
from tools.e0e4 import caption_qwen as Q      # noqa: E402
from tools.e0e4 import fuse_e4 as F           # noqa: E402
from tools.e0e4 import eval_strict as ES      # noqa: E402
from tools.e0e4 import preflight as PF        # noqa: E402

FAILS = []
TMP = tempfile.mkdtemp(prefix="e0e4_test_")


def check(cond, msg):
    if cond:
        print("    ok   %s" % msg)
    else:
        print("    FAIL %s" % msg)
        FAILS.append(msg)


def expect_error(fn, msg):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - the type is part of the assertion
        print("    ok   %s (%s)" % (msg, type(exc).__name__))
        return exc
    print("    FAIL %s (no exception)" % msg)
    FAILS.append(msg)
    return None


# ---------------------------------------------------------------------------
# [1] config
# ---------------------------------------------------------------------------
def test_config():
    print("[1] frozen config")
    cfg = C.load_config(os.path.join(ROOT, "configs", "e0e4_v1.yaml"))
    C.guard_config(cfg)  # must not raise
    check(cfg.get("version") == "e0e4_v1", "version is e0e4_v1")
    check(int(cfg.get("seed")) == 20261020, "seed 20261020")
    check(C.cfg_get(cfg, "manifest.p0_per_group") == 10, "P0 = 10 per stratum")
    check(C.cfg_get(cfg, "manifest.p1") == {"CAMO": 50, "COD10K-CAM": 100,
                                            "COD10K-NonCAM": 50},
          "P1 = 50/100/50")
    check(C.cfg_get(cfg, "e0.minmax") is False, "min-max output disabled")
    check(abs(C.cfg_get(cfg, "e0.threshold") - 0.5) < 1e-9, "E0 threshold 0.5")
    check(C.cfg_get(cfg, "e0.use_ca") is False,
          "E0 architecture reproduces the 740-key checkpoint (use_ca false)")
    check(C.cfg_get(cfg, "roi.max_roi") == 3, "at most 3 ROIs")
    check(abs(C.cfg_get(cfg, "e4.match_iou") - 0.30) < 1e-9, "E4 IoU 0.30")
    check(C.cfg_get(cfg, "eval.main_metrics") == ["mae", "sm", "avgem", "wfm"],
          "4 main metrics are mae/sm/avgem/wfm")
    check(str(C.cfg_get(cfg, "paths.e0_ckpt")).endswith("Net_epoch_40.pth"),
          "E0 checkpoint pre-registered as Net_epoch_40.pth")
    for key in ("allow_gt_during_inference", "allow_filename_context",
                "allow_noncam_skip", "allow_minmax_output",
                "allow_stage_local_threshold_tuning", "allow_extra_fusion_rules"):
        check(C.cfg_get(cfg, "guards.%s" % key) is False, "guard %s is false" % key)

    bad = json.loads(json.dumps(cfg))
    bad["guards"]["allow_noncam_skip"] = True
    expect_error(lambda: C.guard_config(bad), "a true guard is refused")
    nonempty = os.path.join(TMP, "nonempty_stage")
    os.makedirs(nonempty, exist_ok=True)
    open(os.path.join(nonempty, "keep.txt"), "w").close()
    expect_error(lambda: C.guard_stage_dir(nonempty, overwrite=False),
                 "non-empty stage dir is refused")


# ---------------------------------------------------------------------------
# [2] ROI
# ---------------------------------------------------------------------------
def test_roi():
    print("[2] ROI extraction")
    soft = np.zeros((200, 300), np.float32)
    soft[20:80, 30:130] = 0.9          # 6000 px
    soft[100:130, 200:260] = 0.8       # 1800 px
    soft[5:7, 5:7] = 0.9               # 4 px speck
    binary, rois, status = R.build_rois(soft, 0.5, 100, 0.0005, 3, 0.12)
    check(status == "OK", "status OK")
    check(len(rois) == 2, "speck removed, 2 ROIs kept")
    check(rois[0]["area"] == 6000 and rois[1]["area"] == 1800,
          "sorted by area desc")
    check(rois[0]["bbox_xyxy"] == [30, 20, 130, 80], "bbox = component extent")
    pad = rois[0]["bbox_padded_xyxy"]
    check(pad == [18, 12, 142, 88],
          "12%% padding on width/height (got %s)" % (pad,))
    nx = rois[0]["normalized_xywh"]
    check(len(nx) == 4 and all(0.0 <= v <= 1.0 for v in nx), "normalized in [0,1]")
    check(abs(nx[2] - (pad[2] - pad[0]) / 300.0) < 1e-5, "normalized w correct")

    _, none, st = R.build_rois(np.zeros((50, 50), np.float32), 0.5, 100,
                               0.0005, 3, 0.12)
    check(none == [] and st == "NO_ROI", "empty mask -> NO_ROI")
    check(R.mask_to_binary(np.array([[0.50]], np.float32), 0.5).tolist() == [[True]]
          and R.mask_to_binary(np.array([[0.49]], np.float32), 0.5).tolist()
          == [[False]], "threshold is >= (0.5 counts)")
    check(R.pad_bbox_xyxy([0, 0, 10, 10], 0.5, 100, 100) == [0, 0, 15, 15],
          "padding clipped at the image border")


# ---------------------------------------------------------------------------
# [3] composite
# ---------------------------------------------------------------------------
def test_composite():
    print("[3] composite transform")
    import cv2

    img = np.full((120, 200, 3), 40, np.uint8)
    img[30:60, 40:100] = 200
    binary = np.zeros((120, 200), bool)
    binary[30:60, 40:100] = True
    rois = R.roi_components(binary, min_pixels=10, min_area_fraction=0.0, max_roi=3)
    out = os.path.join(TMP, "composite.png")
    tr = R.build_composite(img, binary, rois, out, max_panel_h=100)
    canvas = cv2.imread(out)
    check(canvas.shape[:2] == tuple(tr["canvas_size"]), "canvas size recorded")
    scale = tr["panel_scale"]
    check(abs(scale - 100.0 / 120.0) < 1e-6, "panel_scale = max_h / H")
    px1, py1, px2, py2 = tr["panels"]["left"]
    sx, sy, panel = R.composite_to_source_point(px1 + 10, py1 + 20, tr)
    check(panel == "left" and abs(sx - 10 / scale) < 1e-6
          and abs(sy - 20 / scale) < 1e-6, "left panel maps back to source")
    rec = tr["crops"][0]
    dx1, dy1, _dx2, _dy2 = rec["dst_xyxy"]
    sx, sy, panel = R.composite_to_source_point(dx1 + 3, dy1 + 4, tr)
    check(panel.startswith("crop"), "crop panel identified")
    check(rec["src_xyxy"][0] <= sx <= rec["src_xyxy"][2]
          and rec["src_xyxy"][1] <= sy <= rec["src_xyxy"][3],
          "crop point maps inside the source box")
    check(R.composite_to_source_point(10 ** 6, 10 ** 6, tr) is None,
          "outside the canvas -> None")


# ---------------------------------------------------------------------------
# [4] Qwen parsing
# ---------------------------------------------------------------------------
GOOD = ('{"objects":[{"name":"frog","sam_text":"a mottled brown frog",'
        '"region_id":2}],"uncertain":false,"no_visible_object":false}')


def test_parsing():
    print("[4] Qwen reply parsing")
    p = Q.parse_response(GOOD, [1, 2, 3])
    check(p["status"] == "OK" and len(p["objects"]) == 1, "valid reply -> OK")
    check(p["objects"][0]["region_id"] == 2, "allowed region_id kept")
    check(p["objects"][0]["sam_text"] == "a mottled brown frog", "sam_text kept")

    check(Q.parse_response("```json\n" + GOOD + "\n```")["status"] == "OK",
          "code fences stripped")
    check(Q.parse_response("Sure, here it is: " + GOOD + " hope it helps")["status"]
          == "OK", "outermost JSON object recovered from prose")

    ab = Q.parse_response('{"objects":[],"uncertain":true,"no_visible_object":true}')
    check(ab["status"] == "ABSTAIN", "explicit no-object -> ABSTAIN")
    check(Q.parse_response('{"objects":[]}')["status"] == "EMPTY",
          "empty list without abstain flag -> EMPTY")
    check(Q.parse_response("I cannot see anything")["status"] == "PARSE_ERROR",
          "prose -> PARSE_ERROR (never becomes a prompt)")
    check(Q.parse_response('{"objects":"frog"}')["status"] == "SCHEMA_ERROR",
          "objects must be a list")

    p = Q.parse_response(GOOD, [1])
    check(p["objects"][0]["region_id"] is None
          and "region_id_unknown" in p["reasons"], "unknown region_id dropped")

    long_text = ('{"objects":[{"name":"frog","sam_text":"a very large green '
                 'brown mottled frog sitting","region_id":1}]}')
    p = Q.parse_response(long_text)
    check(p["status"] == "SCHEMA_ERROR"
          and any(r.startswith("reject_text:word_count") for r in p["reasons"]),
          ">6-word sam_text rejected")

    digits = '{"objects":[{"name":"frog","sam_text":"frog 90 20","region_id":1}]}'
    check(Q.parse_response(digits)["status"] == "SCHEMA_ERROR",
          "coordinates inside sam_text rejected")

    dup = ('{"objects":[{"name":"frog","sam_text":"a frog"},'
           '{"name":"Frog","sam_text":"another frog"}]}')
    p = Q.parse_response(dup)
    check(len(p["objects"]) == 1 and "duplicate_name" in p["reasons"],
          "duplicate name collapsed")

    many = json.dumps({"objects": [
        {"name": "frog", "sam_text": "a mottled frog"},
        {"name": "lizard", "sam_text": "a small lizard"},
        {"name": "bird", "sam_text": "a white bird"},
        {"name": "snake", "sam_text": "a thin snake"},
        {"name": "insect", "sam_text": "a green insect"}]})
    p = Q.parse_response(many)
    check(len(p["objects"]) == 3 and "extra_object_dropped" in p["reasons"],
          "at most 3 objects kept")

    no_text = '{"objects":[{"name":"frog","region_id":1}]}'
    p = Q.parse_response(no_text)
    check(p["objects"][0]["sam_text"] == "frog"
          and "sam_text_from_name" in p["reasons"], "missing sam_text -> name")
    check(Q.parse_response('{"objects":["frog"]}')["status"] == "SCHEMA_ERROR",
          "non-dict object rejected")
    check(Q.parse_response('{"objects":[{"name":"青蛙","sam_text":"青蛙"}]}')["status"]
          == "SCHEMA_ERROR", "non-ASCII names rejected")


# ---------------------------------------------------------------------------
# [5] prompt parity
# ---------------------------------------------------------------------------
def test_prompts():
    print("[5] prompts")
    s2, u2 = Q.build_prompts("e2")
    s3, u3 = Q.build_prompts("e3")
    check(s2 == s3, "identical system prompt for E2/E3")
    check(Q._SCHEMA in u2 and Q._SCHEMA in u3, "same JSON schema in both prompts")
    check(u2 != u3, "one pre-registered branch sentence differs")
    blob = (s2 + u2 + u3).lower()
    for token in ("cod10k", "noncam", "filename", "file name", "jpg",
                  "extract_context", "dataset"):
        check(token not in blob, "prompt free of %r" % token)
    expect_error(lambda: Q.build_prompts("e1"), "E1 has no caption prompt")


# ---------------------------------------------------------------------------
# [6] E4 fusion
# ---------------------------------------------------------------------------
def _blob(shape, y0, y1, x0, x1):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_fusion():
    print("[6] E4 fusion")
    shape = (24, 24)
    e0 = _blob(shape, 2, 8, 2, 8)
    box = _blob(shape, 2, 10, 2, 10)
    text = _blob(shape, 3, 9, 3, 9)

    out = F.fuse_image([{"roi_id": 1, "mask": box, "bbox": [2, 2, 10, 10]}],
                       [{"prompt_id": "obj1", "object_name": "frog",
                         "region_id": 1, "mask": text}], e0)
    check(out["status"] == "OK" and len(out["matches"]) == 1, "matched pair -> OK")
    check(out["stats"]["n_unmatched_text"] == 0, "no text-only candidate")
    check(np.array_equal(out["mask"], box | (text & F.dilate_mask(box, 0.8))),
          "fused = B UNION (T INTERSECT dilate(B))")

    far_text = _blob(shape, 16, 22, 16, 22)
    out = F.fuse_image([{"roi_id": 1, "mask": box, "bbox": [2, 2, 10, 10]}],
                       [{"prompt_id": "obj1", "object_name": "new object",
                         "region_id": None, "mask": far_text}], e0)
    check(out["status"] == "FALLBACK_E0", "no IoU match -> FALLBACK_E0")
    check(np.array_equal(out["mask"], e0), "fallback mask is exactly E0")
    check(out["stats"]["n_unmatched_text"] == 1
          and out["unmatched_text"][0]["object_name"] == "new object",
          "text-only candidate recorded, not merged")

    out = F.fuse_image([], [{"prompt_id": "obj1", "mask": text}], e0)
    check(out["status"] == "FALLBACK_E0_NO_BOX", "no box mask -> NO_BOX fallback")

    huge_text = _blob(shape, 2, 12, 2, 12)
    out = F.fuse_image([{"roi_id": 1, "mask": box, "bbox": [2, 2, 10, 10]}],
                       [{"prompt_id": "obj1", "mask": huge_text}], e0)
    check(out["status"] == "OK", "larger text mask still matches")
    check(not out["mask"][11, 11], "text outside dilate(B) is clipped")
    check(out["mask"][5, 5], "text inside the box is kept")

    two_texts = [{"prompt_id": "obj1", "mask": text},
                 {"prompt_id": "obj2", "mask": text.copy()}]
    out = F.fuse_image([{"roi_id": 1, "mask": box, "bbox": [2, 2, 10, 10]}],
                       two_texts, e0)
    check(len(out["matches"]) == 1 and out["stats"]["n_unmatched_text"] == 1,
          "one-to-one matching (second text left unmatched)")

    box2 = _blob(shape, 2, 10, 12, 20)
    boxes = [{"roi_id": 1, "mask": box, "bbox": [2, 2, 10, 10]},
             {"roi_id": 2, "mask": box2, "bbox": [12, 2, 20, 10]}]
    texts = [{"prompt_id": "obj1", "mask": text}]
    out_a = F.fuse_image(boxes, texts, e0)
    out_b = F.fuse_image(boxes, texts, e0)
    check(np.array_equal(out_a["mask"], out_b["mask"]), "fusion is deterministic")
    check(out_a["matches"][0]["roi_id"] == 1, "best-IoU box chosen")
    check(F.mask_iou(box, box) == 1.0 and F.mask_iou(box, far_text) == 0.0,
          "IoU edges")


# ---------------------------------------------------------------------------
# [7] manifest
# ---------------------------------------------------------------------------
def test_manifest():
    print("[7] manifest determinism")
    from tools.e0e4 import build_manifest as BM

    rows = []
    for group in C.GROUPS:
        for i in range(12):
            rows.append({"image_id": "%s/img%02d" % (group, i), "group": group,
                         "dataset": "COD10K" if group.startswith("COD10K") else "CAMO",
                         "stem": "img%02d" % i})
    quota = {"CAMO": 3, "COD10K-CAM": 4, "COD10K-NonCAM": 5}
    a = C.pick_stratified(rows, quota, 20261020)
    b = C.pick_stratified(rows, quota, 20261020)
    check([r["image_id"] for r in a] == [r["image_id"] for r in b],
          "same seed -> same ids")
    check([r["image_id"] for r in a] !=
          [r["image_id"] for r in C.pick_stratified(rows, quota, 7)],
          "different seed -> different ids")
    counts = {}
    for r in a:
        counts[r["group"]] = counts.get(r["group"], 0) + 1
    check(counts == quota, "per-stratum quota honoured")
    p1 = C.pick_stratified(rows, quota, 20261020,
                           exclude=[r["image_id"] for r in a])
    check(not (set(r["image_id"] for r in a)
               & set(r["image_id"] for r in p1)), "P1 excludes P0")
    expect_error(lambda: C.pick_stratified(rows, {"CAMO": 999}, 1),
                 "quota larger than the pool is refused")
    check(BM.parse_quota("CAMO:50,COD10K-CAM:100")
          == {"CAMO": 50, "COD10K-CAM": 100}, "quota parser")
    expect_error(lambda: BM.parse_quota("CAMO=50"), "bad quota string")

    from tools.e0e4 import make_review as MR

    small = [{"group": g} for g in C.GROUPS]          # one image per stratum
    sizes = MR._sample_sizes(60, small)
    check(sum(sizes.values()) == 3 and all(sizes[g] <= 1 for g in C.GROUPS),
          "blind-review sample is capped by the pool (P0 has only 30)")
    check(sum(MR._sample_sizes(60, rows).values()) == min(60, len(rows)),
          "blind-review sample never exceeds the pool")
    check(sum(MR._sample_sizes(60, rows * 3).values()) == 60,
          "blind-review sample reaches 60 when the pool allows")


# ---------------------------------------------------------------------------
# [8] strict evaluation
# ---------------------------------------------------------------------------
def test_eval():
    print("[8] strict evaluation")
    import eval_metrics

    mod = eval_metrics.load_metric_module()
    gt = np.zeros((24, 24), np.uint8)
    gt[6:18, 6:18] = 255
    rec = mod.GrayscaleMetricRecorderV2(metric_names=eval_metrics.METRIC_NAMES)
    rec.step(gt.copy(), gt)
    vals = rec.show()
    for name in ("mae", "sm", "avgem", "wfm"):
        check(name in vals, "backend reports %s" % name)
    check(abs(vals["mae"]) < 1e-6, "perfect map -> MAE 0")
    check(vals["sm"] >= 0.999 and vals["avgem"] >= 0.999 and vals["wfm"] >= 0.999,
          "perfect map -> S/E/F ~ 1")

    mean, lo, hi = ES.bootstrap_ci([0.1] * 20, n=200, seed=1)
    check(abs(mean - 0.1) < 1e-12 and abs(lo - 0.1) < 1e-12 and abs(hi - 0.1) < 1e-12,
          "bootstrap of a constant sample")
    mean, lo, hi = ES.bootstrap_ci(list(np.linspace(-1, 1, 41)), n=200, seed=1)
    check(lo <= mean <= hi, "bootstrap CI brackets the mean")

    rows = [{"image_id": "A/1", "group": "CAMO", "stage": "E0", "mae": 0.2},
            {"image_id": "A/2", "group": "CAMO", "stage": "E0", "mae": 0.4},
            {"image_id": "A/1", "group": "CAMO", "stage": "E1", "mae": 0.1},
            {"image_id": "A/2", "group": "CAMO", "stage": "E1", "mae": 0.3}]
    agg = ES.aggregate_rows(rows, metrics=("mae",))
    by = {(r["stage"], r["group"]): r for r in agg}
    check(by[("E0", "CAMO")]["mae"] == 0.3 and by[("E0", "ALL")]["n"] == 2,
          "macro average over images")
    deltas = ES.compute_deltas(rows, pairs=(("E1", "E0"),), n=100, seed=1)
    d = {x["metric"]: x for x in deltas}["mae"]
    check(abs(d["mean_delta"] + 0.1) < 1e-9, "paired delta E1-E0 = -0.1")
    check(d["significant"] and d["favours_a"] is True, "constant delta is significant")

    out = os.path.join(TMP, "eval_hold")
    os.makedirs(os.path.join(out, "predictions", "E0", "CAMO"))
    open(os.path.join(out, "predictions", "E0", "CAMO", "a.png"), "wb").close()
    fake_rows = [{"image_id": "CAMO/a", "dataset": "CAMO", "stem": "a"},
                 {"image_id": "CAMO/b", "dataset": "CAMO", "stem": "b"}]
    paths, issues = ES.inventory(fake_rows, ["E0"], out)
    check(("E0", "CAMO/a") in paths and len(issues) == 1
          and "CAMO/b" in issues[0], "missing PNG is a HOLD, not a smaller denominator")
    check(ES.hold(issues, out, fake_rows) == 3
          and os.path.isfile(os.path.join(out, "EVAL_HOLD.md")),
          "HOLD report written, exit code 3")


# ---------------------------------------------------------------------------
# [9] preflight + source guards
# ---------------------------------------------------------------------------
def test_preflight():
    print("[9] preflight checks")
    cfg = C.load_config(os.path.join(ROOT, "configs", "e0e4_v1.yaml"))
    check(PF.check_guards(cfg)["status"] == PF.OK, "guards pass on the frozen config")
    check(PF.check_config_file(cfg)["status"] == PF.OK, "all required config keys")
    check(PF._verdict([{"status": PF.OK}, {"status": PF.WARN}]) == "READY",
          "warnings do not block")
    check(PF._verdict([{"status": PF.BLOCKED}]) == PF.BLOCKED, "one BLOCKED blocks")

    qwen_missing = PF.check_qwen(os.path.join(TMP, "no_qwen"))
    check(qwen_missing["status"] == PF.BLOCKED, "missing Qwen is BLOCKED")
    check("transformers" in qwen_missing["detail"]
          or "not found" in qwen_missing["detail"], "missing Qwen reason is explicit")

    ckpt = os.path.join(ROOT, "model_pth", "OUR", "Net_epoch_40.pth")
    if os.path.isfile(ckpt):
        res = PF.check_e0_checkpoint(ckpt, use_ca=False)
        check(res["status"] == PF.OK, "pre-registered 3ch checkpoint strict-loads (%s)"
              % res["detail"])
        res2 = PF.check_e0_checkpoint(ckpt, use_ca=True)
        check(res2["status"] == PF.BLOCKED,
              "use_ca=True does NOT match the 740-key checkpoint (blocked)")
    else:
        print("    skip pre-registered checkpoint (not in this checkout)")

    sam3 = PF.check_sam3(os.path.join(ROOT, "sam3-main", "sam3-main",
                                      "checkpoints", "sam3.pt"))
    check(sam3["status"] in (PF.OK, PF.BLOCKED), "SAM3 check returns a verdict")
    if sam3["status"] == PF.BLOCKED:
        print("      -> %s" % sam3["detail"])


def test_sources():
    print("[10] source-level guards")
    src = {}
    for name in ("common", "roi_from_mask", "infer_e0", "caption_qwen",
                 "segment_sam3", "fuse_e4", "eval_strict", "make_review",
                 "preflight", "run_e0e4", "build_manifest"):
        with open(os.path.join(ROOT, "tools", "e0e4", name + ".py"),
                  encoding="utf-8") as fh:
            src[name] = fh.read()

    check("extract_context(" not in src["caption_qwen"],
          "caption: no filename context")
    check("prescreen_generate" not in src["caption_qwen"]
          and "filename=" not in src["caption_qwen"],
          "caption: only _call_model(image, prompts) is used")
    check("_call_model" in src["caption_qwen"],
          "caption reuses QwenVLAnnotator._call_model")
    check("qwen_composites" not in src["segment_sam3"],
          "SAM3 never reads the Qwen composite")
    check("composite_path" not in src["segment_sam3"],
          "SAM3 has no path to a composite image")
    for name, text in src.items():
        check("backward(" not in text, "%s: no backward()" % name)
        check("optimizer" not in text, "%s: no optimizer" % name)
    for name in ("infer_e0", "segment_sam3"):
        check("inference_mode" in src[name], "%s: runs under inference_mode" % name)
    check("e0.minmax" in src["infer_e0"] and "minmax=True" not in src["infer_e0"],
          "E0 refuses min-max output")
    check("use_ca" in src["infer_e0"] and "use_ca" in src["preflight"],
          "E0/preflight pin the 740-key architecture (use_ca)")
    check("load_gt" not in src["fuse_e4"] and "GT" in src["fuse_e4"],
          "E4 fusion does not read GT")
    check("empty_cache" in src["run_e0e4"], "run_e0e4 frees VRAM between models")
    order = ["preflight", "manifest", "e0", "roi", "qwen", "sam3", "e4", "eval",
             "review"]
    idx = [src["run_e0e4"].find('"%s"' % s) for s in order]
    check(all(i >= 0 for i in idx) and idx == sorted(idx),
          "stage order is preflight -> ... -> review")
    check("configs" in src["preflight"] or "e0e4_v1.yaml" in src["preflight"],
          "preflight points at the frozen config")


# ---------------------------------------------------------------------------
def main():
    print("=== E0-E4 pipeline tests (no GPU / no weights / no dataset) ===")
    test_config()
    test_roi()
    test_composite()
    test_parsing()
    test_prompts()
    test_fusion()
    test_manifest()
    test_eval()
    test_preflight()
    test_sources()
    shutil.rmtree(TMP, ignore_errors=True)
    if FAILS:
        print("\n!!! %d FAILURE(S):" % len(FAILS))
        for msg in FAILS:
            print("  - %s" % msg)
        return 1
    print("\nAll E0-E4 tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
