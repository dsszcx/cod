"""prior_quality_report.py - Layer-0 evidence: how good are the SAM3/Qwen priors, really?

The pipeline `llm_annotator.py -> sam3_regions.py -> region_selector.py` feeds the
network a 4-channel prior (1 target + 3 aux) plus a neg_flag.  Before any ablation
can be interpreted we need the *numbers* of that prior:

  1. coverage      - how many images actually have a prior file at all?
  2. candidates    - how often does SAM3 return nothing / many boxes?
  3. oracle bound  - best possible IoU over ALL candidates (upper bound reached
                     only if the selector picks perfectly)
  4. selected      - IoU of what the selector ACTUALLY feeds the network,
                     and how many samples are zeroed by the quality gate.

Output: console table + JSON (default outputs/prior_quality/report.json).

Usage
-----
test split only::
    python prior_quality_report.py --regions sam3_regions --splits test ^
        --gt_dir Dataset/TestDataset/CAMO/GT ^
        --img_dir Dataset/TestDataset/CAMO/Imgs

train split + coverage against the whole training set::
    python prior_quality_report.py --regions sam3_regions --splits train ^
        --gt_dir Dataset/TrainValDataset/GT --img_dir Dataset/TrainValDataset/Imgs ^
        --out outputs/prior_quality/train.json

with an existing selection folder (measures the real, gated prior)::
    python prior_quality_report.py --regions sam3_regions --prior prior_sam3 ^
        --splits test --gt_dir Dataset/TestDataset/CAMO/GT
"""
import argparse
import json
import os
import sys

import numpy as np

# region_selector is importable normally (its main block is guarded)
try:
    import region_selector as rs
except Exception as _e:                                             # pragma: no cover
    rs = None
    _RS_ERR = _e

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def npz_to_stem(name):
    """`test_camourflage_00012_jpg.npz` -> `camourflage_00012`."""
    base = os.path.basename(str(name))
    if base.lower().endswith(".npz"):
        base = base[:-4]
    for pre in ("train_", "test_", "val_"):
        if base.startswith(pre):
            base = base[len(pre):]
            break
    if "_" in base:
        stem, ext = base.rsplit("_", 1)
        if 2 <= len(ext) <= 5 and ("." + ext.lower()) in IMG_EXTS:
            return stem
    return base


def family(stem):
    """Dataset family of an image stem: 'COD10K-CAM-1-Aquatic-3-Crab-19' ->
    'COD10K', 'camourflage_00012' -> 'camourflage'.  Used for coverage summaries.
    """
    if "-" in stem:
        return stem.split("-")[0]
    parts = stem.split("_")
    return parts[0] if parts else stem


def index_by_stem(d, exts=None):
    out = {}
    if not d or not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        stem, ext = os.path.splitext(f)
        if exts is not None and ext.lower() not in exts:
            continue
        out.setdefault(stem, os.path.join(d, f))
    return out


def binarize(m):
    m = np.asarray(m)
    if m.dtype == np.uint8 and m.max() <= 1:
        return (m > 0)
    return (m > (0.5 if m.dtype.kind == "f" else 127))


def iou(a, b):
    """IoU of two boolean masks.

    The two masks may live at different resolutions: a prior npz is stored at
    `--prior_size` (448) while the GT is at the image resolution, so the first
    mask is nearest-resized to the second before comparing.  Without this the
    report crashed with a broadcast error on compacted priors.
    """
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    if a.shape != b.shape:
        from PIL import Image
        a = np.asarray(Image.fromarray(a.astype(np.uint8) * 255).resize(
            (b.shape[1], b.shape[0]), Image.NEAREST)) > 127
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def pct(v, q=(25, 50, 75)):
    if not v:
        return {}
    a = np.asarray(v, np.float64)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "min": float(a.min()),
            "max": float(a.max()), **{("p%d" % qq): float(np.percentile(a, qq)) for qq in q}}


def rate(v, thr):
    if not v:
        return 0.0
    return float(np.mean(np.asarray(v) >= thr))


def analyze(regions_dir, split, gt_by_stem, img_by_stem, prior_dir=None,
            min_target_ratio=0.005, conf_thresh=0.6, max_aux=3,
            contains=None, limit=None, verbose=False):
    import glob
    import cv2
    from collections import Counter

    files = sorted(glob.glob(os.path.join(regions_dir, "%s_*.npz" % split)))
    if contains:
        files = [f for f in files if contains in os.path.basename(f)]
    if limit:
        files = files[:limit]

    rep = {"regions_dir": os.path.abspath(regions_dir), "split": split,
           "n_regions_npz": len(files), "contains": contains,
           "quality_gate": {"min_target_ratio": min_target_ratio,
                            "conf_thresh": conf_thresh, "max_aux": max_aux}}
    if not files:
        rep["error"] = "no %s_*.npz found in %s" % (split, regions_dir)
        return rep

    # ---------------- coverage ----------------
    stems_region = [npz_to_stem(f) for f in files]
    stems_img = set(img_by_stem)
    stems_gt = set(gt_by_stem)
    if img_by_stem:
        have = [s for s in stems_img if s in set(stems_region)]
        miss = [s for s in stems_img if s not in set(stems_region)]
        pref = Counter(family(s) for s in miss)
        rep["coverage"] = {
            "n_images_in_img_dir": len(stems_img),
            "n_images_with_regions": len(have),
            "coverage_ratio": len(have) / max(len(stems_img), 1),
            "n_images_without_regions": len(miss),
            "missing_top_prefix": pref.most_common(8),
        }
        pref_h = Counter(family(s) for s in have)
        rep["coverage"]["with_regions_by_prefix"] = pref_h.most_common(8)
    if stems_gt:
        rep["coverage_gt"] = {"n_gt": len(stems_gt),
                              "n_region_matched_gt": len([s for s in stems_region if s in stems_gt])}

    n_empty = 0
    cand_counts, oracle, oracle_gated = [], [], []
    n_no_good_cand, n_no_good_cand_g = 0, 0
    n_gate_area, n_gate_conf, n_zero_prior, n_prior_ok = 0, 0, 0, 0
    n_prior_dir_hits = 0          # how many samples were read from prior_dir
    sel_iou, sel_score, tags_top = [], [], Counter()
    has_gt = bool(stems_gt)

    for f in files:
        stem = npz_to_stem(f)
        with np.load(f, allow_pickle=True) as d:
            tags = [str(t) for t in d["tags"]]
            masks = np.asarray(d["masks"], np.float32)
            scores = np.asarray(d["scores"], np.float64).reshape(-1)
        for t in tags:
            tags_top[t] += 1
        n = 0 if masks.ndim < 3 else masks.shape[0]
        cand_counts.append(n)
        if n == 0:
            n_empty += 1

        g = None
        if has_gt and stem in stems_gt:
            g = cv2.imread(gt_by_stem[stem], cv2.IMREAD_GRAYSCALE)
            if g is not None:
                g = binarize(g)

        # ---- oracle: best IoU over ALL candidates vs the area-gated subset ----
        if n > 0 and g is not None:
            H, W = masks.shape[-2:]
            binm = masks > 0.5
            ious = [iou(binm[i], g) for i in range(n)]
            areas = [float(binm[i].sum()) / (H * W) for i in range(n)]
            keep = [i for i in range(n) if min_target_ratio <= areas[i] <= 0.95]
            best = max(ious)
            oracle.append(best)
            if best < 0.3:
                n_no_good_cand += 1
            bg = max(ious[i] for i in keep) if keep else 0.0
            oracle_gated.append(bg)
            if bg < 0.3:
                n_no_good_cand_g += 1

        # ---- what the pipeline actually feeds the network ----
        pf = os.path.join(prior_dir, os.path.basename(f)) if prior_dir else None
        if pf and os.path.exists(pf):
            n_prior_dir_hits += 1
            with np.load(pf, allow_pickle=True) as d:
                tgt = np.asarray(d["target_mask"], np.float32)
            if float(tgt.sum()) == 0:
                n_zero_prior += 1
                if g is not None:
                    sel_iou.append(0.0)
            else:
                n_prior_ok += 1
                if g is not None:
                    sel_iou.append(iou(binarize(tgt.squeeze()), g))
        elif rs is not None and n > 0 and g is not None:
            target, _aux, _tag, _tags, sc = rs.heuristic_select(
                tags, masks, scores, max_aux=max_aux)
            H, W = target.shape[-2:]
            area_ratio = float((target > 0.5).sum()) / (H * W)
            ka, kc = area_ratio < min_target_ratio, sc < conf_thresh
            sel_score.append(sc)
            if ka or kc:
                n_zero_prior += 1
                n_gate_area += int(ka)
                n_gate_conf += int(kc)
                sel_iou.append(0.0)
            else:
                n_prior_ok += 1
                sel_iou.append(iou(binarize(target.squeeze()), g))
        elif n == 0 and g is not None:
            n_zero_prior += 1
            sel_iou.append(0.0)
        if verbose and len(cand_counts) % 200 == 0:
            print("    ... %d/%d" % (len(cand_counts), len(files)))

    rep["candidates"] = {
        "n_empty_candidate_files": n_empty,
        "empty_ratio": n_empty / max(len(files), 1),
        "cands_per_image": pct(cand_counts),
        "n_images_with_candidates": int(np.sum(np.asarray(cand_counts) > 0)),
    }
    if oracle:
        rep["oracle"] = {
            "n": len(oracle), "iou_all": pct(oracle), "iou_area_gated": pct(oracle_gated),
            "ratio_iou_ge_0.5": rate(oracle, 0.5),
            "ratio_iou_ge_0.7": rate(oracle, 0.7),
            "ratio_iou_ge_0.5_area_gated": rate(oracle_gated, 0.5),
            "n_images_no_cand_reaches_0.3": n_no_good_cand,
            "ratio_no_cand_reaches_0.3": n_no_good_cand / max(len(oracle), 1),
            "ratio_no_cand_reaches_0.3_area_gated": n_no_good_cand_g / max(len(oracle_gated), 1),
        }
    sel = [v for v in sel_iou if v is not None]
    if sel:
        rep["selected"] = {
            "source": ((os.path.abspath(prior_dir) + " [%d npz]" % n_prior_dir_hits)
                       if (prior_dir and n_prior_dir_hits) else
                       ("heuristic_simulated (no npz found under '%s')" % prior_dir
                        if prior_dir else "heuristic_simulated")),
            "n": len(sel), "iou": pct(sel),
            "ratio_iou_ge_0.5": rate(sel, 0.5),
            "ratio_iou_ge_0.3": rate(sel, 0.3),
            "ratio_iou_lt_0.3": float(np.mean(np.asarray(sel) < 0.3)),
            "n_zero_prior": n_zero_prior,
            "zero_prior_ratio": n_zero_prior / max(len(sel), 1),
            "n_kept": n_prior_ok,
            "gated_by_area": n_gate_area, "gated_by_conf": n_gate_conf,
        }
        if sel_score:
            rep["selected"]["sam3_score_of_selected"] = pct(sel_score)
    rep["tags_top"] = tags_top.most_common(10)
    return rep


def print_report(rep):
    print("\n" + "=" * 78)
    print("prior quality report  [%s]  regions=%s" % (rep["split"], rep["regions_dir"]))
    print("=" * 78)
    if "error" in rep:
        print("!! " + rep["error"])
        return
    c = rep.get("coverage")
    if c:
        print("coverage")
        print("  images in Imgs dir       : %d" % c["n_images_in_img_dir"])
        print("  images WITH sam3 regions : %d  (%.1f%%)"
              % (c["n_images_with_regions"], 100 * c["coverage_ratio"]))
        print("  images WITHOUT regions   : %d" % c["n_images_without_regions"])
        if c["missing_top_prefix"]:
            print("  missing by prefix        : %s" % c["missing_top_prefix"])
    cand = rep["candidates"]
    cp = cand["cands_per_image"]
    print("candidates (%d npz files)" % rep["n_regions_npz"])
    print("  empty-candidate files    : %d  (%.1f%%)"
          % (cand["n_empty_candidate_files"], 100 * cand["empty_ratio"]))
    print("  cands/image mean/med/p75/max: %.2f / %.0f / %.0f / %.0f"
          % (cp["mean"], cp["median"], cp["p75"], cp["max"]))
    o = rep.get("oracle")
    if o:
        print("oracle upper bound (best IoU over ALL candidates), n=%d" % o["n"])
        print("  all candidates  mean %.3f  med %.3f  p25 %.3f  p75 %.3f"
              % (o["iou_all"]["mean"], o["iou_all"]["median"],
                 o["iou_all"]["p25"], o["iou_all"]["p75"]))
        print("  IoU>=0.5 %.1f%%   IoU>=0.7 %.1f%%"
              % (100 * o["ratio_iou_ge_0.5"], 100 * o["ratio_iou_ge_0.7"]))
        print("  area-gated      mean %.3f   IoU>=0.5 %.1f%%"
              % (o["iou_area_gated"]["mean"], 100 * o["ratio_iou_ge_0.5_area_gated"]))
        print("  unrecoverable (no candidate reaches IoU 0.3): %d  (%.1f%%)"
              % (o["n_images_no_cand_reaches_0.3"], 100 * o["ratio_no_cand_reaches_0.3"]))
    s = rep.get("selected")
    if s:
        print("actually fed to the network (%s, n=%d)" % (s["source"], s["n"]))
        print("  IoU       mean %.3f  med %.3f  p25 %.3f  p75 %.3f"
              % (s["iou"]["mean"], s["iou"]["median"], s["iou"]["p25"], s["iou"]["p75"]))
        print("  IoU>=0.5 %.1f%%   IoU>=0.3 %.1f%%   IoU<0.3 (misleading) %.1f%%"
              % (100 * s["ratio_iou_ge_0.5"], 100 * s["ratio_iou_ge_0.3"],
                 100 * s["ratio_iou_lt_0.3"]))
        print("  zero prior: %d  (%.1f%%)   [gated by area %d, by conf %d]"
              % (s["n_zero_prior"], 100 * s["zero_prior_ratio"],
                 s["gated_by_area"], s["gated_by_conf"]))
        if "sam3_score_of_selected" in s:
            print("  sam3 score of selected target: mean %.3f"
                  % s["sam3_score_of_selected"]["mean"])
    print("top tags: %s" % ", ".join("%s(%d)" % (t, n) for t, n in rep["tags_top"][:8]))
    print("=" * 78 + "\n")


def main():
    p = argparse.ArgumentParser(
        description="Layer-0 prior quality report (SAM3 candidates / selected priors vs GT).")
    p.add_argument("--regions", type=str, default="sam3_regions")
    p.add_argument("--prior", type=str, default=None,
                   help="prior_sam3 dir; when given the REAL (quality-gated) prior is measured")
    p.add_argument("--splits", type=str, default="test", help="comma-separated, e.g. train,test")
    p.add_argument("--gt_dir", type=str, default=None)
    p.add_argument("--img_dir", type=str, default=None)
    p.add_argument("--contains", type=str, default=None, help="substring filter on npz file name")
    p.add_argument("--limit", type=int, default=None, help="only the first N npz files (debug)")
    p.add_argument("--min_target_ratio", type=float, default=0.005)
    p.add_argument("--conf_thresh", type=float, default=0.6)
    p.add_argument("--max_aux", type=int, default=3)
    p.add_argument("--out", type=str, default="outputs/prior_quality/report.json")
    p.add_argument("--verbose", action="store_true")
    opt = p.parse_args()

    gt_by_stem = index_by_stem(opt.gt_dir, IMG_EXTS)
    img_by_stem = index_by_stem(opt.img_dir, IMG_EXTS)
    if opt.gt_dir and not gt_by_stem:
        print("[warn] no GT found in %s" % opt.gt_dir)
    if opt.img_dir and not img_by_stem:
        print("[warn] no images found in %s" % opt.img_dir)
    if opt.prior and not os.path.isdir(opt.prior):
        print("[warn] --prior '%s' does not exist -> the REAL gated prior cannot be "
              "measured; 'selected' falls back to a heuristic simulation" % opt.prior)
    if rs is None:
        print("[warn] region_selector not importable (%s); 'selected' stats need an existing --prior dir"
              % _RS_ERR)

    result = {"config": vars(opt),
              "gt_dir_indexed": len(gt_by_stem), "img_dir_indexed": len(img_by_stem),
              "reports": {}}
    for split in [s.strip() for s in opt.splits.split(",") if s.strip()]:
        rep = analyze(opt.regions, split, gt_by_stem, img_by_stem, prior_dir=opt.prior,
                      min_target_ratio=opt.min_target_ratio, conf_thresh=opt.conf_thresh,
                      max_aux=opt.max_aux, contains=opt.contains, limit=opt.limit,
                      verbose=opt.verbose)
        print_report(rep)
        result["reports"][split] = rep

    if opt.out:
        d = os.path.dirname(os.path.abspath(opt.out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(opt.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("[json] %s" % os.path.abspath(opt.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
