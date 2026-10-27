"""tools/e0e4/fuse_e4.py - E4: fixed, non-GT fusion of E1 boxes and E3 text masks.

THE RULE (pre-registered in ``configs/e0e4_v1.yaml``; never re-tuned here)
--------------------------------------------------------------------------
1. ``B_b`` = the SAM3 mask of box prompt ``b`` (E1, from ``M0_binary`` ROIs).
   ``T_t`` = the SAM3 mask of text prompt ``t`` (E3, from Qwen on the composite).
2. Pair each ``B_b`` with each ``T_t`` by mask IoU; keep pairs with
   ``IoU >= e4.match_iou`` (0.30); greedy one-to-one assignment (IoU desc,
   then box index, then text index).
3. For a matched pair::

       r      = e4.box_dilate_fraction * max(roi_w, roi_h)   (10% of the long side)
       R_b,t  = B_b  UNION  ( T_t INTERSECT dilate(B_b, r) )

   i.e. the text mask may only *fill in* the box's own neighbourhood; it can
   never add a new blob elsewhere in the image.
4. ``E4 = union(R)`` over all matched pairs.
5. A text prompt that matched no box (``unmatched_text``) is recorded and **not
   added** -- "text says there is a new object" is exactly the failure mode this
   round is measuring, so it must be visible in the report instead of silently
   merged.
6. No matched pair at all -> ``E4 = E0`` (``FALLBACK_E0`` / ``FALLBACK_E0_NO_BOX``).

No GT is read, no per-image threshold, no ranking of objects by confidence in
the fusion itself.

Usage::

    python tools/e0e4/fuse_e4.py --config configs/e0e4_v1.yaml \
        --ids outputs/E0_E4_ZEROTRAIN_V1/manifest_p0.txt
"""
import argparse
import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from tools.e0e4 import common as C  # noqa: E402
from tools.e0e4.segment_sam3 import resize_mask_nearest  # noqa: E402


def dilate_mask(mask, radius_px):
    """Binary dilation with an elliptical structuring element of radius r."""
    import cv2

    mask = np.asarray(mask).astype(np.uint8)
    r = int(round(float(radius_px)))
    if r <= 0:
        return mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask, kernel, iterations=1).astype(bool)


def mask_iou(a, b):
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum()) / float(union)


def match_pairs(box_masks, text_masks, iou_threshold):
    """Greedy one-to-one (box, text) matching -> [(iou, b_idx, t_idx), ...]."""
    cand = []
    for bi, bm in enumerate(box_masks):
        for ti, tm in enumerate(text_masks):
            iou = mask_iou(bm, tm)
            if iou >= float(iou_threshold):
                cand.append((iou, bi, ti))
    cand.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_b, used_t, pairs = set(), set(), []
    for iou, bi, ti in cand:
        if bi in used_b or ti in used_t:
            continue
        used_b.add(bi)
        used_t.add(ti)
        pairs.append((iou, bi, ti))
    return pairs


def fuse_image(box_items, text_items, e0_binary, match_iou=0.30,
               box_dilate_fraction=0.10, min_candidate_pixels=16):
    """Fuse one image; returns ``{mask, status, matches, unmatched_text, stats}``.

    ``box_items``:  ``[{"roi_id", "mask", "bbox"}]``  (bbox = padded ROI xyxy)
    ``text_items``: ``[{"prompt_id", "object_name", "region_id", "mask"}]``
    """
    height, width = e0_binary.shape[:2]
    box_items = [b for b in box_items if b.get("mask") is not None]
    text_items = [t for t in text_items if t.get("mask") is not None]
    pairs = match_pairs([b["mask"] for b in box_items],
                        [t["mask"] for t in text_items], match_iou)

    fused = np.zeros((height, width), dtype=bool)
    matches = []
    for iou, bi, ti in pairs:
        box = box_items[bi]
        text = text_items[ti]
        x1, y1, x2, y2 = box["bbox"]
        radius = float(box_dilate_fraction) * max(1.0, float(x2 - x1),
                                                  float(y2 - y1))
        region = box["mask"] | (text["mask"] & dilate_mask(box["mask"], radius))
        if int(region.sum()) < int(min_candidate_pixels):
            continue
        fused |= region
        matches.append({"roi_id": box["roi_id"], "prompt_id": text["prompt_id"],
                        "object_name": text.get("object_name"),
                        "region_id": text.get("region_id"),
                        "iou": round(float(iou), 6),
                        "dilate_radius_px": round(radius, 2),
                        "area_px": int(region.sum())})

    used_text = {m["prompt_id"] for m in matches}
    unmatched = [{"prompt_id": t["prompt_id"], "object_name": t.get("object_name"),
                  "region_id": t.get("region_id"), "area_px": int(t["mask"].sum())}
                 for t in text_items if t["prompt_id"] not in used_text]

    if matches:
        status = "OK"
        mask = fused
    else:
        status = "FALLBACK_E0_NO_BOX" if not box_items else "FALLBACK_E0"
        mask = np.asarray(e0_binary, bool)
    stats = {"n_box_masks": len(box_items), "n_text_masks": len(text_items),
             "n_pairs": len(pairs), "n_matches": len(matches),
             "n_unmatched_text": len(unmatched),
             "fused_fraction": round(float(fused.mean()), 6) if matches else 0.0}
    return {"mask": mask, "status": status, "matches": matches,
            "unmatched_text": unmatched, "stats": stats}


# ---------------------------------------------------------------------------
# stage driver
# ---------------------------------------------------------------------------
def load_mask(path):
    import cv2

    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise C.StageError("cannot read mask: %s" % path)
    return m > 127


def _box_items(rec_e1, roi_rec):
    bbox = {r["roi_id"]: r["bbox_padded_xyxy"] for r in roi_rec.get("rois", [])}
    items = []
    for prompt in rec_e1.get("prompts", []):
        if prompt["status"] != "OK":
            continue
        rel = (rec_e1.get("mask_files") or {}).get(prompt["prompt_id"])
        roi_id = int(str(prompt["prompt_id"]).replace("roi", ""))
        if not rel or roi_id not in bbox:
            continue
        items.append({"roi_id": roi_id, "mask": load_mask(C.resolve_path(rel)),
                      "bbox": bbox[roi_id]})
    return items


def _text_items(rec_e3):
    items = []
    for prompt in rec_e3.get("prompts", []):
        if prompt["status"] != "OK":
            continue
        rel = (rec_e3.get("mask_files") or {}).get(prompt["prompt_id"])
        if not rel:
            continue
        items.append({"prompt_id": prompt["prompt_id"],
                      "object_name": prompt.get("object_name"),
                      "region_id": prompt.get("region_id"),
                      "mask": load_mask(C.resolve_path(rel))})
    return items


def run_stage(cfg, out_dir, ids, limit=None, overwrite=False):
    """E1 + E3 (+E0 fallback) -> E4 masks, fusions/*.json, E4_STATUS.csv."""
    import cv2

    C.guard_config(cfg)
    ecfg = cfg.get("e4") or {}
    match_iou = float(ecfg.get("match_iou", 0.30))
    dilate_frac = float(ecfg.get("box_dilate_fraction", 0.10))
    min_px = int(ecfg.get("min_candidate_pixels", 16))

    data_root = C.resolve_path(C.cfg_get(cfg, "paths.data_root"))
    datasets = C.cfg_get(cfg, "paths.datasets", ["CAMO", "COD10K"])
    rows, problems = C.manifest_rows(data_root, datasets)
    if problems:
        raise C.StageError("dataset problems: %s" % "; ".join(problems))
    selected = C.rows_for_ids(rows, ids)
    if limit:
        selected = selected[:int(limit)]

    root = os.path.abspath(out_dir)
    pred_root = os.path.join(root, "predictions", "E4")
    fuse_root = os.path.join(root, "fusions")
    C.guard_stage_dir(pred_root, overwrite=overwrite)
    C.guard_stage_dir(fuse_root, overwrite=overwrite)
    C.ensure_dir(pred_root)
    C.ensure_dir(fuse_root)

    status_rows = []
    summary = {"n_images": 0, "status_counts": {}, "n_matched": 0,
               "n_unmatched_text": 0, "iou_values": []}
    for idx, row in enumerate(selected, 1):
        dataset, stem = row["dataset"], row["stem"]
        e0_path = os.path.join(root, "e0", dataset, stem, "M0_binary.png")
        if not os.path.isfile(e0_path):
            raise C.StageError("E4 needs E0 first (missing %s)" % e0_path)
        e0 = load_mask(e0_path)
        rec_e1 = C.read_json(os.path.join(root, "sam3", "e1", dataset, stem + ".json"))
        rec_e3 = C.read_json(os.path.join(root, "sam3", "e3", dataset, stem + ".json"))
        roi_rec = C.read_json(os.path.join(root, "rois", dataset, stem + ".json")) or {}
        if rec_e1 is None or rec_e3 is None:
            raise C.StageError("E4 needs SAM3 e1+e3 first (missing %s)" % row["image_id"])

        height, width = e0.shape[:2]
        boxes = [dict(b, mask=resize_mask_nearest(b["mask"], height, width))
                 for b in _box_items(rec_e1, roi_rec)]
        texts = [dict(t, mask=resize_mask_nearest(t["mask"], height, width))
                 for t in _text_items(rec_e3)]

        fused = fuse_image(boxes, texts, e0, match_iou=match_iou,
                           box_dilate_fraction=dilate_frac,
                           min_candidate_pixels=min_px)
        pred_path = os.path.join(pred_root, dataset, stem + ".png")
        C.ensure_dir(os.path.dirname(pred_path))
        if not cv2.imwrite(pred_path, fused["mask"].astype(np.uint8) * 255):
            raise C.StageError("cv2.imwrite failed: %s" % pred_path)

        rec = {"image_id": row["image_id"], "dataset": dataset, "group": row["group"],
               "status": fused["status"], "stats": fused["stats"],
               "matches": fused["matches"],
               "unmatched_text": fused["unmatched_text"],
               "e0_fraction": round(float(e0.mean()), 6),
               "e4_fraction": round(float(fused["mask"].mean()), 6),
               "e1_status": rec_e1.get("status"), "e3_status": rec_e3.get("status"),
               "rule": {"match_iou": match_iou, "box_dilate_fraction": dilate_frac,
                        "min_candidate_pixels": min_px,
                        "unmatched_text_included": False},
               "config_version": cfg.get("_version")}
        C.write_json(os.path.join(fuse_root, dataset, stem + ".json"), rec)

        status_rows.append({"image_id": row["image_id"], "group": row["group"],
                            "status": fused["status"],
                            "n_matches": fused["stats"]["n_matches"],
                            "n_unmatched_text": fused["stats"]["n_unmatched_text"],
                            "e0_fraction": rec["e0_fraction"],
                            "e4_fraction": rec["e4_fraction"]})
        summary["n_images"] += 1
        summary["status_counts"][fused["status"]] = \
            summary["status_counts"].get(fused["status"], 0) + 1
        summary["n_matched"] += fused["stats"]["n_matches"]
        summary["n_unmatched_text"] += fused["stats"]["n_unmatched_text"]
        summary["iou_values"].extend(m["iou"] for m in fused["matches"])
        C.progress(idx, len(selected), "%s -> %s (%d match, %d text-only)"
                   % (row["image_id"], fused["status"],
                      fused["stats"]["n_matches"],
                      fused["stats"]["n_unmatched_text"]))

    C.write_csv(os.path.join(root, "E4_STATUS.csv"), status_rows)
    ious = summary.pop("iou_values")
    summary["mean_match_iou"] = round(float(np.mean(ious)), 6) if ious else None
    summary.update({"utc": C.utc_now(),
                    "rule": {"match_iou": match_iou,
                             "box_dilate_fraction": dilate_frac,
                             "min_candidate_pixels": min_px},
                    "config": {"version": cfg.get("_version"),
                               "sha256": cfg.get("_sha256")}})
    C.write_json(os.path.join(root, "E4_SUMMARY.json"), summary)
    print(">>> E4: %s" % summary["status_counts"])
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="E4: fixed non-GT fusion")
    ap.add_argument("--config", default=os.path.join(C.ROOT, "configs", "e0e4_v1.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    opt = ap.parse_args(argv)
    try:
        cfg = C.load_config(opt.config)
        out_dir = C.resolve_path(opt.out or C.cfg_get(cfg, "paths.out_dir"))
        run_stage(cfg, out_dir, C.read_id_file(opt.ids), limit=opt.limit,
                  overwrite=opt.overwrite)
    except C.StageError as exc:
        print(">>> BLOCKED: %s" % exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
