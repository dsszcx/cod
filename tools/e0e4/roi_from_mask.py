"""tools/e0e4/roi_from_mask.py - M0_binary -> ROI list -> Qwen composite image.

E1 input is the **3-channel baseline's own prediction** (``M0_binary``), never a
GT mask.  That is the whole point of the round: the prior quality has to be
measured with the errors the baseline actually makes.

ROI rule (frozen in ``configs/e0e4_v1.yaml``)
---------------------------------------------
1. ``binary = M0_soft >= roi.threshold`` (0.5)
2. connected components, 8-connectivity
3. drop components smaller than ``max(min_pixels, min_area_fraction * H * W)``
4. sort by area desc, ties by (top, left); keep ``roi.max_roi`` (3)
5. pad each box by ``roi.pad_fraction`` (12%) of its own width/height, clipped
6. also export the normalized ``[cx, cy, w, h]`` that SAM3's geometric prompt
   expects (same conversion as ``sam3_regions._bbox_to_normalized_xywh``)

Composite layout (only used by Qwen-E3, never by SAM3)
------------------------------------------------------
    [ original | M0 overlay + boxes | up to 3 numbered ROI crops ]
Every panel is scaled by ONE recorded factor ``panel_scale``; crops have their
own ``scale``.  ``composite_to_source_point`` inverts both, and the transform is
written into ``rois/<dataset>/<stem>.json`` next to the ROIs -- so a reviewer can
map any coordinate Qwen mentions back to the original image.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

if __package__ in (None, ""):  # `python tools/e0e4/roi_from_mask.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from tools.e0e4 import common as C  # noqa: E402

SEPARATOR_PX = 8


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def mask_to_binary(soft, threshold):
    """(H,W) float soft map -> bool binary at ``threshold`` (>=, not >)."""
    soft = np.asarray(soft, dtype=np.float32).squeeze()
    if soft.ndim == 0:                      # a 1x1 map must stay 2-D
        soft = soft.reshape(1, 1)
    if soft.ndim != 2:
        raise C.StageError("M0_soft must be 2-D, got shape %s" % (soft.shape,))
    return soft >= float(threshold)


def pad_bbox_xyxy(bbox, frac, width, height):
    """Grow ``[x1,y1,x2,y2]`` by ``frac`` of its own width/height, clipped."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    px = float(frac) * max(1.0, x2 - x1)
    py = float(frac) * max(1.0, y2 - y1)
    return [int(max(0, math.floor(x1 - px))),
            int(max(0, math.floor(y1 - py))),
            int(min(width, math.ceil(x2 + px))),
            int(min(height, math.ceil(y2 + py)))]


def to_normalized_xywh(bbox, width, height):
    """[x1,y1,x2,y2] absolute -> normalized [cx,cy,w,h] (SAM3 box prompt)."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx = (x1 + x2) / 2.0 / max(1.0, width)
    cy = (y1 + y2) / 2.0 / max(1.0, height)
    w = (x2 - x1) / max(1.0, width)
    h = (y2 - y1) / max(1.0, height)
    return [round(max(0.0, min(1.0, cx)), 6),
            round(max(0.0, min(1.0, cy)), 6),
            round(max(0.001, min(1.0, w)), 6),
            round(max(0.001, min(1.0, h)), 6)]


def roi_components(binary, min_pixels=100, min_area_fraction=0.0005, max_roi=3,
                   pad_fraction=0.12):
    """Connected components of ``binary`` -> ROI dicts (see module docstring)."""
    import cv2

    binary = np.asarray(binary).astype(np.uint8)
    height, width = binary.shape[:2]
    min_area = max(int(min_pixels), int(math.ceil(float(min_area_fraction) * height * width)))
    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    comps = []
    for label in range(1, n_labels):  # 0 = background
        x, y, w, h, area = [int(v) for v in stats[label]]
        if area < min_area:
            continue
        comps.append({"area": area, "x1": x, "y1": y, "x2": x + w, "y2": y + h})
    # area desc, then top, then left -> deterministic under any label order
    comps.sort(key=lambda c: (-c["area"], c["y1"], c["x1"]))
    rois = []
    for i, comp in enumerate(comps[:int(max_roi)]):
        box = [comp["x1"], comp["y1"], comp["x2"], comp["y2"]]
        padded = pad_bbox_xyxy(box, pad_fraction, width, height)
        rois.append({
            "roi_id": i + 1,
            "area": comp["area"],
            "area_fraction": round(comp["area"] / float(height * width), 6),
            "bbox_xyxy": box,
            "bbox_padded_xyxy": padded,
            "normalized_xywh": to_normalized_xywh(padded, width, height),
        })
    return rois


def build_rois(soft, threshold, min_pixels, min_area_fraction, max_roi,
               pad_fraction):
    """soft -> (binary, rois, status).  status is OK / NO_ROI."""
    binary = mask_to_binary(soft, threshold)
    rois = roi_components(binary, min_pixels, min_area_fraction, max_roi,
                          pad_fraction)
    return binary, rois, ("OK" if rois else "NO_ROI")


# ---------------------------------------------------------------------------
# Qwen-E3 composite
# ---------------------------------------------------------------------------
def _resize(img, width, height):
    import cv2
    interp = cv2.INTER_AREA if (width * height) < (img.shape[1] * img.shape[0]) \
        else cv2.INTER_LINEAR
    return cv2.resize(img, (max(1, int(width)), max(1, int(height))),
                      interpolation=interp)


def build_composite(image_rgb, binary, rois, out_path, max_panel_h=448):
    """Write the 3-panel Qwen image and return its invertible transform record.

    Panels: original | M0 overlay with numbered boxes | numbered ROI crops.
    ``panel_scale`` is shared by the first two panels; each crop records its own
    ``scale`` and destination rectangle, which is what makes the mapping back to
    original pixel coordinates exact.
    """
    import cv2

    image_rgb = np.asarray(image_rgb)
    height, width = image_rgb.shape[:2]
    panel_scale = min(1.0, float(max_panel_h) / float(height)) if max_panel_h else 1.0
    pw = max(1, int(round(width * panel_scale)))
    ph = max(1, int(round(height * panel_scale)))

    left = _resize(image_rgb, pw, ph)
    middle = left.copy()
    small = cv2.resize(np.asarray(binary).astype(np.uint8), (pw, ph),
                       interpolation=cv2.INTER_NEAREST)
    red = np.zeros_like(middle)
    red[:, :, 0] = 255
    m = small.astype(bool)
    middle[m] = (0.65 * middle[m] + 0.35 * red[m]).astype(np.uint8)
    contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thickness = max(2, int(round(ph / 300.0)))
    cv2.drawContours(middle, contours, -1, (0, 255, 0), thickness)

    crops = []
    crop_imgs = []
    for roi in rois:
        x1, y1, x2, y2 = roi["bbox_padded_xyxy"]
        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_scale = min(3.0, ph / float(max(1, y2 - y1)))
        cw = max(1, int(round((x2 - x1) * crop_scale)))
        ch = max(1, int(round((y2 - y1) * crop_scale)))
        crop_imgs.append((roi["roi_id"], _resize(crop, cw, ch), ch))
        crops.append({"roi_id": roi["roi_id"], "src_xyxy": [x1, y1, x2, y2],
                      "scale": round(crop_scale, 6), "size": [cw, ch]})

    # box outlines are drawn on the *panel* scale, so they stay aligned
    for roi in rois:
        x1, y1, x2, y2 = [int(round(v * panel_scale)) for v in roi["bbox_padded_xyxy"]]
        cv2.rectangle(middle, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), thickness)
        cv2.putText(middle, str(roi["roi_id"]), (x1 + 3, max(14, y1 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    sep = SEPARATOR_PX
    right_w = sum(c[1].shape[1] for c in crop_imgs) + sep * max(0, len(crop_imgs) - 1)
    canvas_w = 2 * pw + 2 * sep + right_w
    canvas = np.full((ph, max(1, canvas_w), 3), 255, dtype=np.uint8)
    canvas[:, 0:pw] = left
    canvas[:, pw + sep:2 * pw + sep] = middle

    x_cursor = 2 * pw + 2 * sep
    for (roi_id, crop_img, ch), rec in zip(crop_imgs, crops):
        cw = crop_img.shape[1]
        y0 = max(0, (ph - ch) // 2)
        canvas[y0:y0 + ch, x_cursor:x_cursor + cw] = crop_img
        cv2.rectangle(canvas, (x_cursor, y0),
                      (x_cursor + cw - 1, y0 + ch - 1), (0, 0, 0), 1)
        cv2.putText(canvas, str(roi_id), (x_cursor + 4, y0 + max(16, ch // 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        rec["dst_xyxy"] = [x_cursor, y0, x_cursor + cw, y0 + ch]
        x_cursor += cw + sep

    C.ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    if not cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise C.StageError("cv2.imwrite failed: %s" % out_path)

    return {
        "source_size": [int(height), int(width)],
        "canvas_size": [int(ph), int(canvas_w)],
        "panel_scale": round(panel_scale, 6),
        "separator_px": sep,
        "panels": {"left": [0, 0, pw, ph],
                   "middle": [pw + sep, 0, 2 * pw + sep, ph]},
        "crops": crops,
        "composite_path": os.path.relpath(out_path, C.ROOT).replace(os.sep, "/"),
    }


def composite_to_source_point(x, y, transform):
    """Map a canvas coordinate back to original pixels -> (sx, sy, panel)."""
    scale = float(transform.get("panel_scale") or 1.0)
    panels = transform.get("panels") or {}
    for name in ("left", "middle"):
        px1, py1, px2, py2 = panels.get(name, (0, 0, -1, -1))
        if px1 <= x < px2 and py1 <= y < py2:
            return (x - px1) / scale, (y - py1) / scale, name
    for rec in transform.get("crops") or []:
        dx1, dy1, dx2, dy2 = rec["dst_xyxy"]
        if dx1 <= x < dx2 and dy1 <= y < dy2:
            sx1, sy1, _sx2, _sy2 = rec["src_xyxy"]
            k = float(rec["scale"]) or 1.0
            return sx1 + (x - dx1) / k, sy1 + (y - dy1) / k, "crop%d" % rec["roi_id"]
    return None


# ---------------------------------------------------------------------------
# stage driver
# ---------------------------------------------------------------------------
def run_stage(cfg, out_dir, ids, overwrite=False, limit=None):
    """M0_soft.npy -> rois/*.json + qwen_composites/*.png + ROI_STATUS.csv."""
    import cv2

    C.guard_config(cfg)
    roi_cfg = cfg.get("roi") or {}
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
    roi_root = os.path.join(root, "rois")
    comp_root = os.path.join(root, "qwen_composites")
    C.guard_stage_dir(roi_root, overwrite=overwrite)
    C.guard_stage_dir(comp_root, overwrite=overwrite)
    for d in (roi_root, comp_root):
        C.ensure_dir(d)

    status_rows = []
    for idx, row in enumerate(selected, 1):
        dataset, stem = row["dataset"], row["stem"]
        soft_path = os.path.join(e0_root, dataset, stem, "M0_soft.npy")
        if not os.path.isfile(soft_path):
            raise C.StageError("E0 output missing (run stage e0 first): %s" % soft_path)
        soft = np.load(soft_path)
        binary, rois, status = build_rois(
            soft,
            threshold=float(roi_cfg.get("threshold", 0.5)),
            min_pixels=int(roi_cfg.get("min_pixels", 100)),
            min_area_fraction=float(roi_cfg.get("min_area_fraction", 0.0005)),
            max_roi=int(roi_cfg.get("max_roi", 3)),
            pad_fraction=float(roi_cfg.get("pad_fraction", 0.12)))

        image_path, _gt = C.rel_id_to_paths(row, cfg)
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise C.StageError("cannot read image: %s" % image_path)
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        comp_path = os.path.join(comp_root, dataset, stem + ".png")
        transform = build_composite(
            image_rgb, binary, rois, comp_path,
            max_panel_h=int(roi_cfg.get("composite_max_panel_h", 448)))

        rec = {
            "image_id": row["image_id"],
            "dataset": dataset,
            "group": row["group"],
            "status": status,
            "source_size": [int(image_rgb.shape[0]), int(image_rgb.shape[1])],
            "roi_source": "M0_binary (3ch baseline prediction; NOT GT)",
            "rois": rois,
            "transform": transform,
            "config_version": cfg.get("_version"),
        }
        C.write_json(os.path.join(roi_root, dataset, stem + ".json"), rec)
        status_rows.append({
            "image_id": row["image_id"],
            "group": row["group"],
            "status": status,
            "n_roi": len(rois),
            "roi_area_fractions": "|".join("%.4f" % r["area_fraction"] for r in rois),
            "composite": transform["composite_path"],
        })
        C.progress(idx, len(selected), "%s -> %s (%d ROI)"
                   % (row["image_id"], status, len(rois)))

    C.write_csv(os.path.join(root, "ROI_STATUS.csv"), status_rows)
    n_no = sum(1 for r in status_rows if r["status"] != "OK")
    print(">>> ROI stage: %d image(s), %d with NO_ROI -> %s"
          % (len(status_rows), n_no, roi_root))
    return status_rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="M0_binary -> ROI -> Qwen composite")
    ap.add_argument("--config", default=os.path.join(C.ROOT, "configs", "e0e4_v1.yaml"))
    ap.add_argument("--out", default=None, help="run dir (default: config paths.out_dir)")
    ap.add_argument("--ids", required=True, help="image-id list (manifest_p0.txt / p1)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    opt = ap.parse_args(argv)
    try:
        cfg = C.load_config(opt.config)
        out_dir = C.resolve_path(opt.out or C.cfg_get(cfg, "paths.out_dir"))
        run_stage(cfg, out_dir, C.read_id_file(opt.ids),
                  overwrite=opt.overwrite, limit=opt.limit)
    except C.StageError as exc:
        print(">>> BLOCKED: %s" % exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
