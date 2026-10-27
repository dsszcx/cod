"""Local, post-hoc analysis of ROI unions and NonCAM false positives."""
import json
from pathlib import Path
import statistics
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.e0e4 import common as C


def mask(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 127


def iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def mae(a, b):
    return float((a != b).mean())


def main():
    old = ROOT / "outputs/MASK_PROMPT_P0_20261022"
    new = ROOT / "outputs/QWEN_NO_EXAMPLE_20261022"
    out = ROOT / "outputs/COD_FAILURE_ANALYSIS_20261022"
    out.mkdir(exist_ok=True)
    ids = C.read_id_file(str(old / "manifest_p0.txt"))
    image_rows = {r["image_id"]: r for r in C.read_csv(str(old / "INPUT_MANIFEST.csv"))}
    rows = []
    candidate_rows = []
    for ident in ids:
        ds, stem = ident.split("/", 1)
        group = image_rows[ident]["group"]
        gt = mask(ROOT / "Dataset/TestDataset" / ds / "GT" / (stem + ".png"))
        m0 = mask(new / "predictions/E0" / (ident + ".png"))
        m1 = mask(new / "predictions/E1" / (ident + ".png"))
        m3 = mask(new / "predictions/E3" / (ident + ".png"))
        m4 = mask(new / "predictions/E4" / (ident + ".png"))
        boxes = C.read_json(str(old / "sam3/e1" / (ident + ".json")))
        text = C.read_json(str(new / "sam3/e3" / (ident + ".json")))
        fusion = C.read_json(str(new / "fusions" / (ident + ".json")))
        roi = C.read_json(str(old / "rois" / (ident + ".json")))
        candidates = []
        for prompt in boxes["prompts"]:
            rel = boxes["mask_files"].get(prompt["prompt_id"])
            if not rel:
                continue
            bm = mask(ROOT / rel)
            item = {"image_id": ident, "group": group,
                    "prompt_id": prompt["prompt_id"], "score": prompt["score"],
                    "iou": iou(bm, gt), "mae": mae(bm, gt), "area": float(bm.mean())}
            candidate_rows.append(item)
            candidates.append((item, bm))
        best_item, best_mask = max(candidates, key=lambda pair: pair[0]["iou"])
        first_mask = next(bm for rec, bm in candidates if rec["prompt_id"] == "roi1")
        area = gt.size
        score = max((p["score"] for p in text["prompts"]), default=0.0)
        row = {"image_id": ident, "group": group, "n_roi": len(roi["rois"]),
               "e0_iou": iou(m0, gt), "e1_iou": iou(m1, gt),
               "roi1_iou": iou(first_mask, gt), "best_single_iou_ORACLE": best_item["iou"],
               "best_roi_ORACLE": best_item["prompt_id"],
               "e0_mae": mae(m0, gt), "e1_mae": mae(m1, gt),
               "e3_mae": mae(m3, gt), "e4_mae": mae(m4, gt),
               "e0_fg": float(m0.mean()), "e1_fg": float(m1.mean()), "e3_fg": float(m3.mean()),
               "e3_status": text["status"], "e3_top_filtered_score": score,
               "e1_e3_union_iou": iou(m1, m3), "e4_matches": len(fusion["matches"]),
               "e4_status": fusion["status"],
               "e4_lost_baseline_TP": float((gt & m0 & ~m4).sum() / area),
               "e4_gained_TP": float((gt & ~m0 & m4).sum() / area),
               "e4_removed_FP": float((~gt & m0 & ~m4).sum() / area),
               "e4_added_FP": float((~gt & ~m0 & m4).sum() / area),
               "e3_text": " | ".join(p.get("text", "") for p in text["prompts"])}
        rows.append(row)
        if group != "COD10K-NonCAM":
            original = Image.open(ROOT / "Dataset/TestDataset" / ds / "Imgs" / (stem + ".jpg")).convert("RGB")
            panels = [("Original", original), ("GT", gt), ("E0 baseline", m0),
                      ("E1 union", m1), ("ROI1 only", first_mask),
                      ("Best single (GT oracle)", best_mask), ("New E3 text", m3), ("New E4 fusion", m4)]
            canvas = Image.new("RGB", (8 * 230, 220), "white")
            draw = ImageDraw.Draw(canvas)
            for idx, (label, im) in enumerate(panels):
                if isinstance(im, np.ndarray):
                    im = Image.fromarray(im.astype("uint8") * 255)
                canvas.paste(im.convert("RGB").resize((230, 172)), (idx * 230, 22))
                draw.text((idx * 230 + 3, 4), label, fill="black")
            draw.text((3, 199), ident + " | best single uses GT for diagnosis only", fill="black")
            path = out / "REVIEW" / ds / (stem + ".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(path)
    C.write_csv(str(out / "per_image_diagnosis.csv"), rows)
    C.write_csv(str(out / "box_candidate_diagnosis.csv"), candidate_rows)
    summary = {}
    for group in ("CAMO", "COD10K-CAM", "COD10K-NonCAM"):
        selected = [r for r in rows if r["group"] == group]
        summary[group] = {
            "n": len(selected),
            "means": {k: round(statistics.mean(r[k] for r in selected), 6) for k in
                      ("e0_iou", "e1_iou", "roi1_iou", "best_single_iou_ORACLE", "e0_mae", "e1_mae", "e3_mae", "e4_mae")},
            "union_worse_than_best_single": sum(r["e1_iou"] + 1e-6 < r["best_single_iou_ORACLE"] for r in selected),
            "roi1_better_than_union": sum(r["roi1_iou"] > r["e1_iou"] + 1e-6 for r in selected),
            "best_single_can_beat_baseline": sum(r["best_single_iou_ORACLE"] > r["e0_iou"] + 1e-6 for r in selected),
            "baseline_beats_every_single": sum(r["best_single_iou_ORACLE"] + 1e-6 < r["e0_iou"] for r in selected),
            "nonempty_e3": sum(r["e3_status"] == "OK" for r in selected),
            "e3_filtered_score_ge09": sum(r["e3_top_filtered_score"] >= 0.9 for r in selected),
            "fusion_used": sum(r["e4_matches"] > 0 for r in selected),
            "score_range": [min(r["e3_top_filtered_score"] for r in selected), max(r["e3_top_filtered_score"] for r in selected)],
        }
    C.write_json(str(out / "SUMMARY.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("CAMO failures:")
    for r in sorted((r for r in rows if r["group"] == "CAMO"), key=lambda r: r["e1_iou"] - r["e0_iou"]):
        print(r["image_id"], "E0/E1/ROI1/oracle", *(round(r[k], 4) for k in ("e0_iou", "e1_iou", "roi1_iou", "best_single_iou_ORACLE")))
    print("E4 degraded vs baseline:")
    for r in rows:
        if r["e4_mae"] > r["e0_mae"] + 1e-6:
            print(r["image_id"], {k: round(r[k], 6) for k in ("e0_mae", "e4_mae", "e4_lost_baseline_TP", "e4_added_FP", "e4_gained_TP", "e4_removed_FP")})
    print("COD_FAILURE_ANALYSIS_DONE", out)


if __name__ == "__main__":
    main()
