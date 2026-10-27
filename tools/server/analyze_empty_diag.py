"""Post-hoc binary-mask evaluation for the 16-image EMPTY diagnostic."""
import csv
import json
from pathlib import Path
import statistics
import sys
import tarfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.e0e4 import common as C


def read_mask(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) > 127


def main():
    outputs = ROOT / "outputs"
    with tarfile.open(outputs / "empty_diag_results_20261022.tar.gz") as archive:
        for member in archive.getmembers():
            dest = (outputs / member.name).resolve()
            if outputs.resolve() not in dest.parents or not (member.isfile() or member.isdir()):
                raise ValueError(member.name)
        archive.extractall(outputs)
    out = outputs / "EMPTY_DIAG_20261022"
    base = outputs / "MASK_PROMPT_P0_20261022"
    data = json.loads((out / "DIAGNOSTIC_RESULTS.json").read_text(encoding="utf-8"))
    by_id = {}
    for rec in data["records"]:
        by_id.setdefault(rec["image_id"], {})[rec["mode"]] = rec
    records = []
    for ident, probes in by_id.items():
        ds, stem = ident.split("/", 1)
        gt = read_mask(ROOT / "Dataset/TestDataset" / ds / "GT" / (stem + ".png"))
        group = probes["original_text"]["group"]
        masks = {stage: read_mask(base / "predictions" / stage / ds / (stem + ".png"))
                 for stage in ("E0", "E1", "E3")}
        for mode, rec in probes.items():
            top = read_mask(out / "top_candidate_masks" / mode / ds / (stem + ".png"))
            for threshold in (0.5, 0.3, 0.1):
                masks[mode + "@" + str(threshold)] = top if rec["top_score"] > threshold else np.zeros_like(top)
        for name, mask in masks.items():
            intersection = int((mask & gt).sum())
            union = int((mask | gt).sum())
            records.append({"image_id": ident, "group": group, "variant": name,
                            "mae": float((mask != gt).mean()),
                            "iou": intersection / union if union else 1.0,
                            "fg": float(mask.mean())})
        original = Image.open(ROOT / "Dataset/TestDataset" / ds / "Imgs" / (stem + ".jpg")).convert("RGB")
        panels = [("Original", original), ("GT", Image.fromarray(gt.astype("uint8") * 255)),
                  ("E0", Image.fromarray(masks["E0"].astype("uint8") * 255)),
                  ("E1 box", Image.fromarray(masks["E1"].astype("uint8") * 255))]
        for mode in ("original_text", "category_word", "roi_plus_text"):
            score = probes[mode]["top_score"]
            panels.append(("%s %.5f" % (mode, score),
                           Image.fromarray(masks[mode + "@0.5"].astype("uint8") * 255)))
        fitted = [(label, im.convert("RGB").resize((240, 180))) for label, im in panels]
        sheet = Image.new("RGB", (240 * len(fitted), 235), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, (label, im) in enumerate(fitted):
            sheet.paste(im, (idx * 240, 22))
            draw.text((idx * 240 + 3, 4), label, fill="black")
        draw.text((3, 207), probes["original_text"]["text"] + " | category: " + probes["category_word"]["text"], fill="black")
        sheet_path = out / "REVIEW" / ds / (stem + ".png")
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(sheet_path)
    C.write_csv(str(out / "per_image_binary_metrics.csv"), records)
    summary = []
    for variant in sorted({r["variant"] for r in records}):
        for group in ("ALL", "CAM_ONLY", "NonCAM"):
            chosen = [r for r in records if r["variant"] == variant and
                      (group == "ALL" or (r["group"] == "COD10K-NonCAM") == (group == "NonCAM"))]
            summary.append({"variant": variant, "group": group, "n": len(chosen),
                            "mae": round(statistics.mean(r["mae"] for r in chosen), 5),
                            "iou": round(statistics.mean(r["iou"] for r in chosen), 5)})
    C.write_csv(str(out / "summary_binary_metrics.csv"), summary)
    C.write_json(str(out / "LOCAL_ANALYSIS.json"), {"summary": summary, "n_images": len(by_id)})
    for mode in ("original_text", "category_word", "roi_plus_text"):
        vals = [r["top_score"] for r in data["records"] if r["mode"] == mode]
        print(mode, "min/median/max:", min(vals), statistics.median(vals), max(vals))
    for row in summary:
        if row["variant"] in ("E0", "E1", "E3", "original_text@0.3", "original_text@0.1", "category_word@0.5", "roi_plus_text@0.5"):
            print(row)
    print("EMPTY_DIAG_ANALYSIS_DONE", out)


if __name__ == "__main__":
    main()
