"""Step 3: visually score every candidate and select main/auxiliary regions."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.e0e4.caption_qwen import QwenCaptioner
from tools.semantic_prior.common import extract_json, resolve

SYSTEM = (
    "Evaluate candidate masks for camouflaged object detection using visual evidence. "
    "A visible animal is not automatically a camouflaged target. A candidate may cover "
    "the wrong region even when its label sounds correct. You may abstain. "
    "Return a single JSON object only."
)


def flatten(rec):
    return [dict(c, tag=t["tag"], tag_id=t["tag_id"], proposed_kind=t["kind"])
            for t in rec["tags"] for c in t["candidates"]]


def parse_selection(raw, candidates, max_aux):
    data = extract_json(raw)
    if not isinstance(data, dict) or not isinstance(data.get("scores"), list):
        raise ValueError("Expected object with scores array")
    allowed = {c["candidate_id"] for c in candidates}
    scores, seen = [], set()
    for entry in data["scores"]:
        if not isinstance(entry, dict):
            raise ValueError("Score entry must be an object")
        cid = entry.get("id")
        if cid not in allowed or cid in seen:
            raise ValueError("Unknown or duplicate candidate ID")
        if entry.get("role") not in ("target", "part", "context", "unrelated"):
            raise ValueError("Invalid role")
        for key in ("target_score", "aux_score"):
            value = entry.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 100:
                raise ValueError("Scores must be numbers in [0,100]")
        scores.append({k: entry[k] for k in ("id", "role", "target_score", "aux_score")})
        seen.add(cid)
    if seen != allowed:
        raise ValueError("Must score every candidate exactly once")
    if "main_id" not in data or "aux_ids" not in data:
        raise ValueError("Missing main_id or aux_ids")
    main, aux = data["main_id"], data["aux_ids"]
    by_id = {s["id"]: s for s in scores}
    if main is not None and (not isinstance(main, str) or main not in allowed or by_id[main]["role"] != "target"):
        raise ValueError("Main must be null or a target candidate")
    if not isinstance(aux, list) or len(aux) > max_aux or any(not isinstance(a, str) for a in aux):
        raise ValueError("Invalid auxiliary list")
    if len(set(aux)) != len(aux) or any(a not in allowed or a == main or by_id[a]["role"] not in ("part", "context") for a in aux):
        raise ValueError("Auxiliary IDs must be distinct part/context candidates")
    return {"scores": scores, "main_id": main, "aux_ids": aux,
            "target_ranking": [s["id"] for s in sorted(scores, key=lambda s: (-s["target_score"], s["id"]))],
            "aux_ranking": [s["id"] for s in sorted(scores, key=lambda s: (-s["aux_score"], s["id"]))]}


def make_sheet(image_path, candidates, out, destination):
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    with Image.open(image_path) as im:
        image = im.convert("RGB")
    sheet = Image.new("RGB", (960, 504 + 284 * math.ceil(len(candidates) / 3)), "white")
    sheet.paste(ImageOps.pad(image, (960, 480)), (0, 24))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), "ORIGINAL RGB (not a prediction)", fill="black")
    for i, candidate in enumerate(candidates):
        with Image.open(out / candidate["mask_path"]) as im:
            mask = np.asarray(im.convert("L")) > 0
        arr = np.asarray(image).copy()
        if mask.shape != arr.shape[:2]:
            raise ValueError("Mask and RGB dimensions differ")
        arr[mask] = (0.55 * arr[mask] + 0.45 * np.array([255, 0, 180])).astype("uint8")
        tile = ImageOps.pad(Image.fromarray(arr), (320, 260))
        # Tiny parts would disappear in a whole-frame overview: include a
        # contextual zoom in the lower-right corner of the candidate panel.
        x1, y1, x2, y2 = candidate["bbox_xyxy"]
        if candidate["area_fraction"] < 0.02:
            margin = max(15, (x2 - x1), (y2 - y1))
            crop = Image.fromarray(arr).crop((max(0, x1-margin), max(0, y1-margin),
                                             min(image.width, x2+margin), min(image.height, y2+margin)))
            tile.paste(ImageOps.pad(crop, (100, 100)), (220, 160))
        x, y = (i % 3) * 320, 504 + (i // 3) * 284
        draw.text((x+5, y+5), candidate["candidate_id"], fill="black")
        sheet.paste(tile, (x, y+24))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def build_prompt(candidates, max_aux):
    info = [{k: c[k] for k in ("candidate_id", "tag", "proposed_kind", "sam3_score")} for c in candidates]
    return (
        "Top panel: original image. Below: numbered candidate masks tinted magenta; tiny regions also have a zoom inset. "
        "Judge masks against the original, not label alone. Proposed kinds can be wrong. SAM3 scores measure segmentation "
        "confidence, NOT camouflage relevance.\nCandidates:\n" + json.dumps(info) + "\n"
        "Score EVERY candidate exactly once. role: target (whole camouflaged object), part (part of that object), "
        "context (useful surrounding region), unrelated (irrelevant/wrong mask). "
        "target_score: 0-100 suitability as the MAIN camouflaged-object mask. aux_score: 0-100 usefulness as an auxiliary "
        "part/context prior. These are subjective relevance ratings, not calibrated probabilities. "
        "Choose main_id as one target candidate, or null if none is suitable. Never select a tiny part or background "
        "as main just because no whole-object mask exists. Do not assume an object must be camouflaged. "
        "Choose up to %d distinct aux_ids (part/context only), avoid near-identical duplicate masks. "
        "Parts are positive supporting evidence, not negative masks. "
        "Return exactly: {\"scores\":[{\"id\":\"<candidate ID>\",\"role\":\"<role>\","
        "\"target_score\":0,\"aux_score\":0}],\"main_id\":null,\"aux_ids\":[]}. "
        "Replace placeholders and score every listed candidate. No explanations or markdown."
    ) % max_aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/semantic_prior_pilot_v1.yaml")
    args = ap.parse_args()
    cfg = C.load_config(str(resolve(args.config)))
    out = resolve(cfg["paths"]["out_dir"])
    root = out / "selection"
    root.mkdir(exist_ok=False)
    rows = C.read_csv(str(out / "PILOT_MANIFEST.csv"))
    jobs = []
    for row in rows:
        rec = C.read_json(str(out / "candidates" / (row["image_id"] + ".json")))
        if not rec or C.sha256_file(row["image_path"]) != rec["image_sha256"]:
            raise ValueError("Missing candidates or image hash mismatch")
        candidates = flatten(rec)
        sheet = root / "sheets" / (row["image_id"] + ".jpg")
        make_sheet(row["image_path"], candidates, out, sheet)
        jobs.append((row, candidates, sheet))
    model = QwenCaptioner(cfg["paths"]["qwen_model"], cfg["selection"]["max_tokens"])
    statuses, table, started = Counter(), [], time.time()
    for i, (row, candidates, sheet) in enumerate(jobs, 1):
        replies, error, parsed = [], None, None
        prompt = build_prompt(candidates, cfg["selection"]["max_aux"])
        if not candidates:
            parsed = {"scores": [], "main_id": None, "aux_ids": [], "target_ranking": [], "aux_ranking": []}
            status = "NO_CANDIDATES"
        else:
            for _ in range(2):
                raw = model.caption(str(sheet), SYSTEM, prompt)
                replies.append(raw)
                try:
                    parsed = parse_selection(raw, candidates, cfg["selection"]["max_aux"])
                    error = None
                    break
                except ValueError as exc:
                    error = str(exc)
            status = "ERROR" if error else "SELECTED" if parsed["main_id"] else "ABSTAIN"
        rec = {"image_id": row["image_id"], "group": row["group"], "status": status,
               "candidates": candidates, "selection": parsed, "raw_replies": replies, "error": error,
               "system_prompt": SYSTEM, "user_prompt": prompt, "config_sha256": cfg["_sha256"],
               "image_sha256": row["image_sha256"], "gt_used": False}
        C.write_json(str(root / (row["image_id"] + ".json")), rec)
        if parsed:
            for score in parsed["scores"]:
                table.append({"image_id": row["image_id"], **score,
                              "selected_main": score["id"] == parsed["main_id"],
                              "selected_aux": score["id"] in parsed["aux_ids"],
                              "target_rank": parsed["target_ranking"].index(score["id"]) + 1,
                              "aux_rank": parsed["aux_ranking"].index(score["id"]) + 1})
        statuses[status] += 1
        print("[%d/%d] %s %s main=%s aux=%s error=%s" %
              (i, len(jobs), row["image_id"], status, parsed and parsed["main_id"],
               parsed and parsed["aux_ids"], error), flush=True)
    C.write_csv(str(out / "SELECTION_TABLE.csv"), table)
    C.write_json(str(out / "SELECTION_SUMMARY.json"), {
        "n_images": len(jobs), "status_counts": dict(statuses), "n_candidates_scored": len(table),
        "elapsed_s": round(time.time()-started, 1), "config_sha256": cfg["_sha256"],
        "score_meaning": "Qwen subjective relevance rating, not calibrated confidence", "gt_used": False})
    if statuses["ERROR"]:
        return 2
    print("SEMANTIC_SELECTION_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
