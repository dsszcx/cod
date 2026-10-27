"""Step 2: independent SAM3 text prompts; retain ranked per-tag candidates."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.e0e4.segment_sam3 import load_processor, rank_candidates, resize_mask_nearest, write_mask_png
from tools.semantic_prior.common import resolve
from tools.semantic_prior.prior_resume import existing_record, check_candidate_record, write_record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/semantic_prior_pilot_v1.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = C.load_config(str(resolve(args.config)))
    out = resolve(cfg["paths"]["out_dir"])
    rows = C.read_csv(str(out / "PILOT_MANIFEST.csv"))
    total = len(rows)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:args.limit]
    inputs = []
    for row in rows:
        rec = C.read_json(str(out / "tags" / (row["image_id"] + ".json")))
        if rec is None or rec["status"] not in ("OK", "EMPTY"):
            raise ValueError("Tags incomplete for " + row["image_id"])
        if C.sha256_file(row["image_path"]) != rec["image_sha256"]:
            raise ValueError("Image hash mismatch: " + row["image_id"])
        inputs.append((row, rec))
    root = out / "candidates"
    root.mkdir(exist_ok=args.resume)
    import numpy as np
    import torch
    from PIL import Image
    if not torch.cuda.is_available():
        raise RuntimeError("This server stage requires CUDA")
    scfg = cfg["sam3"]
    threshold = float(scfg["confidence_threshold"])
    processor = None
    table, statuses = [], Counter()
    started = time.time()
    for idx, (row, rec) in enumerate(inputs, 1):
        record_path = root / (row["image_id"] + ".json")
        tag_hash = C.sha256_file(str(out / "tags" / (row["image_id"] + ".json")))
        previous = existing_record(record_path, row, cfg["_sha256"]) if args.resume else None
        if previous is not None:
            check_candidate_record(previous, tag_hash, out)
            for tag in previous["tags"]:
                statuses[tag["status"]] += 1
                table.extend({"image_id": row["image_id"], "tag_id": tag["tag_id"],
                              "tag": tag["tag"], "kind": tag["kind"], **c}
                             for c in tag["candidates"])
            print("[%d/%d] CANDIDATE_REUSED %s" % (idx, len(inputs), row["image_id"]), flush=True)
            continue
        if processor is None:
            processor = load_processor(cfg["paths"]["sam3_ckpt"], threshold)
        with Image.open(row["image_path"]) as src:
            image = src.convert("RGB")
        width, height = image.size
        per_tag = []
        for tag in rec["tags"]:
            result = dict(tag, candidates=[], scores_returned=[], n_returned=0)
            if not tag["segmentable"]:
                result["status"] = "TEXT_ONLY_ACTION"
            else:
                # A fresh image state isolates prompts; keep image and text
                # encoding in the same BF16 context (SAM3 dtype requirement).
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    state = processor.set_image(image)
                    state = processor.set_text_prompt(prompt=tag["tag"], state=state)
                scores = state["scores"].detach().float().cpu().numpy().reshape(-1)
                masks = state["masks"]
                if len(masks) != len(scores):
                    raise ValueError("SAM3 masks/scores count mismatch")
                result.update(scores_returned=[float(s) for s in scores], n_returned=len(scores))
                for j in rank_candidates(scores, len(scores)):
                    if len(result["candidates"]) >= int(scfg["max_masks_per_tag"]):
                        break
                    if float(scores[j]) < threshold:
                        continue
                    arr = masks[j].detach().float().cpu().numpy().squeeze()
                    if arr.ndim != 2:
                        raise ValueError("Unexpected mask shape: " + str(arr.shape))
                    mask = resize_mask_nearest(arr > 0.5, height, width)
                    if not np.any(mask):
                        continue
                    rank = len(result["candidates"]) + 1
                    cid = "%s_c%02d" % (tag["tag_id"], rank)
                    path = root / "masks" / row["image_id"] / (cid + ".png")
                    write_mask_png(str(path), mask)
                    ys, xs = np.where(mask)
                    candidate = {"candidate_id": cid, "rank_within_tag": rank,
                                 "sam3_score": float(scores[j]), "sam3_index": int(j),
                                 "mask_path": path.relative_to(out).as_posix(),
                                 "area_fraction": float(mask.mean()),
                                 "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]}
                    result["candidates"].append(candidate)
                    table.append({"image_id": row["image_id"], "tag_id": tag["tag_id"],
                                  "tag": tag["tag"], "kind": tag["kind"], **candidate})
                result["status"] = "OK" if result["candidates"] else "EMPTY"
            per_tag.append(result)
            statuses[result["status"]] += 1
            print(row["image_id"], tag["tag_id"], tag["tag"], result["status"],
                  len(result["candidates"]), flush=True)
        write_record(record_path, {
            "image_id": row["image_id"], "group": row["group"], "source_size": [height, width],
            "image_sha256": row["image_sha256"], "tags": per_tag,
            "score_meaning": "SAM3 model confidence, not camouflage or semantic relevance",
            "config_sha256": cfg["_sha256"], "tags_source_sha256": tag_hash})
        print("[%d/%d] IMAGE_DONE" % (idx, len(inputs)), flush=True)
    C.write_csv(str(out / "CANDIDATE_TABLE.csv"), table)
    C.write_json(str(out / "CANDIDATE_SUMMARY.json"), {
        "n_images": len(inputs), "expected_images": total, "complete": len(inputs) == total,
        "n_tags": sum(statuses.values()),
        "tag_status_counts": dict(statuses), "n_candidates": len(table),
        "confidence_threshold": threshold, "max_masks_per_tag": scfg["max_masks_per_tag"],
        "elapsed_s": round(time.time() - started, 1),
        "config_sha256": cfg["_sha256"], "gt_used": False})
    print("SEMANTIC_CANDIDATES_DONE" if len(inputs) == total else "SEMANTIC_CANDIDATES_PARTIAL", flush=True)


if __name__ == "__main__":
    main()
