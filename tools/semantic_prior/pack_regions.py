"""Package existing SAM3 candidates; no model, GT or final-selection reads."""
import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import resolve
from lib.region_relations import GEOM_NAMES, ROLE_NAMES, PACKAGE_KEYS, region_geometry, load_region_package
from tools.semantic_prior.prior_resume import existing_package, write_record

SCHEMA = "region_relation_package_v1"


def pack_one(row, source_root, output_root, resume=False):
    if resume:
        previous = existing_package(row, source_root, output_root)
        if previous is not None:
            return previous
    source_path = source_root / "candidates" / (row["image_id"] + ".json")
    source = C.read_json(str(source_path))
    if source is None or source["image_id"] != row["image_id"]:
        raise ValueError("Missing or mismatched candidate record")
    digest = C.sha256_file(row["image_path"])
    if digest != row["image_sha256"] or digest != source["image_sha256"]:
        raise ValueError("Image hash mismatch")
    with Image.open(row["image_path"]) as image:
        w, h = image.size
    if source["source_size"] != [h, w]:
        raise ValueError("Source image dimensions mismatch")
    masks, nodes, roles, scores, ids = [], [], [], [], set()
    for tag in source["tags"]:
        for candidate in tag["candidates"]:
            cid = candidate["candidate_id"]
            if cid in ids:
                raise ValueError("Duplicate candidate ID")
            ids.add(cid)
            mask_path = source_root / candidate["mask_path"]
            with Image.open(mask_path) as im:
                raw = np.asarray(im.convert("L"))
            if raw.shape != (h, w) or not np.isin(raw, [0, 255]).all() or not raw.any():
                raise ValueError("Candidate must be nonempty binary PNG aligned to RGB")
            mask = (raw > 0).astype(np.uint8)
            masks.append(mask)
            role = {"object": "whole", "part": "part", "context": "context"}.get(tag["kind"], "unknown")
            if tag["kind"] == "action":
                raise ValueError("Action must not contain spatial candidates")
            roles.append(ROLE_NAMES.index(role))
            score = float(candidate["sam3_score"])
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid SAM3 confidence")
            scores.append(score)
            nodes.append({"candidate_id": cid, "tag_id": tag["tag_id"], "tag": tag["tag"],
                          "proposed_kind": tag["kind"], "role_hint": role, "mask_index": len(nodes),
                          "mask_source": candidate["mask_path"], "mask_sha256": C.sha256_file(str(mask_path)),
                          "source_status": tag["status"], "role_source": tag.get("role_source", "original_tag_generation_only")})
    n = len(masks)
    masks = np.stack(masks) if masks else np.zeros((0, h, w), np.uint8)
    geom, pair_valid, region_valid = region_geometry(masks)
    if not region_valid.all():
        raise ValueError("Nonempty source candidates became invalid")
    role_hint = np.zeros((n, 4), np.float32)
    if n:
        role_hint[np.arange(n), roles] = 1
    arrays = {"masks": masks, "role_hint": role_hint,
              "role_hint_valid": np.asarray([role != 3 for role in roles], bool),
              "sam3_score": np.asarray(scores, np.float32),
              "relation_geom": geom, "relation_valid": pair_valid}
    destination = output_root / row["image_id"]
    destination.mkdir(parents=True, exist_ok=resume)
    temporary = destination / "regions.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(destination / "regions.npz")
    loaded = load_region_package(destination / "regions.npz")
    if any(not np.array_equal(arrays[key], loaded[key]) for key in PACKAGE_KEYS):
        raise ValueError("Package roundtrip changed numeric arrays")
    counts = dict(Counter(t["status"] for t in source["tags"]))
    status = "HAS_REGIONS" if n else "NO_SEGMENTABLE_TAGS" if not any(t.get("segmentable") for t in source["tags"]) else "NO_CANDIDATES"
    if any("ERROR" in t["status"] for t in source["tags"]):
        status = "PARTIAL_CANDIDATE_ERROR" if n else "CANDIDATE_ERROR"
    metadata = {"schema_version": SCHEMA, "image_id": row["image_id"], "image_sha256": digest,
                "original_hw": [h, w], "mask_hw": [h, w], "coordinate_transform": "identity_native_resolution",
                "candidate_source": str(source_path), "candidate_source_sha256": C.sha256_file(str(source_path)),
                "package_sha256": C.sha256_file(str(destination / "regions.npz")),
                "candidate_generation_status": status, "tag_status_counts": counts,
                "tags_without_masks": [{"tag_id": t["tag_id"], "tag": t["tag"], "kind": t["kind"], "status": t["status"]}
                                       for t in source["tags"] if not t["candidates"]],
                "nodes": nodes, "role_names": ROLE_NAMES, "geometry_names": GEOM_NAMES,
                "geometry_center": "mask_pixel_centroid", "bbox_convention": "xyxy_exclusive",
                "role_hint_is_probability": False, "gt_used": False, "selection_scores_used": False,
                "script_sha256": C.sha256_file(__file__)}
    write_record(destination / "regions.json", metadata)
    return {"image_id": row["image_id"], "n_regions": n, "status": status,
            "n_directed_pairs": int(pair_valid.sum()), "roundtrip_equal": True,
            "package": str((destination / "regions.npz").relative_to(output_root))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/semantic_prior_pilot_v1.yaml")
    ap.add_argument("--output", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = C.load_config(str(resolve(args.config)))
    source_root = resolve(cfg["paths"]["out_dir"])
    output_root = resolve(args.output) if args.output else source_root / "region_packages_v1"
    output_root.mkdir(parents=True, exist_ok=args.resume)
    rows = C.read_csv(str(source_root / "PILOT_MANIFEST.csv"))
    total = len(rows)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:args.limit]
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image IDs")
    results = []
    for row in rows:
        result = pack_one(row, source_root, output_root, resume=args.resume)
        results.append(result)
        print(result, flush=True)
    expected = C.read_json(str(source_root / "CANDIDATE_SUMMARY.json"))
    if len(results) != expected["n_images"] or sum(r["n_regions"] for r in results) != expected["n_candidates"]:
        raise ValueError("Packed counts differ from source candidate summary")
    C.write_csv(str(output_root / "PACKAGE_MANIFEST.csv"), results)
    C.write_json(str(output_root / "PACKAGE_SUMMARY.json"), {
        "schema_version": SCHEMA, "n_images": len(results), "expected_images": total,
        "complete": len(results) == total, "n_regions": sum(r["n_regions"] for r in results),
        "per_image_counts": [r["n_regions"] for r in results],
        "n_directed_pairs": sum(r["n_directed_pairs"] for r in results),
        "status_counts": dict(Counter(r["status"] for r in results)), "all_roundtrips_equal": True,
        "native_masks_preserved": True, "gt_used": False, "training": False,
        "qwen_called": False, "sam3_called": False, "selection_scores_used": False,
        "config_sha256": cfg["_sha256"], "geometry_code_sha256": C.sha256_file(str(resolve("lib/region_relations.py")))})
    print("REGION_PACKAGES_DONE" if len(results) == total else "REGION_PACKAGES_PARTIAL", flush=True)


if __name__ == "__main__":
    main()
