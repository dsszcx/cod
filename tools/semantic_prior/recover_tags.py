"""Recover stored Qwen replies without rerunning inference or changing content."""
import argparse
from collections import Counter
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import normalize_tags, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/semantic_prior_pilot_v1.yaml")
    args = ap.parse_args()
    cfg = C.load_config(str(resolve(args.config)))
    out = resolve(cfg["paths"]["out_dir"])
    files = sorted((out / "tags").rglob("*.json"))
    records = []
    # Validate all saved replies before touching the original output.
    for path in files:
        rec = C.read_json(str(path))
        errors = []
        for raw in rec["raw_replies"]:
            try:
                tags, rejected = normalize_tags(raw, cfg["tags"]["max_tags"],
                                                 cfg["tags"]["max_words"])
                if rejected and not tags:
                    raise ValueError("All entries rejected")
                break
            except ValueError as exc:
                errors.append(str(exc))
        else:
            raise ValueError("Cannot recover %s: %s" % (path, errors))
        rec.update(tags=tags, rejected=rejected, error=None,
                   status="OK" if tags else "EMPTY", recovery="array_wrapper_parser_v1")
        records.append((path.relative_to(out / "tags"), rec))
    expected = C.read_csv(str(out / "PILOT_MANIFEST.csv"))
    if {r["image_id"] for _, r in records} != {r["image_id"] for r in expected}:
        raise ValueError("Stored replies do not match pilot manifest")
    (out / "tags").rename(out / "tags_format_failed")
    (out / "TAG_SUMMARY.json").rename(out / "TAG_SUMMARY_format_failed.json")
    (out / "TAG_TABLE.csv").rename(out / "TAG_TABLE_format_failed.csv")
    for rel, rec in records:
        C.write_json(str(out / "tags" / rel), rec)
        print(rec["image_id"], [(t["kind"], t["tag"]) for t in rec["tags"]], flush=True)
    C.write_json(str(out / "TAG_SUMMARY.json"), {
        "n_images": len(records), "status_counts": dict(Counter(r["status"] for _, r in records)),
        "n_tags": sum(len(r["tags"]) for _, r in records),
        "tag_kind_counts": dict(Counter(t["kind"] for _, r in records for t in r["tags"])),
        "recovery": "array_wrapper_parser_v1", "inference_rerun": False,
        "config_sha256": cfg["_sha256"]})
    C.write_csv(str(out / "TAG_TABLE.csv"), [
        {"image_id": r["image_id"], "group": r["group"], **t}
        for _, r in records for t in r["tags"]])
    print("SEMANTIC_TAGS_RECOVERED", flush=True)


if __name__ == "__main__":
    main()
