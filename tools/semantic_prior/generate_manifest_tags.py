"""Generate component tags for an explicit RGB-only manifest, no sampling."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.e0e4.caption_qwen import QwenCaptioner
from tools.semantic_prior.common import normalize_tags, resolve, extract_json
from tools.semantic_prior.prior_resume import existing_record, write_record

SYSTEM = (
    "Describe visible components in the supplied original photograph. "
    "Provide possible whole entities, physical parts and environmental regions. "
    "Do not decide a main target, camouflage presence or which region is most important. "
    "Return JSON only."
)


def build_prompt(max_tags=8, max_words=6):
    return (
        "Return one JSON object with a top-level key tags whose value is an array. "
        "Each array entry has exactly two keys: tag and kind.\n"
        "tag: a concrete English noun phrase of at most %d words, suitable as a segmentation prompt.\n"
        "kind: object, part, context, or action.\n"
        "object means a whole visible entity. part means a physical component of an entity, "
        "not merely its color, texture, pattern or shape. context means a visible environmental "
        "surface, material or region, not an abstract scene or generic environment. "
        "action is optional, retained as text only, not a spatial mask.\n"
        "Use up to %d total tags. Prefer one direct name per whole entity; do not repeat "
        "the same entity with synonyms, color variants or habitat phrases. "
        "Include identifiable physical parts and distinct surrounding regions when visible. "
        "Use color modifiers only as part of an entity phrase, never as standalone object tags. "
        "Do not invent a part or an animal to fill a category. No fixed minimum per kind. "
        "If no components can be identified, the tags array may be empty. "
        "Do not output scores, ranks, main-target selections, coordinates, placeholders or explanations."
    ) % (max_words, max_tags)


def read_inputs(path):
    rows = C.read_csv(str(path))
    required = {"image_id", "image_path", "image_sha256", "width", "height"}
    if not rows or any(set(r) != required for r in rows):
        raise ValueError("Expected RGB-only manifest fields; do not pass training/GT manifests")
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image IDs")
    return [dict(r, group="TrainVal") for r in rows]


def parse_tags(raw, max_tags, max_words):
    tags, rejected = normalize_tags(raw, max_tags, max_words)
    if rejected and not tags:
        raise ValueError("All returned tags were invalid")
    return tags, rejected


def parse_repair_tags(raw, tc):
    try:
        normalized, rejected = parse_tags(raw, tc['max_tags'], tc['max_words'])
        return normalized, rejected, None
    except ValueError:
        obj = extract_json(raw)
        entries = obj.get('tags') if isinstance(obj, dict) else obj
        if not isinstance(entries, list) or not entries or not all(isinstance(t, str) for t in entries):
            raise ValueError('Corrective reply still has invalid tag schema')
        # Preserve generated words, never infer a missing semantic role.
        normalized, rejected = parse_tags(json.dumps({'tags': [dict(tag=t, kind='object') for t in entries]}),
                                           tc['max_tags'], tc['max_words'])
        for tag in normalized:
            tag.update(kind='unknown', segmentable=True, role_source='missing_in_corrective_response')
        return normalized, rejected, 'untyped_list_unknown_roles_v1'


def repair_record(model, row, record, tc):
    """One fixed, category-free corrective call; retain all original replies."""
    prompt = (
        'Look at the photograph and name the visible things, not the instructions. '
        'Reply only as JSON: {"tags": [...]}. Each entry has tag (a short English noun phrase) '
        'and kind (object, part, context, or action). Object is a whole entity; part is a '
        'physical component; context is a visible surrounding surface or material. '
        'Use at most %d tags, each at most %d words. No scores, coordinates, main target, '
        'schema descriptions or placeholders. If nothing is identifiable, use {"tags": []}.'
    ) % (tc['max_tags'], tc['max_words'])
    raw = model.caption(row['image_path'], SYSTEM, prompt)
    record = dict(record)
    try:
        normalized, rejected, format_recovery = parse_repair_tags(raw, tc)
        error = None
    except (ValueError, TypeError) as exc:
        normalized, rejected, error, format_recovery = [], [], str(exc), None
    record.update(tags=normalized, rejected=rejected, error=error,
                  status='ERROR' if error else 'OK' if normalized else 'EMPTY',
                  raw_replies=record['raw_replies'] + [raw],
                  attempts=record['attempts'] + [dict(prompt=prompt, raw=raw, error=error)],
                  recovery_prompt_version='short_components_no_examples_v1',
                  recovery_max_calls=1, format_recovery=format_recovery)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/relation_train_priors_pilot_v1.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Fixed-order smoke run only")
    ap.add_argument("--repair-errors", action="store_true", help="One fixed corrective call for parse errors")
    args = ap.parse_args()
    cfg = C.load_config(str(resolve(args.config)))
    rows = read_inputs(resolve(cfg["paths"]["source_manifest"]))
    if len(rows) != cfg["expected_images"]:
        raise ValueError("Unexpected fixed pilot size")
    out = resolve(cfg["paths"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "tags").mkdir(exist_ok=args.resume)
    saved_config = out / "CONFIG.yaml"
    if saved_config.exists() and C.sha256_file(str(saved_config)) != cfg["_sha256"]:
        raise ValueError("Existing output has a different configuration")
    shutil.copy2(cfg["_path"], out / "CONFIG.yaml")
    tc = cfg["tags"]
    prompt = build_prompt(tc["max_tags"], tc["max_words"])
    (out / "TAG_SYSTEM.txt").write_text(SYSTEM, encoding="utf-8")
    (out / "TAG_USER.txt").write_text(prompt, encoding="utf-8")
    print("Fixed RGB-only manifest:", len(rows), "images. No GT or selection stage.", flush=True)
    check_rows = rows if args.limit is None else rows[:args.limit]
    for i, row in enumerate(check_rows, 1):
        if C.sha256_file(row["image_path"]) != row["image_sha256"]:
            raise ValueError("Image hash mismatch: " + row["image_id"])
        if i % 8 == 0:
            print("INPUT_CHECK %d/%d" % (i, len(check_rows)), flush=True)
    C.write_csv(str(out / "PILOT_MANIFEST.csv"), rows)
    total = len(rows)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:args.limit]
    model = None
    records, started = [], time.time()
    for i, row in enumerate(rows, 1):
        record_path = out / "tags" / (row["image_id"] + ".json")
        previous = existing_record(record_path, row, cfg["_sha256"], ("OK", "EMPTY")) if args.resume else None
        if previous is not None:
            records.append(previous)
            print("[%d/%d] TAG_REUSED %s" % (i, len(rows), row["image_id"]), flush=True)
            continue
        failed = existing_record(record_path, row, cfg["_sha256"]) if args.resume and args.repair_errors else None
        if failed is not None and failed.get('status') == 'ERROR':
            if failed.get('recovery_prompt_version'):
                normalized, rejected, recovery = parse_repair_tags(failed['raw_replies'][-1], tc)
                repaired = dict(failed, tags=normalized, rejected=rejected, error=None,
                                status='OK' if normalized else 'EMPTY', format_recovery=recovery,
                                recovered_saved_reply=True)
                write_record(record_path, repaired)
                records.append(repaired)
                print('[%d/%d] SAVED_REPLY_RECOVERED %s' % (i, len(rows), row['image_id']), flush=True)
                continue
            if model is None:
                model = QwenCaptioner(cfg['paths']['qwen_model'], max_tokens=tc['max_tokens'])
            write_record(out / 'tags_initial_errors' / (row['image_id'] + '.json'), failed)
            repaired = repair_record(model, row, failed, tc)
            write_record(record_path, repaired)
            records.append(repaired)
            print('[%d/%d] TAG_REPAIR %s %s' % (i, len(rows), row['image_id'], repaired['status']), flush=True)
            continue
        if model is None:
            model = QwenCaptioner(cfg["paths"]["qwen_model"], max_tokens=tc["max_tokens"])
        tags, rejected, replies, attempts, error = [], [], [], [], None
        for _ in range(1 + tc["retry_invalid"]):
            actual_prompt = prompt if error is None else prompt + "\nYour previous reply failed parsing: " + error + ". Return the complete tags JSON."
            raw = model.caption(row["image_path"], SYSTEM, actual_prompt)
            replies.append(raw)
            try:
                tags, rejected = parse_tags(raw, tc["max_tags"], tc["max_words"])
                error = None
            except (ValueError, TypeError) as exc:
                error = str(exc)
            attempts.append({"prompt": actual_prompt, "raw": raw, "error": error})
            if error is None:
                break
        status = "ERROR" if error else "OK" if tags else "EMPTY"
        rec = {"image_id": row["image_id"], "group": "TrainVal", "image_sha256": row["image_sha256"],
               "status": status, "tags": tags, "rejected": rejected, "raw_replies": replies,
               "attempts": attempts, "error": error, "prompt_version": tc["prompt_version"],
               "config_sha256": cfg["_sha256"], "gt_used": False, "main_target_selected": False}
        if status == 'ERROR' and args.repair_errors:
            write_record(out / 'tags_initial_errors' / (row['image_id'] + '.json'), rec)
            rec = repair_record(model, row, rec, tc)
        write_record(record_path, rec)
        records.append(rec)
        print("[%d/%d] %s %s %s" % (i, len(rows), row["image_id"], rec['status'],
              [(t["kind"], t["tag"]) for t in rec['tags']]), flush=True)
    C.write_json(str(out / "TAG_SUMMARY.json"), {
        "n_images": len(records), "expected_images": total, "complete": len(records) == total,
        "n_tags": sum(len(r["tags"]) for r in records),
        "status_counts": dict(Counter(r["status"] for r in records)),
        "repair_errors_enabled": args.repair_errors,
        "n_corrective_calls": sum(bool(r.get('recovery_prompt_version')) for r in records),
        "n_untyped_recoveries": sum(r.get('format_recovery') == 'untyped_list_unknown_roles_v1' for r in records),
        "tag_kind_counts": dict(Counter(t["kind"] for r in records for t in r["tags"])),
        "elapsed_s": round(time.time()-started, 1), "source_manifest_sha256": C.sha256_file(str(resolve(cfg["paths"]["source_manifest"]))),
        "config_sha256": cfg["_sha256"], "gt_used": False, "main_target_selected": False})
    C.write_csv(str(out / "TAG_TABLE.csv"), [{"image_id": r["image_id"], **t} for r in records for t in r["tags"]])
    if any(r["status"] == "ERROR" for r in records):
        print("TRAIN_PRIOR_TAGS_FINISHED_WITH_ERRORS", flush=True)
        return 2
    print("TRAIN_PRIOR_TAGS_DONE" if len(records) == total else "TRAIN_PRIOR_TAGS_PARTIAL", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
