"""tools/e0e4/caption_qwen.py - E2/E3: Qwen2.5-VL-3B captioning with a fixed schema.

WHAT IS (AND IS NOT) SENT TO THE MODEL
--------------------------------------
* E2: the **original RGB image** only.
* E3: the 3-panel composite from ``roi_from_mask.build_composite`` (original |
  M0 overlay with numbered boxes | numbered ROI crops).
* Never: the file name, the dataset name, ``llm_annotator.extract_context``'s
  "this is likely a {category}" hint, or any GT mask.  ``guards.allow_filename_context``
  is false and ``preflight``/``test_e0e4.py`` check the source for it.

The system prompt and the JSON schema are identical for E2 and E3; only one
pre-registered sentence about the panel layout differs (otherwise a 3B model
cannot know what "region 2" refers to).

Parsing is deliberately strict and *never* repairs text into a prompt: a reply
that is not the requested JSON is recorded as PARSE_ERROR/SCHEMA_ERROR and the
image is counted, not silently dropped.  One retry is allowed on a parse error
(same image, same prompt), so ``calls <= 1 + qwen.retry_on_parse_error``.

Usage::

    python tools/e0e4/caption_qwen.py --config configs/e0e4_v1.yaml \
        --ids outputs/E0_E4_ZEROTRAIN_V1/manifest_p0.txt --branch e2
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from tools.e0e4 import common as C  # noqa: E402

PROMPT_VERSION = "e0e4_prompt_v1"

SYSTEM_PROMPT = (
    "You are a visual assistant for camouflaged object detection. "
    "The highlighted region(s) come from a possibly WRONG automatic prediction, "
    "so absence of an object is a valid answer. "
    "Describe only what is clearly visible; never guess from color or outline alone. "
    "Reply with one JSON object and nothing else."
)

_SCHEMA = (
    '{"objects":[{"name":"frog","sam_text":"a mottled brown frog","region_id":1}],'
    '"uncertain":false,"no_visible_object":false}'
)

_RULES = (
    "Rules:\n"
    "- 1 to 3 objects, most confident first. Use an empty list if nothing is visible.\n"
    "- \"name\": English category, 1-3 words.\n"
    "- \"sam_text\": English noun phrase, 1-6 words, naming only that object; it is "
    "passed to a text-prompted segmenter.\n"
    "- \"region_id\": the number printed next to the region the object occupies, "
    "or null when no number applies.\n"
    "- If nothing is clearly visible return: "
    '{"objects":[],"uncertain":true,"no_visible_object":true}\n'
    "- Never output coordinates, boxes, timestamps, markdown or explanations."
)

_BRANCH_TAIL = {
    "e2": "No region numbers are available; set \"region_id\" to null.",
    "e3": ("The image has three panels: left = original photo, middle = highlighted "
           "predicted regions with numbered boxes, right = numbered region crops. "
           "Use the printed number of a region as \"region_id\"."),
}

STATUSES = ("OK", "ABSTAIN", "EMPTY", "PARSE_ERROR", "SCHEMA_ERROR")


def build_prompts(branch):
    """(system, user) -- same schema for E2/E3, one fixed sentence differs."""
    if branch not in _BRANCH_TAIL:
        raise C.StageError("branch must be e2 or e3, got %r" % (branch,))
    user = ("Return exactly one JSON object in this schema:\n" + _SCHEMA + "\n\n"
            + _RULES + "\n\n" + _BRANCH_TAIL[branch])
    return SYSTEM_PROMPT, user


# ---------------------------------------------------------------------------
# reply parsing (pure functions -> unit-testable without the model)
# ---------------------------------------------------------------------------
def _valid_sam_text(text, max_words, max_chars):
    if not text or not text.isascii():
        return False, "not_ascii"
    if len(text) > int(max_chars):
        return False, "text_too_long"
    if re.search(r"[\[\]\(\)\{\}\d]", text):
        return False, "contains_digits_or_brackets"
    words = text.split()
    if not (1 <= len(words) <= int(max_words)):
        return False, "word_count_%d" % len(words)
    if not re.search(r"[A-Za-z]", text):
        return False, "no_letters"
    return True, ""


def _loads_json(text):
    """Whole-string JSON first, then the outermost {...} block; None on failure."""
    text = re.sub(r"```(?:json)?", "", str(text or "")).replace("```", "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except (ValueError, TypeError):
            return None
    return None


def parse_response(text, allowed_region_ids=(), max_words=6, max_chars=80,
                   max_objects=3):
    """Strict reply parser -> {status, objects, reasons, raw_object_count}.

    Nothing here can turn prose into a prompt: a missing/extra field is recorded
    in ``reasons`` and the object is dropped.  ``objects`` is always a list.
    """
    out = {"status": "PARSE_ERROR", "objects": [], "reasons": [],
           "raw_object_count": 0}
    data = _loads_json(text)
    if data is None:
        out["reasons"].append("json_decode_failed")
        return out
    if not isinstance(data, dict):
        out["status"] = "SCHEMA_ERROR"
        out["reasons"].append("top_level_not_object")
        return out

    abstain = bool(data.get("no_visible_object")) or bool(data.get("uncertain"))
    objects = data.get("objects")
    if objects is None:
        objects = []
        out["reasons"].append("objects_key_missing")
    if not isinstance(objects, list):
        out["status"] = "SCHEMA_ERROR"
        out["reasons"].append("objects_not_list")
        return out
    out["raw_object_count"] = len(objects)
    if not objects:
        out["status"] = "ABSTAIN" if abstain else "EMPTY"
        return out

    allowed = {int(v) for v in (allowed_region_ids or ())}
    seen = set()
    for raw in objects:
        if not isinstance(raw, dict):
            out["reasons"].append("object_not_dict")
            continue
        name = str(raw.get("name") or "").strip()
        sam_text = str(raw.get("sam_text") or "").strip()
        if not name:
            out["reasons"].append("empty_name")
            continue
        if not sam_text:
            sam_text = name
            out["reasons"].append("sam_text_from_name")
        ok, why = _valid_sam_text(sam_text, max_words, max_chars)
        if not ok:
            out["reasons"].append("reject_text:%s" % why)
            continue
        if not name.isascii():
            out["reasons"].append("reject_name:not_ascii")
            continue
        key = name.lower()
        if key in seen:
            out["reasons"].append("duplicate_name")
            continue
        region = raw.get("region_id")
        try:
            region = int(region) if region is not None else None
        except (TypeError, ValueError):
            region = None
            out["reasons"].append("region_id_not_int")
        if region is not None and region not in allowed:
            out["reasons"].append("region_id_unknown")
            region = None
        if len(out["objects"]) >= int(max_objects):
            out["reasons"].append("extra_object_dropped")
            break
        seen.add(key)
        out["objects"].append({"name": name, "sam_text": sam_text,
                               "region_id": region})

    if out["objects"]:
        out["status"] = "OK"
    else:
        out["status"] = "ABSTAIN" if abstain else "SCHEMA_ERROR"
    return out


def cache_key(model_path, system_prompt, user_prompt, image_sha256):
    """Stable cache key: model + exact prompts + image bytes (no filename)."""
    h = hashlib.sha256()
    for part in (model_path or "", system_prompt, user_prompt, image_sha256 or ""):
        h.update(str(part).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# model wrapper (lazy: importing this module must not need transformers)
# ---------------------------------------------------------------------------
class QwenCaptioner:
    """Thin wrapper over ``llm_annotator.QwenVLAnnotator._call_model``.

    ``_call_model`` takes an image path plus two prompts and nothing else, so
    the filename / dataset-name context used by ``generate()`` and
    ``llm_annotator.extract_context`` is structurally unreachable from here.
    """

    def __init__(self, model_path, max_tokens=192):
        try:
            from llm_annotator import QwenVLAnnotator, _LLM_AVAILABLE
        except Exception as exc:  # pragma: no cover - env dependent
            raise C.StageError("cannot import llm_annotator: %r" % (exc,))
        if not _LLM_AVAILABLE:
            raise C.StageError(
                "Qwen deps missing: pip install transformers accelerate qwen-vl-utils "
                "(and provide local weights via paths.qwen_model)")
        self.model_path = model_path
        self.max_tokens = int(max_tokens)
        self.ann = QwenVLAnnotator(model_path=model_path,
                                   max_tokens=self.max_tokens, temperature=0.0)

    def caption(self, image_path, system_prompt, user_prompt):
        return self.ann._call_model(image_path, system_prompt, user_prompt,
                                    max_tokens=self.max_tokens)


def _image_for_branch(row, cfg, branch, root):
    """E2 -> original RGB; E3 -> the ROI-stage composite (same original pixels)."""
    if branch == "e2":
        return C.rel_id_to_paths(row, cfg)[0]
    roi_json = os.path.join(root, "rois", row["dataset"], row["stem"] + ".json")
    rec = C.read_json(roi_json)
    if not rec:
        raise C.StageError("E3 needs the ROI stage first (missing %s)" % roi_json)
    comp = C.resolve_path(rec["transform"]["composite_path"])
    if not os.path.isfile(comp):
        raise C.StageError("composite image missing: %s" % comp)
    return comp


def run_stage(cfg, out_dir, ids, branch, limit=None, overwrite=False,
              captioner=None):
    """One caption call per image (plus at most one retry on PARSE_ERROR)."""
    C.guard_config(cfg)
    qcfg = cfg.get("qwen") or {}
    model_path = C.resolve_path(C.cfg_get(cfg, "paths.qwen_model"))
    system_prompt, user_prompt = build_prompts(branch)
    max_calls = 1 + int(qcfg.get("retry_on_parse_error", 1))
    max_words = int(qcfg.get("max_text_words", 6))
    max_chars = int(qcfg.get("max_text_chars", 80))

    data_root = C.resolve_path(C.cfg_get(cfg, "paths.data_root"))
    datasets = C.cfg_get(cfg, "paths.datasets", ["CAMO", "COD10K"])
    rows, problems = C.manifest_rows(data_root, datasets)
    if problems:
        raise C.StageError("dataset problems: %s" % "; ".join(problems))
    selected = C.rows_for_ids(rows, ids)
    if limit:
        selected = selected[:int(limit)]

    root = os.path.abspath(out_dir)
    cap_root = os.path.join(root, "captions", branch)
    prompt_dir = os.path.join(root, "prompts")
    cache_dir = os.path.join(prompt_dir, "cache")
    C.guard_stage_dir(cap_root, overwrite=overwrite)
    for d in (cap_root, prompt_dir, cache_dir):
        C.ensure_dir(d)
    with open(os.path.join(prompt_dir, "%s_system.txt" % branch), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write(system_prompt + "\n")
    with open(os.path.join(prompt_dir, "%s_user.txt" % branch), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write(user_prompt + "\n")

    if captioner is None:
        captioner = QwenCaptioner(
            model_path, max_tokens=int(qcfg.get("max_tokens", 192)))
    print(">>> [%s] model=%s max_calls=%d prompt=%s"
          % (branch.upper(), model_path, max_calls,
             qcfg.get("prompt_version", PROMPT_VERSION)))

    records, t_start = [], time.time()
    for idx, row in enumerate(selected, 1):
        dataset, stem = row["dataset"], row["stem"]
        image_path = _image_for_branch(row, cfg, branch, root)
        roi_rec = C.read_json(os.path.join(root, "rois", dataset, stem + ".json")) or {}
        allowed = [r["roi_id"] for r in roi_rec.get("rois", [])]
        key = cache_key(model_path, system_prompt, user_prompt,
                        C.sha256_file(image_path))
        cache_path = os.path.join(cache_dir, key + ".json")

        cached = C.read_json(cache_path)
        calls, raw, parsed, elapsed = 0, "", None, 0.0
        if cached and cached.get("parsed") is not None:
            raw, parsed = cached.get("raw_output", ""), cached["parsed"]
        else:
            t0 = time.time()
            while calls < max_calls:
                calls += 1
                raw = captioner.caption(image_path, system_prompt, user_prompt)
                parsed = parse_response(raw, allowed, max_words=max_words,
                                        max_chars=max_chars)
                if parsed["status"] != "PARSE_ERROR":
                    break
            elapsed = (time.time() - t0) * 1000.0
            C.write_json(cache_path, {"raw_output": raw, "parsed": parsed,
                                      "calls": calls})

        rec = {"image_id": row["image_id"], "dataset": dataset, "group": row["group"],
               "branch": branch, "status": parsed["status"],
               "objects": parsed["objects"], "reasons": parsed["reasons"],
               "raw_output": raw, "calls": calls,
               "image_used": os.path.relpath(image_path, C.ROOT).replace(os.sep, "/"),
               "prompt_version": qcfg.get("prompt_version", PROMPT_VERSION),
               "model": model_path, "region_ids_allowed": allowed,
               "elapsed_ms": int(round(elapsed)),
               "cached": bool(cached and cached.get("parsed") is not None)}
        records.append(rec)
        C.write_json(os.path.join(cap_root, dataset, stem + ".json"), rec)
        C.progress(idx, len(selected), "%s -> %s (%d obj, %d call)"
                   % (row["image_id"], rec["status"], len(rec["objects"]), calls))

    C.write_jsonl(os.path.join(root, "captions", "%s.jsonl" % branch), records)
    counts = {}
    for rec in records:
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    C.write_json(os.path.join(root, "captions", "QWEN_%s_SUMMARY.json" % branch), {
        "branch": branch, "utc": C.utc_now(), "n_images": len(records),
        "status_counts": counts,
        "n_objects": sum(len(r["objects"]) for r in records),
        "n_retried": sum(1 for r in records if r["calls"] > 1),
        "model": model_path,
        "prompt_version": qcfg.get("prompt_version", PROMPT_VERSION),
        "elapsed_s": round(time.time() - t_start, 1),
        "config": {"version": cfg.get("_version"), "sha256": cfg.get("_sha256")},
    })
    print(">>> %s done: %d image(s) %s" % (branch.upper(), len(records), counts))
    return records


def write_text_table(out_dir):
    """Side-by-side E2/E3 text table for review (TEXT_TABLE.csv)."""
    root = os.path.abspath(out_dir)
    e2 = {r["image_id"]: r for r in
          C.read_jsonl(os.path.join(root, "captions", "e2.jsonl"))}
    e3 = {r["image_id"]: r for r in
          C.read_jsonl(os.path.join(root, "captions", "e3.jsonl"))}
    rows = []
    for ident in sorted(set(e2) | set(e3)):
        a, b = e2.get(ident, {}), e3.get(ident, {})
        rows.append({
            "image_id": ident,
            "group": a.get("group") or b.get("group", ""),
            "e2_status": a.get("status", ""),
            "e2_texts": " | ".join(o["sam_text"] for o in a.get("objects", [])),
            "e3_status": b.get("status", ""),
            "e3_texts": " | ".join(o["sam_text"] for o in b.get("objects", [])),
            "e3_region_ids": "|".join(str(o["region_id"])
                                      for o in b.get("objects", [])),
        })
    return C.write_csv(os.path.join(root, "TEXT_TABLE.csv"), rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="E2/E3: Qwen captioning")
    ap.add_argument("--config", default=os.path.join(C.ROOT, "configs", "e0e4_v1.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--branch", required=True, choices=["e2", "e3"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    opt = ap.parse_args(argv)
    try:
        cfg = C.load_config(opt.config)
        out_dir = C.resolve_path(opt.out or C.cfg_get(cfg, "paths.out_dir"))
        run_stage(cfg, out_dir, C.read_id_file(opt.ids), opt.branch,
                  limit=opt.limit, overwrite=opt.overwrite)
        write_text_table(out_dir)
    except C.StageError as exc:
        print(">>> BLOCKED: %s" % exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
