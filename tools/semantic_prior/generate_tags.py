"""Step 1: visible objects, parts and context tags from original RGB images."""
import argparse
from collections import Counter
from pathlib import Path
import shutil
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.e0e4 import common as C
from tools.e0e4.caption_qwen import QwenCaptioner
from tools.semantic_prior.common import choose_pilot, normalize_tags, resolve

SYSTEM_PROMPT = (
    "Describe visible semantic components of the supplied image. "
    "Include whole objects, identifiable object parts and surrounding context. "
    "Do not guess invisible objects. Do not select a main target yet. "
    "Return one JSON object only."
)


def user_prompt(max_tags, max_words):
    return (
        "Produce up to %d distinct visual tags for this image.\n"
        "Return JSON with a tags array. Each entry has:\n"
        '- "tag": a short English noun phrase, at most %d words.\n'
        '- "kind": exactly one of "object", "part", "context", "action".\n'
        "Use object for whole entities, part for visible components of an entity, "
        "context for surrounding surfaces, materials or environmental regions. "
        "Use action only if a visible activity is useful and cannot be expressed as a region.\n"
        "Try to cover distinct semantic components, rather than multiple synonyms "
        "for the same entity. Include visible parts and environmental context when available.\n"
        "Tags must describe actual image content. Do not invent a part merely to fill a category. "
        "Do not output coordinates, scores, object IDs, markdown, explanations or placeholder strings. "
        "If there is no identifiable content, return an empty tags array."
    ) % (max_tags, max_words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/semantic_prior_pilot_v1.yaml")
    args = ap.parse_args()
    cfg = C.load_config(str(resolve(args.config)))
    out = resolve(cfg["paths"]["out_dir"])
    tag_root = out / "tags"
    if tag_root.exists():
        raise FileExistsError("Tag output already exists: " + str(tag_root))
    out.mkdir(parents=True, exist_ok=True)
    tag_root.mkdir()
    selected = choose_pilot(C.read_csv(str(resolve(cfg["paths"]["source_manifest"]))),
                            cfg["sample"]["groups"], cfg["sample"]["per_group"])
    for row in selected:
        ds, stem = row["image_id"].split("/", 1)
        matches = [p for p in (resolve(cfg["paths"]["data_root"]) / ds / "Imgs").glob(stem + ".*")
                   if p.suffix.lower() in C.IMG_EXTS]
        if len(matches) != 1:
            raise ValueError("Image pairing problem: " + row["image_id"])
        row["image_path"] = str(matches[0])
        if C.sha256_file(str(matches[0])) != row["image_sha256"]:
            raise ValueError("Image hash mismatch: " + row["image_id"])
    C.write_csv(str(out / "PILOT_MANIFEST.csv"), selected)
    shutil.copy2(cfg["_path"], out / "CONFIG.yaml")
    tc = cfg["tags"]
    prompt = user_prompt(tc["max_tags"], tc["max_words"])
    (out / "TAG_SYSTEM.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    (out / "TAG_USER.txt").write_text(prompt, encoding="utf-8")
    print("PPT semantic-prior pilot:", len(selected), "images; original RGB only.", flush=True)
    captioner = QwenCaptioner(cfg["paths"]["qwen_model"], max_tokens=tc["max_tokens"])
    index, started = [], time.time()
    for idx, row in enumerate(selected, 1):
        tags, rejected, replies, error = [], [], [], None
        for attempt in range(1 + tc["retry_invalid"]):
            raw = captioner.caption(row["image_path"], SYSTEM_PROMPT, prompt)
            replies.append(raw)
            try:
                tags, rejected = normalize_tags(raw, tc["max_tags"], tc["max_words"])
                error = None
                if tags or not rejected:
                    break
            except ValueError as exc:
                error = str(exc)
        status = "ERROR" if error else "OK" if tags else "EMPTY"
        rec = {"image_id": row["image_id"], "group": row["group"],
               "image_sha256": row["image_sha256"], "status": status,
               "tags": tags, "rejected": rejected, "raw_replies": replies,
               "error": error, "prompt_version": tc["prompt_version"],
               "config_sha256": cfg["_sha256"]}
        C.write_json(str(tag_root / (row["image_id"] + ".json")), rec)
        index.append(rec)
        print("[%d/%d] %s %s %s" % (idx, len(selected), row["image_id"], status,
              [(t["kind"], t["tag"]) for t in tags]), flush=True)
    C.write_json(str(out / "TAG_SUMMARY.json"), {
        "n_images": len(index), "status_counts": dict(Counter(r["status"] for r in index)),
        "tag_kind_counts": dict(Counter(t["kind"] for r in index for t in r["tags"])),
        "n_tags": sum(len(r["tags"]) for r in index),
        "elapsed_s": round(time.time() - started, 1), "config_sha256": cfg["_sha256"],
    })
    C.write_csv(str(out / "TAG_TABLE.csv"), [
        {"image_id": r["image_id"], "group": r["group"], **t}
        for r in index for t in r["tags"]])
    if any(r["status"] == "ERROR" for r in index):
        return 2
    print("SEMANTIC_TAGS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
