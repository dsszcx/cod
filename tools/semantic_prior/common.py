"""Small shared helpers for the semantic-component pilot."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
KINDS = ("object", "part", "context", "action")


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def extract_json(raw):
    """Read a JSON container, allowing leading markdown or explanation."""
    decoder = json.JSONDecoder()
    for idx, char in enumerate(raw):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(raw[idx:])
        except ValueError:
            continue
        if isinstance(value, (dict, list)):
            return value
    raise ValueError("No JSON object in model reply")


def normalize_tags(raw, max_tags=8, max_words=6):
    obj = extract_json(raw)
    if isinstance(obj, list):
        obj = {"tags": obj}
    if not isinstance(obj.get("tags"), list):
        raise ValueError("tags must be a list")
    tags, rejected, seen = [], [], set()
    for entry in obj["tags"]:
        if not isinstance(entry, dict):
            rejected.append({"entry": entry, "reason": "not an object"})
            continue
        tag = entry.get("tag")
        kind = entry.get("kind")
        if not isinstance(tag, str) or kind not in KINDS:
            rejected.append({"entry": entry, "reason": "invalid tag or kind"})
            continue
        tag = " ".join(tag.strip().split()).lower()
        if (not tag or len(tag.split()) > max_words or len(tag) > 80
                or not tag.isascii() or not re.search(r"[a-z]", tag)
                or "<" in tag or ">" in tag):
            rejected.append({"entry": entry, "reason": "invalid phrase or placeholder"})
            continue
        if tag in seen:
            rejected.append({"entry": entry, "reason": "duplicate tag"})
            continue
        seen.add(tag)
        if len(tags) >= max_tags:
            rejected.append({"entry": entry, "reason": "tag count limit"})
            continue
        tags.append({"tag_id": "t%02d" % (len(tags) + 1), "tag": tag,
                     "kind": kind, "segmentable": kind != "action"})
    return tags, rejected


def choose_pilot(rows, groups, per_group):
    selected = []
    for group in groups:
        pool = sorted((r for r in rows if r["group"] == group), key=lambda r: r["image_id"])
        if len(pool) < per_group:
            raise ValueError("Not enough samples for " + group)
        selected.extend(pool[:per_group])
    return selected
