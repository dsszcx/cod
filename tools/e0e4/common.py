"""tools/e0e4/common.py - shared plumbing for the E0-E4 inference-only round.

Design rules baked into this module
-----------------------------------
* **No GT is read here beyond *listing* GT file names.**  ``pair_images_with_gt``
  stats paths so a missing GT is detected up front, but it never opens a mask;
  the E0-E4 stages only ever see ``<dataset>/<stem>`` and the RGB image.
* **Determinism is data, not luck.**  P0/P1 selection is a pure function of
  (sorted rows, seed, per-group quota) so re-running Gate B produces the exact
  same 30 / 200 image ids on any machine.
* **Refuse, don't repair.**  Output directories that already contain files are
  rejected unless ``--overwrite``; a half-written stage would otherwise be
  scored as if it were complete (the failure mode ``eval_metrics.py`` fixed for
  prediction folders).
"""
import csv
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone

# repo root: tools/e0e4/common.py -> tools/e0e4 -> tools -> <root>
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
GT_EXTS = (".tif", ".tiff", ".png", ".jpg", ".bmp")

# Report strata.  Keep the names stable: eval_strict / make_review group by them.
GROUP_CAMO = "CAMO"
GROUP_COD_CAM = "COD10K-CAM"
GROUP_COD_NONCAM = "COD10K-NonCAM"
GROUPS = (GROUP_CAMO, GROUP_COD_CAM, GROUP_COD_NONCAM)

MANIFEST_ALL_CSV = "INPUT_MANIFEST.csv"
MANIFEST_P0 = "manifest_p0.txt"
MANIFEST_P1 = "manifest_p1.txt"
SELECTION_JSON = "SELECTION.json"


class StageError(RuntimeError):
    """A stage cannot proceed (missing input, dirty output dir, guard trip).

    ``main()`` wrappers translate it into exit code 2 so ``run_e0e4.py`` and the
    shell never mistake "stopped on purpose" for "finished".
    """

    exit_code = 2


def bootstrap_sys_path():
    """Make `python tools/e0e4/<stage>.py` work without pip-installing the repo."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    return ROOT


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def load_config(path):
    """Load the frozen YAML config (PyYAML is in requirements[-server].txt)."""
    import yaml

    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise StageError("config must be a mapping: %s" % path)
    cfg["_path"] = os.path.abspath(path)
    cfg["_sha256"] = sha256_file(path)
    cfg["_version"] = cfg.get("version", "unset")
    return cfg


def cfg_get(cfg, dotted, default=None):
    """``cfg_get(cfg, "roi.max_roi")`` with a default for missing leaves."""
    node = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def resolve_path(value, root=None):
    """Absolute path for a config/CLI value (relative paths are repo-relative)."""
    if value is None:
        return None
    value = os.path.expanduser(str(value))
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(root or ROOT, value))


def guard_config(cfg, keys=("guards",)):
    """Refuse to run when a config guard that must stay ``false`` is ``true``."""
    bad = []
    for block in keys:
        node = cfg.get(block) or {}
        if not isinstance(node, dict):
            continue
        for k, v in sorted(node.items()):
            if v is True:
                bad.append("%s.%s" % (block, k))
    if bad:
        raise StageError("config guard(s) enabled, refusing to run: %s"
                         % ", ".join(bad))


# ---------------------------------------------------------------------------
# tiny IO helpers
# ---------------------------------------------------------------------------
def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path, chunk=1 << 20):
    """sha256 of a file; ``None`` when absent (same contract as utils.sha256_file)."""
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def write_json(path, obj):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=False)
    return path


def read_json(path, default=None):
    if not os.path.isfile(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_jsonl(path, rows):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path):
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_csv(path, rows, columns=None):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    rows = list(rows)
    if columns is None:
        columns = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return path


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def guard_out_dir(out_dir, overwrite=False, allowed=()):
    """Refuse a non-empty output dir unless --overwrite (never mix two runs)."""
    out_dir = os.path.abspath(out_dir)
    if not os.path.isdir(out_dir):
        return out_dir
    entries = [e for e in os.listdir(out_dir) if e not in set(allowed)]
    if entries and not overwrite:
        raise StageError(
            "output dir is not empty: %s (%d entries, first: %s). "
            "Use a fresh --out or pass --overwrite."
            % (out_dir, len(entries), entries[0]))
    return out_dir


def guard_stage_dir(path, overwrite=False):
    """Refuse to write a stage's output twice (resume is an explicit --overwrite)."""
    if os.path.isdir(path) and os.listdir(path) and not overwrite:
        raise StageError("stage output already exists: %s (pass --overwrite)" % path)
    return path


# ---------------------------------------------------------------------------
# image identity / strata
# ---------------------------------------------------------------------------
def is_noncam_name(name):
    """COD10K 'NonCAM' test image, via dataloader_sam3.is_noncam (single source).

    The name rule is used ONLY for reporting strata; the pipeline never skips
    these images (``guards.allow_noncam_skip`` is false).
    """
    try:
        from dataloader_sam3 import is_noncam
        return bool(is_noncam(name))
    except Exception:  # torch missing / partial env: identical string rule
        return "NonCAM" in os.path.basename(str(name))


def group_of(dataset, stem):
    if dataset == "CAMO":
        return GROUP_CAMO
    if dataset == "COD10K":
        return GROUP_COD_NONCAM if is_noncam_name(stem) else GROUP_COD_CAM
    return dataset


def image_id(dataset, stem):
    return "%s/%s" % (dataset, stem)


def parse_image_id(ident):
    if "/" not in str(ident):
        raise StageError("image_id must look like 'CAMO/xxx', got %r" % (ident,))
    dataset, stem = str(ident).split("/", 1)
    return dataset, stem


def group_of_id(ident):
    dataset, stem = parse_image_id(ident)
    return group_of(dataset, stem)


def pair_images_with_gt(imgs_dir, gt_dir):
    """Pair by file STEM (never by sorted index); report unpaired files.

    Returns ``(pairs, missing_gt, orphan_gt)`` where ``pairs`` is
    ``[{"stem", "image", "gt"}, ...]`` sorted by stem.  GT files are only
    *listed*, never opened.
    """
    if not os.path.isdir(imgs_dir):
        raise StageError("image dir not found: %s" % imgs_dir)
    if not os.path.isdir(gt_dir):
        raise StageError("GT dir not found: %s" % gt_dir)
    images = sorted(f for f in os.listdir(imgs_dir) if f.lower().endswith(IMG_EXTS))
    gts = sorted(f for f in os.listdir(gt_dir) if f.lower().endswith(GT_EXTS))
    gt_by_stem = {os.path.splitext(g)[0]: g for g in gts}
    pairs, missing = [], []
    for f in images:
        stem = os.path.splitext(f)[0]
        gt = gt_by_stem.get(stem)
        if gt is None:
            missing.append(f)
        else:
            pairs.append({"stem": stem,
                          "image": os.path.join(imgs_dir, f),
                          "gt": os.path.join(gt_dir, gt)})
    img_stems = {os.path.splitext(f)[0] for f in images}
    orphan = sorted(s for s in gt_by_stem if s not in img_stems)
    return pairs, missing, orphan


def dataset_map_root(data_root, dataset):
    """`Dataset/TestDataset/<dataset>/{Imgs,GT}`."""
    base = os.path.join(data_root, dataset)
    return os.path.join(base, "Imgs"), os.path.join(base, "GT")


def manifest_rows(data_root, datasets):
    """All test images with stratum metadata (no GT content).

    Returns ``(rows, problems)``; ``problems`` lists datasets that could not be
    paired so preflight turns them into BLOCKED instead of producing a short
    table that looks fine.
    """
    rows, problems = [], []
    for dataset in datasets:
        imgs_dir, gt_dir = dataset_map_root(data_root, dataset)
        try:
            pairs, missing, orphan = pair_images_with_gt(imgs_dir, gt_dir)
        except StageError as exc:
            problems.append(str(exc))
            continue
        if missing:
            problems.append("%s: %d image(s) have no GT (first: %s)"
                            % (dataset, len(missing), missing[0]))
        if orphan:
            problems.append("%s: %d GT(s) have no image (first: %s)"
                            % (dataset, len(orphan), orphan[0]))
        for p in pairs:
            rows.append({
                "image_id": image_id(dataset, p["stem"]),
                "dataset": dataset,
                "stem": p["stem"],
                "group": group_of(dataset, p["stem"]),
                "image_path": os.path.relpath(p["image"], ROOT).replace(os.sep, "/"),
                "gt_path": os.path.relpath(p["gt"], ROOT).replace(os.sep, "/"),
            })
    rows.sort(key=lambda r: r["image_id"])
    return rows, problems


def pick_stratified(rows, per_group, seed, exclude=()):
    """Deterministic quota-based selection per stratum.

    Rows are sorted by ``image_id`` and only then shuffled with a seeded
    ``random.Random``, so the result does not depend on filesystem order or
    Python's hash randomisation, and re-running Gate B gives the same ids.
    """
    exclude = set(exclude or ())
    picked = []
    for group in GROUPS:
        n = int(per_group.get(group, 0) or 0)
        if n <= 0:
            continue
        pool = sorted((r for r in rows
                       if r["group"] == group and r["image_id"] not in exclude),
                      key=lambda r: r["image_id"])
        if len(pool) < n:
            raise StageError("stratum %s has %d candidate(s) but %d requested"
                             % (group, len(pool), n))
        random.Random("%s|%s" % (seed, group)).shuffle(pool)
        picked.extend(pool[:n])
    picked.sort(key=lambda r: r["image_id"])
    return picked


def read_id_file(path):
    """One image id per line (``#`` comments / blank lines ignored)."""
    if not os.path.isfile(path):
        raise StageError("image-id file not found: %s" % path)
    ids = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if line:
                ids.append(line)
    return ids


def write_id_file(path, ids, header=None):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        if header:
            fh.write("# %s\n" % header)
        for ident in ids:
            fh.write(ident + "\n")
    return path


def rows_for_ids(rows, ids):
    by_id = {r["image_id"]: r for r in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise StageError("%d image id(s) not in the manifest (first: %s)"
                         % (len(missing), missing[0]))
    return [by_id[i] for i in ids]


def rel_id_to_paths(row, cfg):
    """Absolute (image, gt) paths for a manifest row."""
    data_root = cfg_get(cfg, "paths.data_root", "Dataset/TestDataset")
    imgs_dir, gt_dir = dataset_map_root(resolve_path(data_root), row["dataset"])
    img = os.path.join(imgs_dir, os.path.basename(row["image_path"]))
    gt = os.path.join(gt_dir, os.path.basename(row["gt_path"]))
    return img, gt


def progress(idx, total, msg):
    print("[%d/%d] %s" % (idx, total, msg))
    sys.stdout.flush()
