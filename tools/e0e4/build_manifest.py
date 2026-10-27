"""tools/e0e4/build_manifest.py - freeze the P0 (Gate C) / P1 (Gate D/E) image lists.

WHAT THIS WRITES (all under ``--out``)
--------------------------------------
    INPUT_MANIFEST.csv   image_id, dataset, stem, group, width, height,
                         image_sha256, image_path, gt_path   (GT is listed, never read)
    manifest_p0.txt      30 ids for Gate C  (10 CAMO / 10 COD10K-CAM / 10 NonCAM)
    manifest_p1.txt      200 ids for Gate D/E (50 / 100 / 50), disjoint from P0
    SELECTION.json       seed + quotas + the exact ids + duplicate-hash report
    MANIFEST_SUMMARY.md  human-readable counts

WHY IT IS A SEPARATE STAGE
--------------------------
P0/P1 must be identical on every machine and must not be re-drawn after a bad
result.  Selection is therefore ``sorted(rows) -> random.Random("seed|group")
.shuffle -> first N``: no filesystem-order effect, no "replace it because SAM3
failed on it".  P1 is drawn from the complement of P0 with the same seed.

Usage (server, repo root)::

    python tools/e0e4/build_manifest.py --config configs/e0e4_v1.yaml
    python tools/e0e4/build_manifest.py --out outputs/E0_E4_ZEROTRAIN_V1 --no_hash
"""
import argparse
import os
import sys
from collections import Counter

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from tools.e0e4 import common as C  # noqa: E402


def parse_quota(text):
    """"CAMO:50,COD10K-CAM:100,COD10K-NonCAM:50" -> dict (or None)."""
    if not text:
        return None
    out = {}
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise C.StageError("quota must be GROUP:N, got %r" % part)
        group, n = part.split(":", 1)
        out[group.strip()] = int(n)
    return out


def _image_size_and_hash(path, want_hash=True):
    from PIL import Image

    with Image.open(path) as im:
        width, height = im.size
    digest = C.sha256_file(path) if want_hash else ""
    return width, height, digest


def run_stage(cfg, out_dir, seed=None, p0_per_group=None, p1_quota=None,
              want_hash=True, overwrite=False):
    seed = int(seed if seed is not None else cfg.get("seed", 20261020))
    p0_per_group = int(p0_per_group if p0_per_group is not None
                       else C.cfg_get(cfg, "manifest.p0_per_group", 10))
    p1_quota = p1_quota or dict(C.cfg_get(cfg, "manifest.p1", {}) or {})
    if not p1_quota:
        raise C.StageError("manifest.p1 is empty in the config")

    C.guard_config(cfg)
    data_root = C.resolve_path(C.cfg_get(cfg, "paths.data_root"))
    datasets = C.cfg_get(cfg, "paths.datasets", ["CAMO", "COD10K"])
    rows, problems = C.manifest_rows(data_root, datasets)
    if problems:
        raise C.StageError("dataset problems: %s" % "; ".join(problems))
    if not rows:
        raise C.StageError("no image/GT pair found under %s" % data_root)

    root = os.path.abspath(out_dir)
    C.ensure_dir(root)
    p0 = C.pick_stratified(rows, {g: p0_per_group for g in C.GROUPS}, seed)
    p0_ids = [r["image_id"] for r in p0]
    p1 = C.pick_stratified(rows, p1_quota, seed, exclude=p0_ids)
    p1_ids = [r["image_id"] for r in p1]
    overlap = set(p0_ids) & set(p1_ids)
    if overlap:
        raise C.StageError("P0/P1 overlap (%d ids), first: %s"
                           % (len(overlap), sorted(overlap)[0]))
    for name, ids in (("P0", p0_ids), ("P1", p1_ids)):
        if len(ids) != len(set(ids)):
            raise C.StageError("%s contains duplicate image ids" % name)

    # ---- INPUT_MANIFEST.csv (all test images; hash/size are provenance only)
    manifest, by_hash = [], {}
    for idx, row in enumerate(rows, 1):
        width, height, digest = _image_size_and_hash(
            C.rel_id_to_paths(row, cfg)[0], want_hash=want_hash)
        manifest.append({"image_id": row["image_id"], "dataset": row["dataset"],
                         "stem": row["stem"], "group": row["group"],
                         "width": width, "height": height,
                         "image_sha256": digest,
                         "image_path": row["image_path"],
                         "gt_path": row["gt_path"]})
        if digest:
            by_hash.setdefault(digest, []).append(row["image_id"])
        if idx % 250 == 0 or idx == len(rows):
            C.progress(idx, len(rows), "manifest size/hash")
    C.write_csv(os.path.join(root, C.MANIFEST_ALL_CSV), manifest)

    C.write_id_file(os.path.join(root, C.MANIFEST_P0), p0_ids,
                    header="P0 / Gate C | seed=%d | per-group=%d"
                           % (seed, p0_per_group))
    C.write_id_file(os.path.join(root, C.MANIFEST_P1), p1_ids,
                    header="P1 / Gate D+E | seed=%d | quota=%s | excludes P0"
                           % (seed, p1_quota))

    dup = {h: ids for h, ids in by_hash.items() if len(ids) > 1}
    selection = {
        "seed": seed,
        "algorithm": "sorted by image_id -> random.Random('seed|group').shuffle -> head N",
        "p0": {"per_group": {g: p0_per_group for g in C.GROUPS}, "ids": p0_ids},
        "p1": {"quota": p1_quota, "ids": p1_ids},
        "counts": {
            "all": dict(Counter(r["group"] for r in rows)),
            "p0": dict(Counter(r["group"] for r in p0)),
            "p1": dict(Counter(r["group"] for r in p1)),
        },
        "dataset_counts": dict(Counter(r["dataset"] for r in rows)),
        "duplicate_image_hashes": {"n_groups": len(dup),
                                   "n_images": sum(len(v) for v in dup.values())},
        "config": {"path": cfg.get("_path"), "sha256": cfg.get("_sha256"),
                   "version": cfg.get("_version")},
        "utc": C.utc_now(),
    }
    C.write_json(os.path.join(root, C.SELECTION_JSON), selection)

    counts = selection["counts"]
    lines = [
        "# Manifest summary (%s)" % cfg.get("_version"),
        "",
        "| split | " + " | ".join(C.GROUPS) + " | total |",
        "|---" * (len(C.GROUPS) + 2) + "|",
    ]
    for split in ("p0", "p1"):
        c = counts[split]
        lines.append("| %s | %s | %d |"
                     % (split.upper(),
                        " | ".join(str(c.get(g, 0)) for g in C.GROUPS),
                        sum(c.values())))
    lines += [
        "",
        "All test images: %d  (dataset counts: %s)"
        % (len(rows), selection["dataset_counts"]),
        "",
        "- seed: `%d`" % seed,
        "- P0 file: `%s`  |  P1 file: `%s`" % (C.MANIFEST_P0, C.MANIFEST_P1),
        "- duplicate image hashes: %d group(s) / %d image(s)"
        % (selection["duplicate_image_hashes"]["n_groups"],
           selection["duplicate_image_hashes"]["n_images"]),
        "- No image is ever replaced because of a downstream failure.",
    ]
    with open(os.path.join(root, "MANIFEST_SUMMARY.md"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")

    print(">>> manifest: %d images indexed; P0=%d, P1=%d -> %s"
          % (len(rows), len(p0_ids), len(p1_ids), root))
    return selection


def main(argv=None):
    ap = argparse.ArgumentParser(description="build P0/P1 image manifests")
    ap.add_argument("--config", default=os.path.join(C.ROOT, "configs", "e0e4_v1.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--p0_per_group", type=int, default=None)
    ap.add_argument("--p1", default=None,
                    help="override quota, e.g. CAMO:50,COD10K-CAM:100,COD10K-NonCAM:50")
    ap.add_argument("--no_hash", action="store_true",
                    help="skip image sha256 (dev only; not valid for Gate C+)")
    ap.add_argument("--overwrite", action="store_true")
    opt = ap.parse_args(argv)
    try:
        cfg = C.load_config(opt.config)
        out_dir = C.resolve_path(opt.out or C.cfg_get(cfg, "paths.out_dir"))
        existing = os.path.join(out_dir, C.SELECTION_JSON)
        if os.path.isfile(existing) and not opt.overwrite:
            raise C.StageError("selection already frozen: %s (pass --overwrite to redo)"
                               % existing)
        run_stage(cfg, out_dir, seed=opt.seed, p0_per_group=opt.p0_per_group,
                  p1_quota=parse_quota(opt.p1), want_hash=not opt.no_hash,
                  overwrite=opt.overwrite)
    except C.StageError as exc:
        print(">>> BLOCKED: %s" % exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
