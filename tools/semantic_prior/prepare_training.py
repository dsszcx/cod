"""CPU-only paired manifests, exact-content split and pilot prior coverage."""
import argparse
from collections import defaultdict, Counter
import hashlib
import math
from pathlib import Path
import sys

from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import resolve


def split_rows(rows, val_fraction=.1, seed=20261022):
    if not 0 < val_fraction < 1:
        raise ValueError("Invalid validation fraction")
    groups = defaultdict(list)
    for row in rows:
        groups[row["image_sha256"]].append(dict(row))
    if len(groups) < 2:
        raise ValueError("Need two distinct content groups")
    order = sorted(groups, key=lambda h: hashlib.sha256((str(seed)+":"+h).encode()).hexdigest())
    nv = min(len(order)-1, max(1, math.ceil(len(order)*val_fraction)))
    train, val = [], []
    for i, digest in enumerate(order):
        split = "val" if i < nv else "train"
        for row in sorted(groups[digest], key=lambda r: r["image_id"]):
            row["split"] = split
            (val if split == "val" else train).append(row)
    return train, val


def pilot_rows(rows, quota):
    selected, seen = [], set()
    for row in rows:
        if len(selected) >= quota and row["image_sha256"] not in seen:
            break
        selected.append(row)
        seen.add(row["image_sha256"])
    return selected  # keep a duplicate group together even if it exceeds quota


def scan_pool(data_root, test_root, output_root):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    test_hashes = {C.sha256_file(str(p)) for p in sorted(test_root.glob("*/Imgs/*")) if p.suffix.lower() in exts}
    if not test_hashes:
        raise ValueError("No official test images found")
    gt_files = defaultdict(list)
    for p in (data_root / "GT").iterdir():
        if p.suffix.lower() in exts | {".tif", ".tiff"}:
            gt_files[p.stem].append(p)
    rows, excluded, stems = [], [], set()
    images = sorted(p for p in (data_root / "Imgs").iterdir() if p.suffix.lower() in exts)
    for p in images:
        if p.stem in stems or len(gt_files[p.stem]) != 1:
            raise ValueError("Image/GT basename ambiguity: " + p.name)
        stems.add(p.stem)
        gt = gt_files[p.stem][0]
        with Image.open(p) as im, Image.open(gt) as target:
            w, h = im.size
            if im.size != target.size:
                raise ValueError("Image/GT size mismatch")
        digest = C.sha256_file(str(p))
        row = {"image_id": "TrainVal/"+p.stem, "image_path": str(p), "image_sha256": digest,
               "gt_path": str(gt), "width": w, "height": h,
               "region_package": str(output_root / "priors" / "TrainVal" / p.stem / "regions.npz")}
        if digest in test_hashes:
            excluded.append(dict(row, reason="exact_image_content_in_official_test"))
        else:
            rows.append(row)
    if set(gt_files) != stems:
        raise ValueError("Unpaired GT files in training pool")
    return rows, excluded, len(images), len(test_hashes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/relation_train_prep_v1.yaml")
    args = ap.parse_args()
    cfg = C.load_config(str(resolve(args.config)))
    out = resolve(cfg["paths"]["out_dir"])
    out.mkdir(parents=True, exist_ok=False)
    rows, excluded, n_pool, n_test_hash = scan_pool(resolve(cfg["paths"]["train_pool"]), resolve(cfg["paths"]["official_test"]), out)
    train, val = split_rows(rows, cfg["split"]["val_fraction"], cfg["seed"])
    small_train, small_val = pilot_rows(train, cfg["pilot"]["train_images"]), pilot_rows(val, cfg["pilot"]["val_images"])
    for name, subset in (("train", train), ("val", val), ("pilot_train", small_train), ("pilot_val", small_val)):
        C.write_csv(str(out / (name+".csv")), subset)
    C.write_csv(str(out / "excluded_test_overlap.csv"), excluded)
    # Separate model-generation input: no GT path, split/class or NonCAM flag.
    C.write_csv(str(out / "PRIOR_INPUT_PILOT.csv"), [
        {k: r[k] for k in ("image_id", "image_path", "image_sha256", "width", "height")}
        for r in small_train + small_val])
    coverage = {name: sum(Path(r["region_package"]).is_file() for r in subset)
                for name, subset in (("pilot_train", small_train), ("pilot_val", small_val))}
    report = {"n_pool": n_pool, "n_exact_test_overlap_excluded": len(excluded), "n_unique_official_test_hashes": n_test_hash,
              "n_train": len(train), "n_val": len(val), "n_pilot_train": len(small_train), "n_pilot_val": len(small_val),
              "train_val_exact_hash_overlap": len({r["image_sha256"] for r in train} & {r["image_sha256"] for r in val}),
              "split_seed": cfg["seed"], "pilot_prior_files_present": coverage,
              "prior_generation_status": "NOT_STARTED", "training_started": False,
              "duplicate_rule": "exact file bytes grouped; near-duplicates/re-encodings not detected",
              "gt_use": "pairing and dimensions only; GT is not sent to prior generator",
              "initialization_plan": "same ImageNet backbone and random decoder seed across arms, no historical COD checkpoint",
              "config_sha256": cfg["_sha256"]}
    C.write_json(str(out / "PREPARATION_REPORT.json"), report)
    print(report, flush=True)
    print("RELATION_TRAIN_PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
