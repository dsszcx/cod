"""CAM-only dataset scope. Preserve existing split membership and source files."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import resolve


def category(row):
    """Dataset category only; never inspect GT or predictions to select rows."""
    ident = row["image_id"].replace("\\", "/")
    stem = ident.rsplit("/", 1)[-1].lower()
    filename = str(row.get("image_path", "")).replace("\\", "/").rsplit("/", 1)[-1].lower()
    if "noncam" in stem or "noncam" in filename:
        return "NonCAM"
    if stem.startswith("cod10k-cam-"):
        return "COD10K-CAM"
    if stem.startswith("camourflage_") or ident.split("/", 1)[0].upper() == "CAMO":
        return "CAMO"
    raise ValueError("Unknown dataset category; not silently dropped: " + ident)


def filter_rows(rows):
    kept, excluded = [], []
    for row in rows:
        if category(row) == "NonCAM":
            excluded.append(dict(row, reason="NonCAM_excluded_by_CAM_only_scope"))
        else:
            kept.append(dict(row))
    return kept, excluded


def assert_splits(train, val, test):
    for name, rows in (("train", train), ("val", val), ("test", test)):
        if not rows or len({r["image_id"] for r in rows}) != len(rows):
            raise ValueError("Empty split or duplicate ID: " + name)
        if any(category(r) == "NonCAM" for r in rows):
            raise ValueError("NonCAM remains: " + name)
    hashes = [{r["image_sha256"] for r in rows} for rows in (train, val, test)]
    if hashes[0] & hashes[1] or (hashes[0] | hashes[1]) & hashes[2]:
        raise ValueError("Exact image content crosses train/val/test")


def test_rows(root):
    rows = []
    for dataset in ("CAMO", "COD10K"):
        gt_files = defaultdict(list)
        for path in (root/dataset/"GT").iterdir():
            if path.suffix.lower() in C.GT_EXTS:
                gt_files[path.stem].append(path)
        images = sorted(p for p in (root/dataset/"Imgs").iterdir() if p.suffix.lower() in C.IMG_EXTS)
        stems = set()
        for path in images:
            if path.stem in stems or len(gt_files[path.stem]) != 1:
                raise ValueError("Ambiguous test image/GT: " + str(path))
            stems.add(path.stem)
            row = {"image_id": dataset+"/"+path.stem, "image_path": str(path),
                   "gt_path": str(gt_files[path.stem][0]), "split": "test", "dataset": dataset}
            # Excluded test images need not be decoded/hashed again.
            if category(row) != "NonCAM":
                with Image.open(path) as im:
                    row["width"], row["height"] = im.size
                row["image_sha256"] = C.sha256_file(str(path))
            rows.append(row)
        if set(gt_files) != stems:
            raise ValueError("Unpaired test GT: " + dataset)
    return rows


def make_view(root, subsets):
    """Symlinks only; never move, delete, or overwrite source RGB/GT."""
    for name, rows in subsets.items():
        for row in rows:
            suffix = row["dataset"] if name == "TestDataset" else ""
            folder = root/name/suffix
            for key, child in (("image_path", "Imgs"), ("gt_path", "GT")):
                source = Path(row[key])
                destination = folder/child/source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not source.is_file():
                    raise FileNotFoundError(str(source))
                destination.symlink_to(source)


def verify_view(root, subsets):
    expected = set()
    for name, rows in subsets.items():
        for row in rows:
            folder = root/name/(row["dataset"] if name == "TestDataset" else "")
            for key, child in (("image_path", "Imgs"), ("gt_path", "GT")):
                link = folder/child/Path(row[key]).name
                if not link.is_symlink() or not link.is_file() or link.resolve() != Path(row[key]).resolve():
                    raise ValueError("Wrong or broken data-view link: " + str(link))
                expected.add(link)
    actual = {p for p in root.rglob("*") if p.is_symlink() or p.is_file()}
    if actual != expected:
        raise ValueError("Extra or missing files in CAM-only data view")
    if any("noncam" in p.name.lower() for p in actual):
        raise ValueError("NonCAM remains in data view")
    return len(expected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/relation_cam_only_20261026.yaml")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    cfg = C.load_config(str(resolve(args.config)))
    source = resolve(cfg["paths"]["source_manifests"])
    out = resolve(cfg["paths"]["out_dir"])
    if args.verify_only:
        rows = {s:C.read_csv(str(out/(s+".csv"))) for s in ("train","val","test")}
        assert_splits(rows["train"],rows["val"],rows["test"])
        n_links = verify_view(out/"dataset_view", {"TrainDataset":rows["train"], "ValDataset":rows["val"], "TestDataset":rows["test"]})
        print("CAM_ONLY_VIEW_VERIFIED links=%d" % n_links, flush=True)
        return
    old, kept, excluded = {}, {}, []
    for split in ("train", "val", "pilot_train", "pilot_val"):
        old[split] = C.read_csv(str(source/(split+".csv")))
        kept[split], dropped = filter_rows(old[split])
        excluded.extend(dict(r, source_manifest=split+".csv") for r in dropped)
    old["test"] = test_rows(resolve(cfg["paths"]["official_test"]))
    kept["test"], dropped = filter_rows(old["test"])
    excluded.extend(dict(r, source_manifest="test.csv") for r in dropped)
    assert_splits(kept["train"], kept["val"], kept["test"])
    for split in ("train", "val"):
        full = {r["image_id"]: r for r in kept[split]}
        if not kept["pilot_"+split] or any(full.get(r["image_id"]) != r for r in kept["pilot_"+split]):
            raise ValueError("Pilot no longer a subset of original split")
    out.mkdir(parents=True, exist_ok=False)
    for split, rows in kept.items():
        C.write_csv(str(out/(split+".csv")), rows)
    C.write_csv(str(out/"excluded_noncam.csv"), excluded)
    generation_keys = ("image_id", "image_path", "image_sha256", "width", "height")
    C.write_csv(str(out/"PRIOR_INPUT_PILOT.csv"), [{k:r[k] for k in generation_keys}
                for r in kept["pilot_train"]+kept["pilot_val"]])
    C.write_id_file(str(out/"test_ids.txt"), [r["image_id"] for r in kept["test"]], header="CAM-only; no inference or evaluation performed")
    if cfg.get("create_symlink_view", False):
        subsets = {"TrainDataset":kept["train"], "ValDataset":kept["val"], "TestDataset":kept["test"]}
        make_view(out/"dataset_view", subsets)
        verify_view(out/"dataset_view", subsets)
    report = {"scope":"CAM_only", "utc":C.utc_now(), "split_membership_preserved":True,
              "selection_uses_gt_pixels":False, "selection_uses_predictions":False,
              "source_files_deleted":False, "training_started":False, "test_evaluated":False,
              "counts":{s:{"before":len(old[s]), "after":len(rows), "excluded":len(old[s])-len(rows),
                            "categories":dict(Counter(category(r) for r in rows))} for s,rows in kept.items()},
              "pilot_prior_files_present":{s:sum(Path(r["region_package"]).is_file() for r in kept["pilot_"+s]) for s in ("train","val")},
              "source_manifest_sha256":{s+".csv":C.sha256_file(str(source/(s+".csv"))) for s in old if s != "test"},
              "exact_hash_overlap_train_val_test":0,
              "duplicate_rule":"exact bytes only; near duplicates not checked",
              "test_role":"index only; previous development use is not undone by filtering",
              "new_metrics_comparable_with_old_mixed_metrics":False,
              "config_sha256":cfg["_sha256"]}
    C.write_json(str(out/"SCOPE_REPORT.json"), report)
    print(report, flush=True)
    print("CAM_ONLY_DATA_PREPARATION_DONE", flush=True)


if __name__ == "__main__":
    main()
