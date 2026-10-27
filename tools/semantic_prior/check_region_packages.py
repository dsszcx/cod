"""CPU-only roundtrip/source/batch check of existing packed regions."""
import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import resolve
from lib.region_relations import load_region_package, transform_region_sample, collate_region_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/semantic_prior_pilot_v1.yaml")
    args = ap.parse_args()
    cfg = C.load_config(str(resolve(args.config)))
    source = resolve(cfg["paths"]["out_dir"])
    root = source / "region_packages_v1"
    samples, native_counts = [], []
    for row in C.read_csv(str(source / "PILOT_MANIFEST.csv")):
        path = root / row["image_id"]
        meta = C.read_json(str(path / "regions.json"))
        if C.sha256_file(str(path / "regions.npz")) != meta["package_sha256"] or C.sha256_file(row["image_path"]) != meta["image_sha256"]:
            raise ValueError("Package or RGB hash mismatch")
        if C.sha256_file(meta["candidate_source"]) != meta["candidate_source_sha256"]:
            raise ValueError("Source record changed")
        sample = load_region_package(path / "regions.npz")
        native_counts.append(len(sample["masks"]))
        for node in meta["nodes"]:
            mask_path = source / node["mask_source"]
            if C.sha256_file(str(mask_path)) != node["mask_sha256"]:
                raise ValueError("Source PNG changed")
            with Image.open(mask_path) as im:
                expected = np.asarray(im.convert("L")) > 0
            if not np.array_equal(sample["masks"][node["mask_index"]], expected):
                raise ValueError("Packed mask differs from source")
        with Image.open(row["image_path"]) as im:
            sample["rgb"] = np.asarray(im.convert("RGB"), np.float32).transpose(2, 0, 1) / 255.
        samples.append(transform_region_sample(sample, output_hw=(352, 352)))
    batch = collate_region_samples(samples, as_torch=False)
    if not all(np.isfinite(v).all() for v in batch.values()):
        raise ValueError("Nonfinite batch")
    for i, n in enumerate(native_counts):
        if batch["region_valid"][i, n:].any() or batch["relation_valid"][i, n:].any() or batch["relation_valid"][i, :, n:].any():
            raise ValueError("Padding entered valid pairs")
    report = {"native_counts": native_counts, "native_masks_equal_to_source": True,
              "batch_shapes": {k: list(v.shape) for k, v in batch.items()},
              "valid_regions_after_352_resize": int(batch["region_valid"].sum()),
              "regions_lost_to_resize": sum(native_counts) - int(batch["region_valid"].sum()),
              "padding_excluded": True, "all_finite": True, "models_called": False, "training": False}
    C.write_json(str(root / "PACKAGE_BATCH_CHECK.json"), report)
    print(report, flush=True)
    print("REGION_PACKAGE_CHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
