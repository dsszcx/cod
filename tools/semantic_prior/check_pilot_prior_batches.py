"""Inspect every pilot package and a real batch per split/arm. No training."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import resolve
from dataloader_relation import RelationDataset, collate_relation_batch
from Network_PVTv2 import Network as RGBNetwork
from Network_Relation import Network as RelationNetwork


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/relation_train_prep_v1.yaml")
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(22)
    cfg = C.load_config(str(resolve(args.config)))
    root = resolve(cfg["paths"]["out_dir"])
    report_path = root / "PILOT_PRIOR_BATCH_CHECK.json"
    if report_path.exists():
        raise FileExistsError(str(report_path))
    # Random RGB weights here check only structural equivalence/finite forward.
    # Formal experiments will use identical ImageNet initialization instead.
    rgb_model = RGBNetwork(channel=64, imagenet_pretrained=False, use_ca=False).eval()
    region_model = RelationNetwork(channel=64, imagenet_pretrained=False, use_ca=False).eval()
    region_model.load_rgb_state_dict(rgb_model.state_dict())
    reports = {}
    for split in ("train", "val"):
        manifest = root / ("pilot_"+split+".csv")
        datasets = {arm: RelationDataset(manifest, split, arm, seed=cfg["seed"]) for arm in ("rgb", "direct", "relation")}
        region_dataset = datasets["relation"]
        counts, native_n, valid_n, first = [], 0, 0, []
        for index in range(len(region_dataset)):
            row = region_dataset.rows[index]
            item = region_dataset[index]
            sample = item["inputs"]
            n, v = len(sample["masks"]), int(sample["region_valid"].sum())
            metadata = C.read_json(str(Path(row["region_package"]).with_suffix(".json")))
            if metadata is None or len(metadata["nodes"]) != n:
                raise AssertionError("Per-package node count differs from loaded masks")
            native_n += n
            valid_n += v
            if not all(np.isfinite(sample[k]).all() for k in ("rgb", "masks", "role_hint", "relation_geom")):
                raise ValueError("Nonfinite real sample")
            if not torch.isfinite(item["gt"]).all() or item["gt"].min() < 0 or item["gt"].max() > 1:
                raise ValueError("Invalid GT")
            counts.append({"image_id": row["image_id"], "native_regions": n, "valid_after_transform": v})
            if index < 2:
                first.append(item)
            if (index+1) % 8 == 0:
                print("PACKAGE_LOAD %s %d/%d" % (split, index+1, len(region_dataset)), flush=True)
        reference_batch = collate_relation_batch(first)
        arms = {}
        for arm in ("rgb", "direct", "relation"):
            batch = reference_batch if arm == "relation" else collate_relation_batch([datasets[arm][i] for i in range(len(first))])
            if not torch.equal(batch["gt"], reference_batch["gt"]) or not torch.equal(batch["inputs"]["rgb"], reference_batch["inputs"]["rgb"]):
                raise AssertionError("Ablation data/augmentation mismatch")
            if "gt" in batch["inputs"]:
                raise AssertionError("GT leaked into inputs")
            with torch.inference_mode():
                if arm == "rgb":
                    outputs = rgb_model(batch["inputs"]["rgb"])
                else:
                    region_model.region_fusion.alpha_direct.fill_(.1)
                    region_model.region_fusion.alpha_relation.fill_(.1 if arm == "relation" else 0)
                    outputs = region_model(**batch["inputs"])
            b = len(first)
            if len(outputs) != 4 or any(t.shape != (b, 1, 352, 352) or not torch.isfinite(t).all() for t in outputs):
                raise AssertionError("Bad real-batch outputs")
            arms[arm] = {"input_shapes": {k: list(t.shape) for k, t in batch["inputs"].items()},
                         "gt_shape": list(batch["gt"].shape), "output_shapes": [list(t.shape) for t in outputs],
                         "same_rgb_gt_as_relation": True, "outputs_finite": True}
            print("BATCH_FORWARD", split, arm, "OK", flush=True)
        reports[split] = {"images_checked": len(region_dataset), "native_regions": native_n,
                          "valid_regions_after_transform": valid_n, "regions_lost_to_transform": native_n-valid_n,
                          "images_with_no_candidates": sum(c["native_regions"] == 0 for c in counts),
                          "counts": counts, "arms": arms}
    package_summary = C.read_json(str(root / "priors/PACKAGE_SUMMARY.json"))
    n_regions = sum(s["native_regions"] for s in reports.values())
    if package_summary is None and cfg.get("data_scope") != "cam_only":
        raise FileNotFoundError("Missing original package summary")
    n_images = sum(s["images_checked"] for s in reports.values())
    summary_covers_manifest = package_summary is not None and package_summary.get("n_images") == n_images
    if package_summary is not None and not summary_covers_manifest and cfg.get("data_scope") != "cam_only":
        raise AssertionError("Package summary does not cover this manifest")
    if summary_covers_manifest and n_regions != package_summary["n_regions"]:
        raise AssertionError("Region total does not match packed summary")
    report = {"status": "REAL_PRIOR_BATCH_CHECK_PASSED", "splits": reports,
              "all_pilot_packages_checked": True, "n_regions": n_regions,
              "region_count_reference":"per-package nodes metadata for every manifest row",
              "local_package_summary_covers_full_manifest":summary_covers_manifest,
              "data_scope":cfg.get("data_scope","historical_mixed"),
              "model_initialization": "random RGB weights shared with relation; forward interface check only",
              "alpha_test_value": .1, "inference_device": "cpu", "optimizer_steps": 0,
              "training_started": False, "gt_used_for_prior_generation": False,
              "qwen_called": False, "sam3_called": False, "performance_claim": "none"}
    C.write_json(str(report_path), report)
    print("PILOT_PRIOR_BATCH_CHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
