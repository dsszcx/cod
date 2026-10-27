"""CPU real-package forward acceptance. No optimizer, GT or weight export."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.e0e4.infer_e0 import load_model, load_image_tensor
from tools.semantic_prior.common import resolve
from lib.region_relations import load_region_package, transform_region_sample, collate_region_samples
from Network_Relation import Network


def max_diff(a, b):
    return max(float((x-y).abs().max()) for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/semantic_prior_pilot_v1.yaml")
    ap.add_argument("--checkpoint", default="model_pth/OUR/Net_epoch_40.pth")
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(22)
    cfg = C.load_config(str(resolve(args.config)))
    source = resolve(cfg["paths"]["out_dir"])
    output = source / "region_network_check_v1"
    output.mkdir(exist_ok=False)
    checkpoint = str(resolve(args.checkpoint))
    ckpt_hash = C.sha256_file(checkpoint)
    baseline = load_model(checkpoint, torch.device("cpu"), use_ca=False)
    model = Network(channel=64, imagenet_pretrained=False, use_ca=False).eval()
    n_keys = model.load_rgb_state_dict(baseline.state_dict())
    results, started = [], time.time()
    for row in C.read_csv(str(source / "PILOT_MANIFEST.csv")):
        path = source / "region_packages_v1" / row["image_id"]
        meta = C.read_json(str(path / "regions.json"))
        if C.sha256_file(row["image_path"]) != meta["image_sha256"] or C.sha256_file(str(path / "regions.npz")) != meta["package_sha256"]:
            raise ValueError("Image/package changed")
        sample = load_region_package(path / "regions.npz")
        with Image.open(row["image_path"]) as im:
            sample["rgb"] = np.asarray(im.convert("RGB"), np.float32).transpose(2, 0, 1) / 255.
        native_n = len(sample["masks"])
        sample = transform_region_sample(sample, output_hw=(352, 352))
        # Use the baseline's exact established PIL resize/normalization, not
        # the float-channel transform's slightly different RGB interpolation.
        image, _ = load_image_tensor(row["image_path"], 352)
        sample["rgb"] = image[0].numpy()
        batch = collate_region_samples([sample])
        with torch.inference_mode():
            model.region_fusion.alpha_direct.zero_()
            model.region_fusion.alpha_relation.zero_()
            reference = baseline(batch["rgb"])
            initial = model(**batch)
            initial_diff = max_diff(reference, initial)
            if initial_diff != 0:
                raise AssertionError("Zero-alpha outputs differ from RGB baseline")
            model.region_fusion.alpha_direct.fill_(.1)
            model.region_fusion.alpha_relation.fill_(.1)
            active = model(**batch)
            if len(active) != 4 or not all(t.shape == (1, 1, 352, 352) and torch.isfinite(t).all() for t in active):
                raise AssertionError("Invalid active outputs")
            active_diff = max_diff(reference, active)
            if native_n == 0 and active_diff != 0:
                raise AssertionError("Empty-prior RGB equivalence failed")
        result = {"image_id": row["image_id"], "n_regions": native_n,
                  "valid_regions": int(batch["region_valid"].sum()),
                  "initial_max_logit_diff": initial_diff,
                  "active_untrained_max_logit_diff": active_diff,
                  "output_shapes": [list(t.shape) for t in active], "outputs_finite": True}
        results.append(result)
        print(result, flush=True)
    if C.sha256_file(checkpoint) != ckpt_hash:
        raise AssertionError("Checkpoint changed")
    summary = {"n_images": len(results), "n_regions": sum(r["n_regions"] for r in results),
               "rgb_keys_strictly_loaded": n_keys, "checkpoint_sha256": ckpt_hash,
               "all_zero_alpha_equal_to_rgb": all(r["initial_max_logit_diff"] == 0 for r in results),
               "empty_priors_equal_with_active_alphas": all(r["active_untrained_max_logit_diff"] == 0 for r in results if r["n_regions"] == 0),
               "all_outputs_finite": True, "inference_device": "cpu", "checkpoint_unchanged": True,
               "new_branch_parameters": sum(p.numel() for p in model.region_fusion.parameters()),
               "alpha_test_value": .1, "training": False, "optimizer_steps": 0,
               "gt_used": False, "performance_claim": "none: active branch is randomly initialized",
               "elapsed_s": round(time.time()-started, 1), "results": results,
               "model_code_sha256": C.sha256_file(str(resolve("Network_Relation.py"))),
               "fusion_code_sha256": C.sha256_file(str(resolve("lib/region_fusion.py")))}
    C.write_json(str(output / "NETWORK_CHECK.json"), summary)
    print("REGION_NETWORK_CHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
