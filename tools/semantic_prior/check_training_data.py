"""Read a small real RGB batch; report region coverage without fabricating it."""
import argparse
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import resolve
from dataloader_relation import RelationDataset, collate_relation_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/relation_train_prep_v1.yaml")
    args = ap.parse_args()
    cfg = C.load_config(str(resolve(args.config)))
    root = resolve(cfg["paths"]["out_dir"])
    reports = {}
    for split in ("train", "val"):
        path = root / ("pilot_" + split + ".csv")
        dataset = RelationDataset(path, split, arm="rgb", size=352, seed=cfg["seed"])
        batch = collate_relation_batch([dataset[i] for i in range(min(2, len(dataset)))])
        assert "gt" not in batch["inputs"]
        assert all(torch.isfinite(t).all() for t in batch["inputs"].values())
        assert torch.isfinite(batch["gt"]).all() and batch["gt"].min() >= 0 and batch["gt"].max() <= 1
        n_present = sum(Path(r["region_package"]).is_file() for r in dataset.rows)
        reports[split] = {"rgb_batch_shapes": {k: list(t.shape) for k, t in batch["inputs"].items()},
                          "gt_shape": list(batch["gt"].shape), "n_images": len(dataset),
                          "region_packages_present": n_present,
                          "region_batches_checked": False, "reason": "new-format region priors not generated yet"}
    report = {"splits": reports, "training_started": False, "optimizer_steps": 0,
              "status": "WAITING_FOR_NEW_PRIORS", "qwen_called": False, "sam3_called": False}
    C.write_json(str(root / "LOADER_CHECK.json"), report)
    print(report, flush=True)
    print("RELATION_LOADER_CHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
