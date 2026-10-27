"""Manifest-driven RGB/region loader. GT is returned separately from inputs."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from tools.e0e4 import common as C
from lib.region_relations import load_region_package, transform_region_sample, collate_region_samples


def spatial_plan(hw, training, seed, epoch, image_id):
    h, w = hw
    if not training:
        return None, False
    digest = hashlib.sha256((str(seed) + ":" + str(epoch) + ":" + image_id).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    ch, cw = max(1, int(h*rng.uniform(.85, 1))), max(1, int(w*rng.uniform(.85, 1)))
    y, x = int(rng.integers(h-ch+1)), int(rng.integers(w-cw+1))
    return (x, y, x+cw, y+ch), bool(rng.integers(2))


def transform_gt(gt, crop, flip, size):
    if crop is not None:
        gt = gt.crop(crop)
    if flip:
        gt = gt.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    arr = np.asarray(gt.resize((size, size), Image.Resampling.NEAREST), np.float32)[None]
    return arr if arr.max() <= 1 else arr / 255.


class RelationDataset(Dataset):
    def __init__(self, manifest, split, arm="relation", size=352, seed=20261022):
        if split not in ("train", "val") or arm not in ("rgb", "direct", "relation"):
            raise ValueError("Invalid split or ablation arm")
        self.rows = C.read_csv(str(manifest))
        if not self.rows or any(r["split"] != split for r in self.rows):
            raise ValueError("Manifest must contain only the requested nonempty split")
        if len({r["image_id"] for r in self.rows}) != len(self.rows):
            raise ValueError("Duplicate image IDs")
        self.split, self.arm, self.size, self.seed, self.epoch = split, arm, int(size), seed, 0
        if size <= 0:
            raise ValueError("Invalid image size")

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        row = self.rows[index]
        if C.sha256_file(row["image_path"]) != row["image_sha256"]:
            raise ValueError("Image content changed: " + row["image_id"])
        with Image.open(row["image_path"]) as im:
            rgb = np.asarray(im.convert("RGB"), np.float32).transpose(2, 0, 1) / 255.
        h, w = rgb.shape[1:]
        with Image.open(row["gt_path"]) as im:
            if im.size != (w, h):
                raise ValueError("GT/RGB size mismatch")
            gt = im.convert("L")
        # Valid empty package != missing/failed package. Never silently replace
        # missing priors with zeros in either region ablation.
        if self.arm == "rgb":
            sample = {"masks": np.zeros((0, h, w), np.uint8), "role_hint": np.zeros((0, 4), np.float32),
                      "role_hint_valid": np.zeros(0, bool)}
        else:
            path = Path(row["region_package"])
            meta = C.read_json(str(path.with_suffix(".json")))
            if meta is None or meta["image_sha256"] != row["image_sha256"] or meta["candidate_generation_status"] not in ("HAS_REGIONS", "NO_CANDIDATES", "NO_SEGMENTABLE_TAGS"):
                raise ValueError("Missing, failed or mismatched region package: " + row["image_id"])
            if C.sha256_file(str(path)) != meta["package_sha256"]:
                raise ValueError("Region package hash mismatch")
            sample = load_region_package(path)
            if sample["masks"].shape[1:] != (h, w):
                raise ValueError("Region/RGB dimensions mismatch")
        crop, flip = spatial_plan((h, w), self.split == "train", self.seed, self.epoch, row["image_id"])
        sample["rgb"] = rgb
        sample = transform_region_sample(sample, crop=crop, hflip=flip, output_hw=(self.size, self.size))
        sample["rgb"] = (sample["rgb"] - np.array([.485, .456, .406], np.float32)[:, None, None]) / np.array([.229, .224, .225], np.float32)[:, None, None]
        # GT never enters region geometry or the model argument dictionary.
        return {"inputs": sample, "gt": torch.from_numpy(transform_gt(gt, crop, flip, self.size)).float(),
                "image_id": row["image_id"], "native_hw": (h, w), "spatial_plan": (crop, flip)}


def collate_relation_batch(items):
    return {"inputs": collate_region_samples([x["inputs"] for x in items]),
            "gt": torch.stack([x["gt"] for x in items]),
            "image_ids": [x["image_id"] for x in items],
            "native_hw": [x["native_hw"] for x in items]}
