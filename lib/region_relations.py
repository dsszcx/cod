"""CPU region geometry, numeric package loading and variable-N batching.

No image names, Qwen scores, labels or GT are used to decide node validity.
All nonempty candidate masks remain eligible regardless of proposed role.
"""
from pathlib import Path

import numpy as np

ROLE_NAMES = ("whole", "part", "context", "unknown")
GEOM_NAMES = ("iou", "contain_i_in_j", "contain_j_in_i", "area_log_ratio",
              "dx", "dy", "bbox_gap", "near_duplicate")
PACKAGE_KEYS = ("masks", "role_hint", "role_hint_valid", "sam3_score",
                "relation_geom", "relation_valid")


def binary_masks(masks):
    arr = np.asarray(masks)
    if arr.ndim != 3 or min(arr.shape[1:]) <= 0:
        raise ValueError("masks must have shape [N,H,W] with positive H,W")
    if not np.isfinite(arr).all() or not np.isin(arr, [0, 1]).all():
        raise ValueError("masks must contain only finite 0/1 values")
    return arr.astype(bool, copy=False)


def region_geometry(masks, region_valid=None):
    masks = binary_masks(masks)
    n, h, w = masks.shape
    valid = masks.any(axis=(1, 2))
    if region_valid is not None:
        explicit = np.asarray(region_valid, bool)
        if explicit.shape != (n,):
            raise ValueError("region_valid must have shape [N]")
        valid &= explicit
    geom = np.zeros((n, n, len(GEOM_NAMES)), np.float32)
    pair_valid = valid[:, None] & valid[None, :] & ~np.eye(n, dtype=bool)
    areas = masks.sum(axis=(1, 2), dtype=np.float64)
    boxes, centers = {}, {}
    for i in np.flatnonzero(valid):
        ys, xs = np.nonzero(masks[i])
        boxes[i] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        centers[i] = (xs.mean() + .5, ys.mean() + .5)  # pixel-center centroid
    for i, j in zip(*np.nonzero(pair_valid)):
        inter = np.logical_and(masks[i], masks[j]).sum()
        iou = inter / (areas[i] + areas[j] - inter)
        a, b = boxes[i], boxes[j]
        gx = max(0, a[0] - b[2], b[0] - a[2])
        gy = max(0, a[1] - b[3], b[1] - a[3])
        geom[i, j] = (iou, inter / areas[i], inter / areas[j],
                      np.tanh(np.log(areas[i] / areas[j])),
                      (centers[j][0] - centers[i][0]) / w,
                      (centers[j][1] - centers[i][1]) / h,
                      np.hypot(gx, gy) / np.hypot(h, w), float(iou > .9))
    return geom, pair_valid, valid


def prepare_region_sample(sample):
    """Recompute validity and geometry after ANY spatial transform."""
    result = dict(sample)
    masks = binary_masks(sample["masks"])
    n = len(masks)
    roles = np.asarray(sample["role_hint"], np.float32)
    role_valid = np.asarray(sample["role_hint_valid"], bool)
    if roles.shape != (n, 4) or role_valid.shape != (n,):
        raise ValueError("Role arrays do not match masks")
    if not np.isfinite(roles).all() or np.any(roles < 0):
        raise ValueError("Invalid role hints")
    geom, pair_valid, valid = region_geometry(masks, sample.get("region_valid"))
    result.update(masks=masks.astype(np.uint8), role_hint=roles,
                  role_hint_valid=role_valid & valid, relation_geom=geom,
                  relation_valid=pair_valid, region_valid=valid)
    return result


def load_region_package(path):
    """Load numeric arrays only; never enable pickle. Check stored geometry."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != set(PACKAGE_KEYS):
            raise ValueError("Unexpected package fields")
        sample = {k: archive[k].copy() for k in PACKAGE_KEYS}
    if sample["masks"].dtype != np.uint8 or sample["role_hint"].dtype != np.float32:
        raise ValueError("Unexpected mask/role dtype")
    for name in ("role_hint_valid", "relation_valid"):
        if sample[name].dtype != np.bool_:
            raise ValueError("Expected boolean " + name)
    scores = sample["sam3_score"]
    if (scores.shape != (len(sample["masks"]),) or scores.dtype != np.float32
            or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))):
        raise ValueError("Invalid stored SAM3 scores")
    if sample["relation_geom"].dtype != np.float32:
        raise ValueError("Expected float32 geometry")
    prepared = prepare_region_sample(sample)
    for key in ("relation_geom", "relation_valid"):
        if sample[key].shape != prepared[key].shape or not np.allclose(sample[key], prepared[key], atol=1e-6, rtol=0):
            raise ValueError("Stored geometry is inconsistent with masks: " + key)
    return prepared


def transform_region_sample(sample, crop=None, hflip=False, output_hw=None):
    """Apply identical crop/flip/resize to RGB and ALL masks, then recompute.

    RGB is float CHW (normalization may already have been applied). Crop is
    integer (left, top, right, bottom), right/bottom exclusive. No GT input.
    A future training loader must apply these SAME parameters to its GT.
    """
    from PIL import Image

    masks = binary_masks(sample["masks"]).astype(np.uint8)
    rgb = np.asarray(sample["rgb"], np.float32)
    if rgb.shape != (3,) + masks.shape[1:] or not np.isfinite(rgb).all():
        raise ValueError("RGB must be finite CHW aligned with masks")
    h, w = masks.shape[1:]
    if crop is not None:
        x1, y1, x2, y2 = crop
        if not all(isinstance(v, (int, np.integer)) for v in crop) or not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError("Invalid crop")
        rgb, masks = rgb[:, y1:y2, x1:x2], masks[:, y1:y2, x1:x2]
    if hflip:
        rgb, masks = rgb[:, :, ::-1], masks[:, :, ::-1]
    if output_hw is not None:
        oh, ow = map(int, output_hw)
        if min(oh, ow) <= 0:
            raise ValueError("Output size must be positive")
        rgb = np.stack([np.asarray(Image.fromarray(c).resize((ow, oh), Image.Resampling.BILINEAR)) for c in rgb])
        masks = (np.stack([np.asarray(Image.fromarray(m).resize((ow, oh), Image.Resampling.NEAREST)) for m in masks])
                 if len(masks) else np.zeros((0, oh, ow), np.uint8))
    result = dict(sample, rgb=np.ascontiguousarray(rgb), masks=np.ascontiguousarray(masks))
    # Never reuse stale region_valid after a spatial transformation.
    result.pop("region_valid", None)
    return prepare_region_sample(result)


def collate_region_samples(samples, as_torch=True):
    """Pad variable N, recompute geometry and exclude padding from all pairs.

    No candidate limit or role-based filtering. SAM3 scores and string metadata
    are deliberately NOT in the model input dictionary.
    """
    if not samples:
        raise ValueError("Empty batch")
    prepared = [prepare_region_sample(s) for s in samples]
    h, w = prepared[0]["masks"].shape[1:]
    b, nmax = len(prepared), max(1, max(len(s["masks"]) for s in prepared))
    batch = {"rgb": np.zeros((b, 3, h, w), np.float32),
             "region_masks": np.zeros((b, nmax, h, w), np.float32),
             "region_valid": np.zeros((b, nmax), bool),
             "role_hint": np.zeros((b, nmax, 4), np.float32),
             "role_hint_valid": np.zeros((b, nmax), bool),
             "relation_geom": np.zeros((b, nmax, nmax, 8), np.float32),
             "relation_valid": np.zeros((b, nmax, nmax), bool)}
    for i, sample in enumerate(prepared):
        n = len(sample["masks"])
        rgb = np.asarray(sample["rgb"], np.float32)
        if sample["masks"].shape[1:] != (h, w) or rgb.shape != (3, h, w) or not np.isfinite(rgb).all():
            raise ValueError("Batch RGB/masks must have one aligned spatial size")
        batch["rgb"][i] = rgb
        batch["region_masks"][i, :n] = sample["masks"] * sample["region_valid"][:, None, None]
        batch["region_valid"][i, :n] = sample["region_valid"]
        batch["role_hint"][i, :n] = sample["role_hint"] * sample["role_hint_valid"][:, None]
        batch["role_hint_valid"][i, :n] = sample["role_hint_valid"]
        batch["relation_geom"][i, :n, :n] = sample["relation_geom"]
        batch["relation_valid"][i, :n, :n] = sample["relation_valid"]
    if as_torch:
        import torch
        return {k: torch.from_numpy(v) for k, v in batch.items()}
    return batch
