"""Train-only foreground-purity supervision, NOT semantic relation labels.

Called after the RGB-only/prior-only forward. GT never enters model inputs,
candidate generation, node validity or inference gating. A low-purity node
may still be valuable context: relation edges/messages are not supervised or
masked here. All quantities use the same post-augmentation pixel grid.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def candidate_targets(region_masks, region_valid, gt):
    if region_masks.ndim != 4 or gt.ndim != 4 or gt.shape[1] != 1:
        raise ValueError("Expected masks [B,N,H,W], GT [B,1,H,W]")
    b, n, h, w = region_masks.shape
    if tuple(gt.shape) != (b, 1, h, w) or tuple(region_valid.shape) != (b, n):
        raise ValueError("Candidates and GT must share the augmented spatial grid")
    masks, target = region_masks.detach().float(), gt.detach().float()
    if (not torch.isfinite(masks).all() or not torch.isfinite(target).all()
            or (masks < 0).any() or (masks > 1).any()
            or (target < 0).any() or (target > 1).any()):
        raise ValueError("Candidates and GT must be finite values in [0,1]")
    areas = masks.sum((-2, -1))
    valid = region_valid.bool() & (areas > 0)
    intersection = (masks * target).sum((-2, -1))
    purity = intersection / areas.clamp_min(1e-8)
    # Diagnostic only: a useful small part can have low recall but high purity.
    recall = intersection / target.sum((-2, -1)).clamp_min(1e-8)
    return {"purity": purity * valid, "foreground_recall": recall * valid,
            "valid": valid}


def image_balanced_mean(values, valid):
    """Equal nodes within each nonempty image; equal nonempty images per batch."""
    if values.shape != valid.shape or values.ndim != 2:
        raise ValueError("Expected aligned [B,N] values and validity")
    counts = valid.sum(1)
    per_image = (values * valid).sum(1) / counts.clamp_min(1)
    return per_image.sum() / (counts > 0).sum().clamp_min(1)


def candidate_reliability_loss(logits, region_masks, region_valid, gt):
    targets = candidate_targets(region_masks, region_valid, gt)
    if logits.shape != targets["purity"].shape:
        raise ValueError("Gate logits must be [B,N]")
    loss = F.binary_cross_entropy_with_logits(logits.float(), targets["purity"], reduction="none")
    return image_balanced_mean(loss, targets["valid"])
