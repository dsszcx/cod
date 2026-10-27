"""Shared initialization, ablation freezing, loss and state fingerprints."""
import hashlib
import random

import numpy as np
import torch
import torch.nn.functional as F

from Network_PVTv2 import Network as RGBNetwork
from Network_Relation import Network as RelationNetwork

RELATION_PREFIXES = ("edge_encoder.", "values.", "message.", "update_norm.",
                     "relation_project.", "relation_gate.", "alpha_relation",
                     "quality_to_relation.", "quality_to_edges.")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_model(arm, seed, pretrained_path, fusion_options=None, quality_options=None):
    if arm not in ("rgb", "direct", "relation"):
        raise ValueError("Unknown ablation arm")
    seed_all(seed)
    rgb = RGBNetwork(channel=64, imagenet_pretrained=False, use_ca=False)
    state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    own = rgb.backbone.state_dict()
    # The ImageNet classification head is absent in this backbone. Reject all
    # other unexpected/missing tensors instead of silently partially loading.
    state = {k: v for k, v in state.items() if k not in ("head.weight", "head.bias") or k in own}
    rgb.backbone.load_state_dict(state, strict=True)
    rgb_digest = state_digest(rgb.state_dict())
    if arm == "rgb":
        if quality_options is not None:
            raise ValueError('Quality head requires a prior arm')
        return rgb, {"rgb_initial_sha256": rgb_digest, "frozen_parameters": [], "node_geometry": False}
    model = RelationNetwork(channel=64, imagenet_pretrained=False, use_ca=False)
    model.load_rgb_state_dict(rgb.state_dict())
    model.region_fusion.use_node_geometry = False
    options = dict(fusion_options or {})
    if set(options)-{"alpha_init","projection_normalization"}:
        raise ValueError("Unknown fusion options")
    normalization = options.get("projection_normalization","raw")
    if normalization not in ("raw","feature_rms"):
        raise ValueError("Unknown projection normalization")
    alpha_init = float(options.get("alpha_init",0.))
    if not np.isfinite(alpha_init): raise ValueError("Nonfinite alpha initialization")
    model.region_fusion.projection_normalization = normalization
    with torch.no_grad():
        model.region_fusion.alpha_direct.fill_(alpha_init)
        model.region_fusion.alpha_relation.fill_(alpha_init)
    # Fingerprint common branch initialization BEFORE the direct-arm fixed-zero
    # intervention. Effective states are separately recorded below.
    region_digest = state_digest(model.region_fusion.state_dict())
    quality_meta = None
    if quality_options is not None:
        quality_meta = dict(quality_options)
        if set(quality_meta) - {'stop_gradient', 'feedback'}:
            raise ValueError('Unknown independent quality options')
        quality_meta = {'stop_gradient': bool(quality_meta.get('stop_gradient', True)),
                        'feedback': bool(quality_meta.get('feedback', False))}
        model.region_fusion.enable_quality_head(seed + 1000003, **quality_meta)
    frozen = []
    if arm == "direct":
        for name, param in model.region_fusion.named_parameters():
            if name.startswith(RELATION_PREFIXES):
                param.requires_grad_(False)
                frozen.append("region_fusion."+name)
        with torch.no_grad():
            model.region_fusion.alpha_relation.zero_()
    return model, {"rgb_initial_sha256": rgb_digest, "frozen_parameters": frozen,
                   "region_initial_sha256": region_digest,
                   "effective_region_initial_sha256":state_digest(model.region_fusion.state_dict()),
                   "fusion_options":{"alpha_init":alpha_init,"projection_normalization":normalization},
                   "quality_options": quality_meta,
                   "quality_initial_sha256": (state_digest(model.region_fusion.quality_head.state_dict())
                                              if quality_meta is not None else None),
                   "node_geometry": False}


def optimizer_parameters(model):
    """Keep auxiliary-only gradients OUT of the main optimizer's clip norm."""
    fusion = getattr(model, 'region_fusion', None)
    head = getattr(fusion, 'quality_head', None)
    quality = [p for p in head.parameters() if p.requires_grad] if head is not None else []
    quality_ids = {id(p) for p in quality}
    main = [p for p in model.parameters() if p.requires_grad and id(p) not in quality_ids]
    return main, quality


def core_state_dict(model):
    """Legacy COD parameters/buffers, excluding independent quality additions."""
    return {k: v for k, v in model.state_dict().items()
            if not k.startswith(('region_fusion.quality_head.', 'region_fusion.quality_to_'))}


def structure_loss(logits, mask):
    logits, mask = logits.float(), mask.float()
    weight = 1 + 5 * (F.avg_pool2d(mask, 31, stride=1, padding=15)-mask).abs()
    bce = (weight * F.binary_cross_entropy_with_logits(logits, mask, reduction="none")).sum((2, 3)) / (weight.sum((2, 3))+1e-8)
    pred = logits.sigmoid()
    a = pred - pred.mean((2, 3), keepdim=True)
    b = mask - mask.mean((2, 3), keepdim=True)
    enhanced = ((1 + (2*a*b+1e-8)/(a*a+b*b+1e-8))**2 / 4).mean((2, 3))
    inter = (pred*mask*weight).sum((2, 3))
    union = ((pred+mask)*weight).sum((2, 3))
    wiou = 1 - (inter+1+1e-8)/(union-inter+1+1e-8)
    return (bce + 1-enhanced + wiou).mean()


def deep_loss(outputs, mask):
    if len(outputs) != 4:
        raise ValueError("Expected four supervision outputs")
    return sum(structure_loss(o, mask) for o in outputs)


def forward_arm(model, inputs, arm):
    return model(inputs["rgb"]) if arm == "rgb" else model(**inputs)
