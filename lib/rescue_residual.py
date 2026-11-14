"""Frozen original COD plus an independently normalized rescue-only residual.

Inputs are separated by candidate provenance before reaching this module.
No target labels, confidence thresholds or candidate filtering occur here.
"""
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F


class RescueResidual(nn.Module):
    def __init__(self, original_fusion, learn_candidate_gate=True):
        super().__init__()
        self.node_encoder = deepcopy(original_fusion.node_encoder)
        self.project = deepcopy(original_fusion.direct_project)
        self.gate = deepcopy(original_fusion.direct_gate)
        self.beta = nn.Parameter(torch.zeros(()))
        for module in (self.node_encoder, self.project):
            module.requires_grad_(False)
        self.gate.requires_grad_(learn_candidate_gate)

    def forward(self, features, inputs, zero_gates=False):
        masks, valid = inputs['region_masks'], inputs['region_valid'].bool()
        b, c, h, w = features.shape
        if masks.shape[:2] != valid.shape or masks.shape[0] != b:
            raise ValueError('Rescue masks/valid shape mismatch')
        if masks.shape[1] == 0:
            return torch.zeros_like(features), valid
        coverage = F.interpolate(masks.float(), size=(h, w), mode='area')
        valid = valid & (coverage.sum((-2, -1)) > 0)
        coverage = coverage * valid[:, :, None, None]
        pooled = torch.einsum('bnhw,bchw->bnc', coverage, features.float())
        pooled = pooled / coverage.sum((-2, -1)).clamp_min(1e-8)[:, :, None]
        hints_valid = inputs['role_hint_valid'].bool() & valid
        hints = inputs['role_hint'].to(features.dtype) * hints_valid[:, :, None]
        geometry = features.new_zeros((b, masks.shape[1], 8))
        node = torch.cat((pooled.to(features.dtype), hints,
                          hints_valid[:, :, None].to(features.dtype), geometry,
                          coverage.mean((-2, -1))[:, :, None].to(features.dtype)), -1)
        tokens = self.node_encoder(node) * valid[:, :, None]
        context = features.mean((-2, -1))[:, None].expand(-1, masks.shape[1], -1)
        gates = self.gate(torch.cat((tokens, context), -1)).sigmoid().squeeze(-1)
        gates = gates * valid
        if zero_gates:
            gates = torch.zeros_like(gates)
        projection = self.project(tokens)
        scale = features.detach().float().square().mean((1, 2, 3)).sqrt()[:, None, None]
        projection = projection.float() / (projection.float().square().mean(-1, keepdim=True) + 1e-6).sqrt() * scale
        support = coverage * gates[:, :, None, None]
        residual = torch.einsum('bnc,bnhw->bchw', projection, support) / (1 + support.sum(1, keepdim=True))
        return residual.to(features.dtype), valid


class FrozenRescueCOD(nn.Module):
    def __init__(self, base, learn_candidate_gate=True):
        super().__init__()
        if base.region_fusion.use_node_geometry or base.region_fusion.use_quality_feedback:
            raise ValueError('First rescue protocol requires legacy direct configuration')
        self.base = base.requires_grad_(False).eval()
        self.rescue = RescueResidual(base.region_fusion, learn_candidate_gate)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # Freeze normalization buffers as well as parameters.
        return self

    def forward(self, original_inputs, rescue_inputs, enabled=True, zero_gates=False):
        if not enabled:
            return self.base(**original_inputs)
        if not bool(rescue_inputs['region_valid'].any()):
            return self.base(**original_inputs)
        base = self.base
        with torch.no_grad():
            endpoints = base.backbone.extract_endpoints(original_inputs['rgb'])
            x2 = base.rfb2_1(endpoints['reduction_3'])
            x3 = base.rfb3_1(endpoints['reduction_4'])
            x4 = base.rfb4_1(endpoints['reduction_5'])
            keys = ('region_masks', 'region_valid', 'role_hint', 'role_hint_valid', 'relation_geom', 'relation_valid')
            fused = base.region_fusion(x2, *(original_inputs[k] for k in keys))
        residual, valid = self.rescue(x2, rescue_inputs, zero_gates)
        adjusted = fused + self.rescue.beta * residual
        # Empty examples in a mixed batch preserve the original feature exactly.
        x2 = torch.where(valid.any(1)[:, None, None, None], adjusted, fused)
        coarse = base.NCD(x4, x3, x2)
        coarse_pred = F.interpolate(coarse, scale_factor=8, mode='bilinear', align_corners=True)
        guidance = F.interpolate(coarse, scale_factor=.25, mode='bilinear', align_corners=True)
        s5 = base.RS5(x4, guidance) + guidance
        s5_pred = F.interpolate(s5, scale_factor=32, mode='bilinear', align_corners=True)
        guidance = F.interpolate(s5, scale_factor=2, mode='bilinear', align_corners=True)
        s4 = base.RS4(x3, guidance) + guidance
        s4_pred = F.interpolate(s4, scale_factor=16, mode='bilinear', align_corners=True)
        guidance = F.interpolate(s4, scale_factor=2, mode='bilinear', align_corners=True)
        s3 = base.RS3(x2, guidance) + guidance
        s3_pred = F.interpolate(s3, scale_factor=8, mode='bilinear', align_corners=True)
        return coarse_pred, s5_pred, s4_pred, s3_pred
