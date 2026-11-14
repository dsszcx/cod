"""Exploratory non-saturating correction; not a trained-performance result.

The copied original gate stays fixed. A fresh multiplier starts at exactly one,
preserving the original scalar branch for any beta, while providing a new
gradient path if the copied sigmoid gate is saturated.
"""
import torch
from torch import nn
import torch.nn.functional as F
from lib.rescue_residual import RescueResidual, FrozenRescueCOD


class ModulatedRescueResidual(RescueResidual):
    def __init__(self, original_fusion):
        super().__init__(original_fusion, learn_candidate_gate=False)
        self.modulation = nn.Linear(original_fusion.direct_gate.in_features, 1)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

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
        node = torch.cat((pooled.to(features.dtype), hints, hints_valid[:, :, None].to(features.dtype),
                          features.new_zeros((b, masks.shape[1], 8)),
                          coverage.mean((-2, -1))[:, :, None].to(features.dtype)), -1)
        tokens = self.node_encoder(node) * valid[:, :, None]
        context = features.mean((-2, -1))[:, None].expand(-1, masks.shape[1], -1)
        gate_input = torch.cat((tokens, context), -1)
        original_gate = self.gate(gate_input).sigmoid().squeeze(-1)
        multiplier = 2 * self.modulation(gate_input).sigmoid().squeeze(-1)
        # This is a support multiplier in [0,2], not a quality probability.
        gates = original_gate * multiplier * valid
        if zero_gates:
            gates = torch.zeros_like(gates)
        projection = self.project(tokens)
        scale = features.detach().float().square().mean((1, 2, 3)).sqrt()[:, None, None]
        projection = projection.float() / (projection.float().square().mean(-1, keepdim=True) + 1e-6).sqrt() * scale
        support = coverage * gates[:, :, None, None]
        residual = torch.einsum('bnc,bnhw->bchw', projection, support) / (1 + support.sum(1, keepdim=True))
        return residual.to(features.dtype), valid


class ModulatedRescueCOD(FrozenRescueCOD):
    def __init__(self, base):
        super().__init__(base, learn_candidate_gate=False)
        self.rescue = ModulatedRescueResidual(base.region_fusion)
