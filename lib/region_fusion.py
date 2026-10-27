"""RGB-conditioned region tokens, soft pair relations and residual fusion.

Three independent relation heads are intended for part/whole, context and
redundancy hypotheses. Without explicit relation supervision their semantic
interpretation is NOT guaranteed. Role hints never hard-filter nodes/edges.
"""
import torch
from torch import nn
import torch.nn.functional as F


class RegionRelationFusion(nn.Module):
    def __init__(self, channels, token_dim=64):
        super().__init__()
        self.channels, self.token_dim = channels, token_dim
        self.use_node_geometry = True  # legacy prototype; training ablation explicitly disables in both arms
        self.projection_normalization = "raw"  # opt-in scale-calibrated experiment
        self.node_encoder = nn.Sequential(nn.Linear(channels + 14, token_dim),
                                          nn.LayerNorm(token_dim), nn.GELU())
        self.edge_encoder = nn.Sequential(nn.Linear(2*token_dim + 8, token_dim),
                                          nn.GELU(), nn.Linear(token_dim, 3))
        self.values = nn.ModuleList([nn.Linear(token_dim, token_dim, bias=False) for _ in range(3)])
        self.message = nn.Sequential(nn.Linear(3*token_dim, token_dim, bias=False), nn.GELU())
        self.update_norm = nn.LayerNorm(token_dim)
        self.direct_project = nn.Linear(token_dim, channels)
        self.relation_project = nn.Linear(token_dim, channels, bias=False)
        self.direct_gate = nn.Linear(token_dim + channels, 1)
        self.relation_gate = nn.Linear(token_dim + channels, 1)
        self.alpha_direct = nn.Parameter(torch.zeros(()))
        self.alpha_relation = nn.Parameter(torch.zeros(()))
        self.quality_head = None  # opt-in; legacy state_dict remains unchanged
        self.quality_to_direct = None
        self.quality_to_relation = None
        self.quality_to_edges = None
        self.quality_stop_gradient = True
        self.use_quality_feedback = False

    def enable_quality_head(self, seed, stop_gradient=True, feedback=False):
        """Visual-only quality estimator; its score is never a hard fusion gate.

        Initialization preserves the caller's CPU RNG stream. COD adapters are
        zero-initialized, signed and learned from segmentation loss only.
        """
        if self.quality_head is not None:
            raise ValueError('Quality head already enabled')
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
            self.quality_head = nn.Sequential(
                nn.Linear(2*self.channels + 1, self.token_dim),
                nn.LayerNorm(self.token_dim), nn.GELU(), nn.Linear(self.token_dim, 1))
            self.quality_to_direct = nn.Linear(1, 1, bias=False)
            self.quality_to_relation = nn.Linear(1, 1, bias=False)
            self.quality_to_edges = nn.Linear(2, 3, bias=False)
            for module in (self.quality_to_direct, self.quality_to_relation, self.quality_to_edges):
                nn.init.zeros_(module.weight)
        self.quality_stop_gradient = bool(stop_gradient)
        self.use_quality_feedback = bool(feedback)

    def forward(self, features, region_masks, region_valid, role_hint,
                role_hint_valid, relation_geom, relation_valid, return_debug=False):
        b, c, h, w = features.shape
        if region_masks.ndim != 4 or region_masks.shape[0] != b:
            raise ValueError("region_masks must have shape [B,N,H,W]")
        n = region_masks.shape[1]
        expected = ((region_valid, (b, n)), (role_hint, (b, n, 4)),
                    (role_hint_valid, (b, n)), (relation_geom, (b, n, n, 8)),
                    (relation_valid, (b, n, n)))
        if c != self.channels or any(tuple(t.shape) != shape for t, shape in expected):
            raise ValueError("Region feature/interface shape mismatch")
        if n == 0:
            debug = {"direct_map": torch.zeros_like(features), "relation_map": torch.zeros_like(features),
                     "edge_weights": features.new_zeros((b, 0, 0, 3)),
                     "direct_logits": features.new_zeros((b, 0)),
                     "direct_gates": features.new_zeros((b, 0)),
                     "effective_valid": region_valid.bool(), "coverage": features.new_zeros((b, 0, h, w))}
            if self.quality_head is not None:
                debug['quality_logits'] = features.new_zeros((b, 0))
                debug['quality_scores'] = features.new_zeros((b, 0))
            return (features, debug) if return_debug else features
        # Area downsampling preserves fractional coverage of tiny masks.
        coverage = F.interpolate(region_masks.float(), size=(h, w), mode="area")
        valid = region_valid.bool() & (coverage.sum(dim=(-2, -1)) > 0)
        coverage = coverage * valid[:, :, None, None]
        denom = coverage.sum(dim=(-2, -1)).clamp_min(1e-8)
        pooled = torch.einsum("bnhw,bchw->bnc", coverage, features.float()) / denom[:, :, None]
        pairs = (relation_valid.bool() & valid[:, :, None] & valid[:, None, :]
                 & ~torch.eye(n, device=features.device, dtype=torch.bool)[None])
        geom = relation_geom.to(features.dtype) * pairs[:, :, :, None]
        geom_summary = geom.sum(dim=2) / pairs.sum(dim=2).clamp_min(1)[:, :, None]
        if not self.use_node_geometry:
            geom_summary = torch.zeros_like(geom_summary)
        hints_valid = role_hint_valid.bool() & valid
        hints = role_hint.to(features.dtype) * hints_valid[:, :, None]
        node_input = torch.cat([pooled.to(features.dtype), hints,
                                hints_valid[:, :, None].to(features.dtype), geom_summary,
                                coverage.mean(dim=(-2, -1))[:, :, None].to(features.dtype)], dim=-1)
        tokens = self.node_encoder(node_input) * valid[:, :, None]
        image_context = features.mean(dim=(-2, -1))[:, None, :].expand(-1, n, -1)
        quality_logits, quality_scores, quality_signal = None, None, None
        if self.quality_head is not None:
            # No weak roles, text, GT or pair geometry in the quality head.
            quality_input = torch.cat([pooled.to(features.dtype), image_context,
                                       coverage.mean((-2, -1))[:, :, None].to(features.dtype)], -1)
            if self.quality_stop_gradient:
                quality_input = quality_input.detach()
            quality_logits = self.quality_head(quality_input).squeeze(-1)
            quality_scores = quality_logits.sigmoid()
            # Segmentation cannot redefine quality by backpropagating through
            # this side channel; COD learns how/when to consume the score.
            quality_signal = (quality_scores.detach() - .5)[:, :, None] * valid[:, :, None]
        left = tokens[:, :, None].expand(-1, -1, n, -1)
        right = tokens[:, None, :].expand(-1, n, -1, -1)
        edge_logits = self.edge_encoder(torch.cat([left, right, geom], dim=-1))
        if self.use_quality_feedback:
            if quality_signal is None:
                raise ValueError('Quality feedback requires the independent head')
            qi = quality_signal[:, :, None].expand(-1, -1, n, -1)
            qj = quality_signal[:, None, :].expand(-1, n, -1, -1)
            edge_logits = edge_logits + self.quality_to_edges(torch.cat([qi, qj], -1))
        weights = edge_logits.sigmoid()
        weights = weights * pairs[:, :, :, None]
        values = torch.stack([layer(tokens) for layer in self.values], dim=2)
        messages = torch.einsum("bijr,bjrd->bird", weights, values)
        # +1 retains absolute gate strength, unlike normalization by weights
        # alone which would force a singleton helper to contribute fully.
        messages = messages / (1 + weights.sum(dim=2))[:, :, :, None]
        updated = self.update_norm(tokens + self.message(messages.flatten(2)))
        # No actual pair -> no relation branch (a singleton still has direct support).
        has_pair = pairs.any(dim=2) & valid
        direct_logits = self.direct_gate(torch.cat([tokens, image_context], dim=-1))
        relation_logits = self.relation_gate(torch.cat([updated, image_context], dim=-1))
        if self.use_quality_feedback:
            direct_logits = direct_logits + self.quality_to_direct(quality_signal)
            relation_logits = relation_logits + self.quality_to_relation(quality_signal)
        direct_gates = direct_logits.sigmoid()
        relation_gates = relation_logits.sigmoid()
        relation_delta = updated - self.update_norm(tokens)
        direct_projection = self.direct_project(tokens)
        relation_projection = self.relation_project(relation_delta)
        if self.projection_normalization == "feature_rms":
            # Normalize BEFORE learned gates, so suppressing a gate still
            # suppresses its contribution rather than being divided away.
            # Detach RGB scale: branch scale must not create a gradient shortcut.
            scale = features.detach().float().square().mean((1,2,3)).sqrt()[:,None,None]
            def calibrate(projection):
                rms = (projection.float().square().mean(-1,keepdim=True)+1e-6).sqrt()
                return (projection.float()/rms*scale).to(features.dtype)
            direct_projection = calibrate(direct_projection)
            relation_projection = calibrate(relation_projection)
        elif self.projection_normalization != "raw":
            raise ValueError("Unknown projection normalization")
        direct_tokens = direct_projection * direct_gates * valid[:, :, None]
        relation_tokens = relation_projection * relation_gates * has_pair[:, :, None]
        # Fixed coverage denominator, independent of learned gates: all gates
        # may approach zero without their effect being divided back out.
        spatial_denom = 1 + coverage.sum(dim=1, keepdim=True)
        direct_map = torch.einsum("bnc,bnhw->bchw", direct_tokens, coverage.to(features.dtype)) / spatial_denom
        relation_map = torch.einsum("bnc,bnhw->bchw", relation_tokens, coverage.to(features.dtype)) / spatial_denom
        fused = features + self.alpha_direct * direct_map + self.alpha_relation * relation_map
        if return_debug:
            debug = {"direct_map": direct_map, "relation_map": relation_map,
                     "direct_logits": direct_logits.squeeze(-1),
                     "direct_gates": direct_gates.squeeze(-1),
                     "edge_weights": weights, "effective_valid": valid, "coverage": coverage}
            if quality_logits is not None:
                debug.update(quality_logits=quality_logits, quality_scores=quality_scores)
            return fused, debug
        return fused
