"""Independent 3ch RGB COD + candidate relation branch; no old entry changes."""
import torch.nn.functional as F
from Network_PVTv2 import Network as RGBNetwork
from lib.region_fusion import RegionRelationFusion


class Network(RGBNetwork):
    def __init__(self, channel=64, imagenet_pretrained=False, use_ca=False, token_dim=64):
        super().__init__(channel=channel, imagenet_pretrained=imagenet_pretrained, use_ca=use_ca)
        self.region_fusion = RegionRelationFusion(channel, token_dim=token_dim)

    def load_rgb_state_dict(self, state):
        """Strictly validate ALL baseline keys/shapes before copying any tensor.

        Only the explicitly new region_fusion.* tensors are excluded. This
        method accepts a bare state_dict, not arbitrary checkpoint wrappers.
        """
        own = self.state_dict()
        expected = {k for k in own if not k.startswith("region_fusion.")}
        if set(state) != expected:
            raise ValueError("RGB checkpoint key mismatch: missing=%s extra=%s" %
                             (sorted(expected-set(state)), sorted(set(state)-expected)))
        bad = [k for k in expected if own[k].shape != state[k].shape]
        if bad:
            raise ValueError("RGB checkpoint shape mismatch: " + str(bad))
        merged = dict(own)
        merged.update(state)
        self.load_state_dict(merged, strict=True)
        return len(expected)

    def forward(self, rgb, region_masks, region_valid, role_hint, role_hint_valid,
                relation_geom, relation_valid, return_aux=False):
        endpoints = self.backbone.extract_endpoints(rgb)
        x2 = self.rfb2_1(endpoints["reduction_3"])
        x3 = self.rfb3_1(endpoints["reduction_4"])
        x4 = self.rfb4_1(endpoints["reduction_5"])
        fused = self.region_fusion(x2, region_masks, region_valid, role_hint,
                                  role_hint_valid, relation_geom, relation_valid,
                                  return_debug=return_aux)
        if return_aux:
            x2, auxiliary = fused
        else:
            x2 = fused
        # Same RGB decoder order/interpolation as Network_PVTv2. The fused
        # x2 feeds BOTH the coarse decoder and the last reverse stage.
        coarse = self.NCD(x4, x3, x2)
        coarse_pred = F.interpolate(coarse, scale_factor=8, mode="bilinear", align_corners=True)
        guidance = F.interpolate(coarse, scale_factor=.25, mode="bilinear", align_corners=True)
        s5 = self.RS5(x4, guidance) + guidance
        s5_pred = F.interpolate(s5, scale_factor=32, mode="bilinear", align_corners=True)
        guidance = F.interpolate(s5, scale_factor=2, mode="bilinear", align_corners=True)
        s4 = self.RS4(x3, guidance) + guidance
        s4_pred = F.interpolate(s4, scale_factor=16, mode="bilinear", align_corners=True)
        guidance = F.interpolate(s4, scale_factor=2, mode="bilinear", align_corners=True)
        s3 = self.RS3(x2, guidance) + guidance
        s3_pred = F.interpolate(s3, scale_factor=8, mode="bilinear", align_corners=True)
        outputs = coarse_pred, s5_pred, s4_pred, s3_pred
        return (outputs, auxiliary) if return_aux else outputs
