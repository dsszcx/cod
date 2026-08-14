"""Network_SAM3.py - SAM3 region-prior guided camouflaged object detection network.

New input pipeline:
  x       (B,3,H,W)  normalized RGB image
  target  (B,1,H,W)  target region mask from SAM3
  aux     (B,3,H,W)  auxiliary region masks (zero-padded)

Combined 7-channel input: [RGB3, target1, aux3] -> PVTv2-B2 (in_chans=7).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
from Network_PVTv2 import RFB_modified, NeighborConnectionDecoder, ReverseStage


def load_pretrained_backbone(model, pretrained_path="./pretrained/pvt_v2_b2.pth"):
    """Load ImageNet pretrained weights into a 7-channel backbone.
    First 3 conv channels use pretrained weights, extra channels kaiming init."""
    if not os.path.exists(pretrained_path):
        print(f">>> WARNING: pretrained {pretrained_path} not found")
        return
    state = torch.load(pretrained_path, map_location="cpu")
    state.pop("head.weight", None)
    state.pop("head.bias", None)
    own = model.state_dict()
    n = 0
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v)
            n += 1
    if "patch_embed1.proj.weight" in state:
        w3 = state["patch_embed1.proj.weight"]
        with torch.no_grad():
            model.patch_embed1.proj.weight[:, :3].copy_(w3)
            nn.init.kaiming_normal_(model.patch_embed1.proj.weight[:, 3:],
                                    mode="fan_out", nonlinearity="relu")
    print(f">>> Backbone loaded {n} layers from {pretrained_path}")


class Network(nn.Module):
    """SAM3 region-prior-guided network. Input 7 channels."""

    def __init__(self, channel=64, imagenet_pretrained=True,
                 pretrained_path="./pretrained/pvt_v2_b2.pth"):
        super().__init__()
        self.backbone = pvt_v2_b2(pretrained=False, in_chans=7)
        if imagenet_pretrained:
            load_pretrained_backbone(self.backbone, pretrained_path)

        self.rfb2_1 = RFB_modified(128, channel)
        self.rfb3_1 = RFB_modified(320, channel)
        self.rfb4_1 = RFB_modified(512, channel)
        self.NCD = NeighborConnectionDecoder(channel)
        self.RS5 = ReverseStage(channel)
        self.RS4 = ReverseStage(channel)
        self.RS3 = ReverseStage(channel)

    def forward(self, x, target, aux):
        # ---- build 7-channel input ----
        inp = torch.cat([x, target, aux], dim=1)   # (B, 7, H, W)

        # ---- backbone ----
        endpoints = self.backbone.extract_endpoints(inp)
        x2 = endpoints["reduction_3"]
        x3 = endpoints["reduction_4"]
        x4 = endpoints["reduction_5"]

        # ---- RFB ----
        x2_rfb = self.rfb2_1(x2)
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)

        # ---- NCD ----
        S_g = self.NCD(x4_rfb, x3_rfb, x2_rfb)
        S_g_pred = F.interpolate(S_g, scale_factor=8, mode="bilinear", align_corners=True)

        # ---- reverse guidance ----
        guidance_g = F.interpolate(S_g, scale_factor=0.25, mode="bilinear", align_corners=True)
        ra4_feat = self.RS5(x4_rfb, guidance_g)
        S_5 = ra4_feat + guidance_g
        S_5_pred = F.interpolate(S_5, scale_factor=32, mode="bilinear", align_corners=True)

        guidance_5 = F.interpolate(S_5, scale_factor=2, mode="bilinear", align_corners=True)
        ra3_feat = self.RS4(x3_rfb, guidance_5)
        S_4 = ra3_feat + guidance_5
        S_4_pred = F.interpolate(S_4, scale_factor=16, mode="bilinear", align_corners=True)

        guidance_4 = F.interpolate(S_4, scale_factor=2, mode="bilinear", align_corners=True)
        ra2_feat = self.RS3(x2_rfb, guidance_4)
        S_3 = ra2_feat + guidance_4
        S_3_pred = F.interpolate(S_3, scale_factor=8, mode="bilinear", align_corners=True)

        return S_g_pred, S_5_pred, S_4_pred, S_3_pred


if __name__ == "__main__":
    net = Network(channel=64, imagenet_pretrained=True)
    net.eval()
    x = torch.randn(1, 3, 352, 352)
    t = torch.randn(1, 1, 352, 352)
    a = torch.randn(1, 3, 352, 352)
    with torch.no_grad():
        outs = net(x, t, a)
        for i, o in enumerate(outs):
            print(f"Output {i+1}: {o.shape}")
    print("Total params:", sum(p.numel() for p in net.parameters()))