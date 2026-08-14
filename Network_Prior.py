"""Network_Prior.py - PVTv2-B2 + LLM prior guided camouflaged object detection network.

New input pipeline:
  x      (B,3,H,W)  normalized RGB image
  pos_map (B,1,H,W) bbox gaussian heatmap from Qwen2.5-VL
  tag_bow (B,V)     bag-of-words semantic embedding from Qwen2.5-VL tags

Combined 5-channel input: [RGB, pos_map, sem_map] -> PVTv2-B2 (in_chans=5).
All decoders (RFB/NCD/ReverseStage) are reused from Network_PVTv2.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
from Network_PVTv2 import RFB_modified, NeighborConnectionDecoder, ReverseStage


def load_pretrained_backbone(model, pretrained_path="./pretrained/pvt_v2_b2.pth"):
    """Load ImageNet pretrained weights into a 5-channel backbone.
    First 3 conv channels use pretrained weights, extra channels are
    initialized with small random values (kaiming)."""
    if not os.path.exists(pretrained_path):
        print(f">>> WARNING: pretrained {pretrained_path} not found, train from scratch")
        return
    state = torch.load(pretrained_path, map_location="cpu")
    state.pop("head.weight", None)
    state.pop("head.bias", None)
    own = model.state_dict()
    n_loaded = 0
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v)
            n_loaded += 1
    # patch_embed1.proj: (64, 3, 7, 7) -> copy to first 3 channels of (64, 5, 7, 7)
    if "patch_embed1.proj.weight" in state:
        w3 = state["patch_embed1.proj.weight"]
        with torch.no_grad():
            model.patch_embed1.proj.weight[:, :3].copy_(w3)
            # kaiming init for extra 2 channels
            nn.init.kaiming_normal_(model.patch_embed1.proj.weight[:, 3:],
                                    mode="fan_out", nonlinearity="relu")
    print(f">>> Backbone: loaded {n_loaded} layers from {pretrained_path}")


class PriorBackbone(nn.Module):
    """PVTv2-B2 backbone with 5-channel input (RGB3 + pos1 + sem1)."""

    def __init__(self, in_chans=5, pretrained_path="./pretrained/pvt_v2_b2.pth",
                 pretrained=True):
        super().__init__()
        self.backbone = pvt_v2_b2(pretrained=False, in_chans=in_chans)
        if pretrained:
            load_pretrained_backbone(self.backbone, pretrained_path)

    def forward(self, x):
        return self.backbone.extract_endpoints(x)


class SemanticProjector(nn.Module):
    """Project bag-of-words tag embedding into a spatial semantic map (B,1,H,W)."""

    def __init__(self, vocab_size, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, tag_bow, hw):
        """tag_bow: (B, V); hw: (H, W) -> (B, 1, H, W)"""
        s = self.net(tag_bow)                     # (B, 1)
        s = s.view(-1, 1, 1, 1)
        return s.expand(-1, 1, hw[0], hw[1])


class Network(nn.Module):
    """LLM-prior-guided camouflaged object detection network.

    Inputs:
      x       (B,3,H,W) RGB image
      pos_map (B,1,H,W) gaussian heatmap prior
      tag_bow (B,V)     semantic bag-of-words
    Outputs:
      S_g_pred, S_5_pred, S_4_pred, S_3_pred  (same as original)
    """

    def __init__(self, channel=64, vocab_size=256, imagenet_pretrained=True,
                 pretrained_path="./pretrained/pvt_v2_b2.pth"):
        super().__init__()
        self.backbone = PriorBackbone(in_chans=5, pretrained_path=pretrained_path,
                                      pretrained=imagenet_pretrained)
        self.sem_proj = SemanticProjector(vocab_size)
        self.rfb2_1 = RFB_modified(128, channel)
        self.rfb3_1 = RFB_modified(320, channel)
        self.rfb4_1 = RFB_modified(512, channel)
        self.NCD = NeighborConnectionDecoder(channel)
        self.RS5 = ReverseStage(channel)
        self.RS4 = ReverseStage(channel)
        self.RS3 = ReverseStage(channel)

    def forward(self, x, pos_map, tag_bow):
        # ---- build 5-channel input ----
        sem_map = self.sem_proj(tag_bow, x.shape[-2:])   # (B,1,H,W)
        inp = torch.cat([x, pos_map, sem_map], dim=1)    # (B,5,H,W)

        # ---- backbone ----
        endpoints = self.backbone(inp)
        x2 = endpoints["reduction_3"]
        x3 = endpoints["reduction_4"]
        x4 = endpoints["reduction_5"]

        # ---- RFB + CBAM ----
        x2_rfb = self.rfb2_1(x2)
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)

        # ---- neighbor connection decoder ----
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
    net = Network(channel=64, vocab_size=64, imagenet_pretrained=True)
    net.eval()
    x = torch.randn(1, 3, 352, 352)
    pos = torch.randn(1, 1, 352, 352)
    bow = torch.randn(1, 64)
    with torch.no_grad():
        outs = net(x, pos, bow)
        for i, o in enumerate(outs):
            print(f"Output {i+1} shape: {o.shape}")