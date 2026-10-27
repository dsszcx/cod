"""Network_SAM3.py - SAM3 region-prior guided camouflaged object detection network.

New input pipeline:
  x       (B,3,H,W)  normalized RGB image
  target  (B,1,H,W)  target region mask from SAM3
  aux     (B,3,H,W)  auxiliary region masks (zero-padded)

Combined 8-channel input: [enhanced_RGB3, target1, aux3, neg_flag1] -> PVTv2-B2 (in_chans=8).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
from Network_PVTv2 import RFB_modified, NeighborConnectionDecoder, ReverseStage, BasicConv2d


class PriorFusion(nn.Module):
    """显式融合先验特征(target + aux)与RGB特征。

    通过注意力机制让先验信息增强RGB特征，而非简单concat。
    target_mask: (B,1,H,W) 目标区域
    aux_masks: (B,3,H,W) 辅助区域
    → 生成(B,3,H,W)的注意力图，增强RGB的对应区域。
    """

    def __init__(self):
        super().__init__()
        # 先验特征压缩：target(1) + aux(3) → 16 → 3通道注意力
        self.prior_encoder = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        # 生成RGB注意力权重
        self.prior_to_attn = nn.Sequential(
            nn.Conv2d(16, 3, 1, bias=False),
            nn.Sigmoid()
        )
        # 可学习的融合强度（初始值0，即开始时几乎不起作用，由网络自行学习）
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, rgb, target, aux):
        """
        rgb:    (B,3,H,W) 归一化RGB
        target: (B,1,H,W) 目标区域mask
        aux:    (B,3,H,W) 辅助区域masks
        return: (B,3,H,W) 增强后的RGB特征
        """
        prior = torch.cat([target, aux], dim=1)  # (B,4,H,W)
        prior_feat = self.prior_encoder(prior)     # (B,16,H,W)
        attn = self.prior_to_attn(prior_feat)      # (B,3,H,W) 0~1
        # 残差增强：enhanced = rgb * (1 + scale * attn)
        enhanced = rgb * (1.0 + self.scale * attn)
        return enhanced


def load_pretrained_backbone(model, pretrained_path="./pretrained/pvt_v2_b2.pth",
                             strict=True):
    """Load ImageNet pretrained weights into an 8-channel backbone.

    The first 3 input channels take the pretrained RGB filters; channels 3..7 are
    kaiming-initialised.  Returns the number of copied parameter tensors.

    Layer-0 guard (2026-10): a missing or key-mismatched file used to print one
    warning line and continue, so a run could train from random weights while the
    log still said "backbone: pvt_v2_b2" -- and the difference between two such
    runs is bigger than any ablation.  ``strict=True`` raises instead, and the
    RGB copy is verified afterwards.
    """
    if not os.path.exists(pretrained_path):
        msg = "pretrained backbone not found: %s" % pretrained_path
        if strict:
            raise FileNotFoundError(
                msg + " -- pass imagenet_pretrained=False if you really want a "
                      "from-scratch backbone (state it in the report)")
        print(">>> WARNING: " + msg + " -> backbone is RANDOMLY initialised")
        return 0
    state = torch.load(pretrained_path, map_location="cpu")
    if not isinstance(state, dict):
        state = getattr(state, "state_dict", lambda: {})()
    # mmseg / timm style wrappers
    for wrapper in ("model", "state_dict", "backbone"):
        if isinstance(state.get(wrapper), dict) and "patch_embed1.proj.weight" not in state:
            state = state[wrapper]
            break
    state.pop("head.weight", None)
    state.pop("head.bias", None)
    own = model.state_dict()
    n = 0
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v)
            n += 1
    if "patch_embed1.proj.weight" not in state:
        raise RuntimeError(
            "%s has no patch_embed1.proj.weight -> it is not a PVTv2-B2 "
            "checkpoint (loaded %d other layers, which is worse than useless)"
            % (pretrained_path, n))
    w3 = state["patch_embed1.proj.weight"]
    with torch.no_grad():
        model.patch_embed1.proj.weight[:, :3].copy_(w3)
        nn.init.kaiming_normal_(model.patch_embed1.proj.weight[:, 3:],
                                mode="fan_out", nonlinearity="relu")
    # the copy is the entire point of this function -> prove it happened
    if not torch.equal(model.patch_embed1.proj.weight[:, :3].cpu(), w3.float()):
        raise RuntimeError("patch_embed1 RGB copy failed (%s)" % pretrained_path)
    print(f">>> Backbone loaded {n} layers from {pretrained_path} "
          f"(first conv: 3/8 channels ImageNet, 5/8 kaiming)")
    return n


class Network(nn.Module):
    """SAM3 region-prior-guided network. Input 8 channels: RGB(3) + target(1) + aux(3) + negative_flag(1)."""

    def __init__(self, channel=64, imagenet_pretrained=True,
                 pretrained_path="./pretrained/pvt_v2_b2.pth", pretrained_strict=True):
        super().__init__()
        # 先验融合模块：在输入骨干前显式融合target+aux与RGB
        self.prior_fusion = PriorFusion()
        self.backbone = pvt_v2_b2(pretrained=False, in_chans=8)
        if imagenet_pretrained:
            load_pretrained_backbone(self.backbone, pretrained_path,
                                     strict=pretrained_strict)

        self.rfb2_1 = RFB_modified(128, channel)
        self.rfb3_1 = RFB_modified(320, channel)
        self.rfb4_1 = RFB_modified(512, channel)
        self.NCD = NeighborConnectionDecoder(channel)
        self.RS5 = ReverseStage(channel)
        self.RS4 = ReverseStage(channel)
        self.RS3 = ReverseStage(channel)

    def forward(self, x, target, aux, neg_flag):
        # ---- 先验融合：增强RGB特征 ----
        enhanced_rgb = self.prior_fusion(x, target, aux)  # (B,3,H,W)
        # ---- build 8-channel input: enhanced_RGB(3) + target(1) + aux(3) + neg_flag(1) ----
        inp = torch.cat([enhanced_rgb, target, aux, neg_flag], dim=1)   # (B, 8, H, W)

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
    f = torch.zeros(1, 1, 352, 352)  # neg_flag
    with torch.no_grad():
        outs = net(x, t, a, f)
        for i, o in enumerate(outs):
            print(f"Output {i+1}: {o.shape}")
    print("Total params:", sum(p.numel() for p in net.parameters()))
