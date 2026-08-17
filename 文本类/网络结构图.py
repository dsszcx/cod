# verbose_structure_fixed.py
import torch
import torch.nn.functional as F
from Network_PVTv2 import Network  # 根据实际路径调整

def verbose_flow(net, x):
    endpoints = net.backbone.extract_endpoints(x)
    print("\n=== 1. Backbone 特征提取 ===")
    # 定义 reduction_i 对应的 Stage 名称映射
    stage_map = {
        'reduction_1': 'Patch Embedding',
        'reduction_2': 'Transformer Stage-1',
        'reduction_3': 'Transformer Stage-2',
        'reduction_4': 'Transformer Stage-3',
        'reduction_5': 'Transformer Stage-4',
    }
    # 按顺序打印实际存在的 endpoints
    for idx in range(1, 6):
        key = f'reduction_{idx}'
        if key in endpoints:
            feat = endpoints[key]
            print(f" {stage_map[key]:18s} ──▶ {key:12s} shape {tuple(feat.shape)}")
    print()

    # 只取 reduction_2~5 里实际存在的三/四个用于后续 RFB 模块
    s2 = endpoints.get('reduction_2', None)
    s3 = endpoints.get('reduction_3', None)
    s4 = endpoints.get('reduction_4', None)
    s5 = endpoints.get('reduction_5', None)

    # 根据你的 Network 定义，rfb2_1 对应 reduction_3，rfb3_1→reduction_4，rfb4_1→reduction_5
    print("=== 2. RFB_modified (多尺度+CBAM) 处理 ===")
    if s3 is not None:
        x2_r = net.rfb2_1(s3)
        print(f" reduction_3 ({tuple(s3.shape)}) ──▶ rfb2_1 ──▶ x2_rfb: {tuple(x2_r.shape)}")
    if s4 is not None:
        x3_r = net.rfb3_1(s4)
        print(f" reduction_4 ({tuple(s4.shape)}) ──▶ rfb3_1 ──▶ x3_rfb: {tuple(x3_r.shape)}")
    if s5 is not None:
        x4_r = net.rfb4_1(s5)
        print(f" reduction_5 ({tuple(s5.shape)}) ──▶ rfb4_1 ──▶ x4_rfb: {tuple(x4_r.shape)}")
    print()

    # Decoder
    print("=== 3. NeighborConnectionDecoder (融合多尺度特征) ===")
    S_g = net.NCD(x4_r, x3_r, x2_r)
    print(f" inputs: x4_rfb, x3_rfb, x2_rfb ──▶ NCD ──▶ S_g: {tuple(S_g.shape)}")
    S_g_pred = F.interpolate(S_g, scale_factor=8, mode='bilinear', align_corners=True)
    print(f" S_g ──▶ up×8 ──▶ S_g_pred (全局预测): {tuple(S_g_pred.shape)}\n")

    # ReverseStages
    print("=== 4. ReverseStage (反向多级引导) ===")
    # Level-5
    guidance_g = F.interpolate(S_g, scale_factor=0.25, mode='bilinear', align_corners=True)
    print(f" S_g ──▶ down×4 ──▶ guidance_g: {tuple(guidance_g.shape)}")
    y5 = net.RS5(x4_r, guidance_g)
    S5 = y5 + guidance_g
    S5_pred = F.interpolate(S5, scale_factor=32, mode='bilinear', align_corners=True)
    print(f" RS5: x4_rfb+guidance_g ──▶ y5: {tuple(y5.shape)} ──▶ +res → S_5 ──▶ up×32 → S_5_pred: {tuple(S5_pred.shape)}\n")

    # Level-4
    guidance_5 = F.interpolate(S5, scale_factor=2, mode='bilinear', align_corners=True)
    print(f" S_5 ──▶ up×2 ──▶ guidance_5: {tuple(guidance_5.shape)}")
    y4 = net.RS4(x3_r, guidance_5)
    S4 = y4 + guidance_5
    S4_pred = F.interpolate(S4, scale_factor=16, mode='bilinear', align_corners=True)
    print(f" RS4: x3_rfb+guidance_5 ──▶ y4: {tuple(y4.shape)} ──▶ +res → S_4 ──▶ up×16 → S_4_pred: {tuple(S4_pred.shape)}\n")

    # Level-3
    guidance_4 = F.interpolate(S4, scale_factor=2, mode='bilinear', align_corners=True)
    print(f" S_4 ──▶ up×2 ──▶ guidance_4: {tuple(guidance_4.shape)}")
    y3 = net.RS3(x2_r, guidance_4)
    S3 = y3 + guidance_4
    S3_pred = F.interpolate(S3, scale_factor=8, mode='bilinear', align_corners=True)
    print(f" RS3: x2_rfb+guidance_4 ──▶ y3: {tuple(y3.shape)} ──▶ +res → S_3 ──▶ up×8 → S_3_pred: {tuple(S3_pred.shape)}\n")

    # Final Outputs
    print("=== 5. 最终输出 特征预测图 ===")
    print(f" [0] S_g_pred : {tuple(S_g_pred.shape)}")
    print(f" [1] S_5_pred : {tuple(S5_pred.shape)}")
    print(f" [2] S_4_pred : {tuple(S4_pred.shape)}")
    print(f" [3] S_3_pred : {tuple(S3_pred.shape)}\n")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Network(channel=64, imagenet_pretrained=False).to(device).eval()
    x = torch.randn(1, 3, 352, 352, device=device)
    verbose_flow(net, x)
