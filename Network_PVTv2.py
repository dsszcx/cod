import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2


class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        # 优化1：动态调整压缩比，增加最低限制
        self.reduction_ratio = max(8, reduction_ratio)  # 保证最低压缩比为8
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 优化2：添加BatchNorm和更合理的激活函数
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels, channels // self.reduction_ratio, 1, bias=False),
            nn.BatchNorm2d(channels // self.reduction_ratio),
            nn.GELU(),  # 替换ReLU为GELU
            nn.Conv2d(channels // self.reduction_ratio, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # 优化3：多尺度空间注意力（保持单卷积层结构）
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=5, padding=2, bias=False),  # 5x5卷积扩大感受野
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # 仅初始化顶层 Conv2d（避免覆盖 channel_attention/spatial_attention 内部已设计的权重）
        for m in [self.channel_attention, self.spatial_attention]:
            if isinstance(m[0], nn.Conv2d):
                nn.init.kaiming_normal_(m[0].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # 保持原有流程，增加残差连接
        identity = x
        # 通道注意力
        ca = self.avg_pool(x)
        ca = self.channel_attention(ca)
        x = x * ca
        # 空间注意力
        sa = self.spatial_attention(x)
        x = x * sa
        # 优化5：添加残差连接
        return x + identity
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.cbam = CBAM(out_channel)  # 新增CBAM
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        x = self.cbam(x)  # 添加CBAM
        return x


class NeighborConnectionDecoder(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)
        self.cbam = CBAM(3 * channel)  # 新增CBAM

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.cbam(x)  # 添加CBAM
        x = self.conv5(x)

        return x


class GRA(nn.Module):
    def __init__(self, channel, subchannel):
        super(GRA, self).__init__()
        self.group = channel // subchannel
        self.conv = nn.Sequential(
            nn.Conv2d(channel + self.group, channel, 3, padding=1), nn.ReLU(True),
        )
        self.score = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, x, y):
        if self.group == 1:
            x_cat = torch.cat((x, y), 1)
        else:
            xs = torch.chunk(x, self.group, dim=1)
            x_cat = torch.cat([v for t in zip(xs, [y] * self.group) for v in t], dim=1)

        x = x + self.conv(x_cat)
        y = y + self.score(x)

        return x, y


class ReverseStage(nn.Module):
    def __init__(self, channel):
        super(ReverseStage, self).__init__()
        self.weak_gra = GRA(channel, channel)
        self.medium_gra = GRA(channel, 8)
        self.strong_gra = GRA(channel, 1)

    def forward(self, x, y):
        y = -1 * (torch.sigmoid(y)) + 1
        x, y = self.weak_gra(x, y)
        x, y = self.medium_gra(x, y)
        _, y = self.strong_gra(x, y)
        return y


class Network(nn.Module):
    def __init__(self, channel=32, imagenet_pretrained=True):
        super(Network, self).__init__()
        self.backbone = pvt_v2_b2(pretrained=imagenet_pretrained)
        self.rfb2_1 = RFB_modified(128, channel)
        self.rfb3_1 = RFB_modified(320, channel)
        self.rfb4_1 = RFB_modified(512, channel)
        self.NCD = NeighborConnectionDecoder(channel)
        self.RS5 = ReverseStage(channel)
        self.RS4 = ReverseStage(channel)
        self.RS3 = ReverseStage(channel)

    def forward(self, x):
        endpoints = self.backbone.extract_endpoints(x)
        x2 = endpoints['reduction_3']
        x3 = endpoints['reduction_4']
        x4 = endpoints['reduction_5']

        x2_rfb = self.rfb2_1(x2)
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)

        S_g = self.NCD(x4_rfb, x3_rfb, x2_rfb)
        S_g_pred = F.interpolate(S_g, scale_factor=8, mode='bilinear', align_corners=True)

        guidance_g = F.interpolate(S_g, scale_factor=0.25, mode='bilinear', align_corners=True)
        ra4_feat = self.RS5(x4_rfb, guidance_g)
        S_5 = ra4_feat + guidance_g
        S_5_pred = F.interpolate(S_5, scale_factor=32, mode='bilinear', align_corners=True)

        guidance_5 = F.interpolate(S_5, scale_factor=2, mode='bilinear', align_corners=True)
        ra3_feat = self.RS4(x3_rfb, guidance_5)
        S_4 = ra3_feat + guidance_5
        S_4_pred = F.interpolate(S_4, scale_factor=16, mode='bilinear', align_corners=True)

        guidance_4 = F.interpolate(S_4, scale_factor=2, mode='bilinear', align_corners=True)
        ra2_feat = self.RS3(x2_rfb, guidance_4)
        S_3 = ra2_feat + guidance_4
        S_3_pred = F.interpolate(S_3, scale_factor=8, mode='bilinear', align_corners=True)

        return S_g_pred, S_5_pred, S_4_pred, S_3_pred


if __name__ == '__main__':
    net = Network(imagenet_pretrained=False)
    net.eval()
    dump_x = torch.randn(1, 3, 352, 352)
    with torch.no_grad():
        outputs = net(dump_x)
        for i, out in enumerate(outputs):
            print(f"Output {i+1} shape: {out.shape}")
