# 🔍 Camouflaged Object Detection — Improved SINet-V2

<div align="center">

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Model](https://img.shields.io/badge/Backbone-PVTv2--B2-orange)]()
[![Datasets](https://img.shields.io/badge/Datasets-4-lightgrey)]()

**基于 PVTv2 + 改进 CBAM 注意力机制的伪装目标检测网络**

*SINet-V2 增强版 · Transformer 主干 · 多尺度特征融合 · 四数据集评估*

[📖 论文参考](#-论文参考) · [🚀 快速开始](#-快速开始) · [📊 实验结果](#-实验结果) · [🧠 模型架构](#-模型架构)

</div>

---

## 📌 项目简介

本项目在 **[SINet-V2](https://github.com/GewelsJI/SINet-V2)**（CVPR 2021）的基础上进行了四方面改进，用于**伪装目标检测（Camouflaged Object Detection, COD）**——从复杂背景中精准分割出与背景高度融合的伪装目标。

### ✨ 核心改进

| 改进项 | 原始 SINet-V2 | 本模型 | 效果 |
|:---|:---|:---|:---|
| **主干网络** | Res2Net-50 (CNN) | **PVTv2-B2** (Transformer) | 增强全局上下文建模与多尺度特征提取 |
| **注意力机制** | 无 | **改进 CBAM**（通道+空间注意力+残差连接） | 聚焦关键区域，抑制背景噪声，加速收敛 |
| **RFB 模块** | 标准 RFB | **RFB + CBAM 嵌入** | 纹理增强后立即注意力筛选 |
| **损失函数** | BCE + IoU | **Hybrid E-loss**（加权 BCE + E-loss + 加权 IoU） | 更稳定的训练，更精细的边缘分割 |
| **数据增强** | 基础增强 | **多策略增强**（翻转/裁剪/旋转/色彩/噪声） | 提升泛化能力 |
| **GRA 模块** | 静态分组 | **动态分组策略** | 计算效率更高，代码可维护性更好 |

---

## 🧠 模型架构

```
输入图像 (3×352×352)
       │
       ▼
┌──────────────────┐
│   PVTv2-B2       │  ← Transformer 主干（替换 Res2Net-50）
│   Backbone       │
└──┬───────┬───────┘
   │       │       │
   ▼       ▼       ▼
 x2_rfb  x3_rfb  x4_rfb    ← RFB_modified + CBAM（纹理增强 + 注意力）
   │       │       │
   └───────┼───────┘
           ▼
   ┌───────────────┐
   │  Neighbor      │       ← 邻接解码器 + CBAM
   │  Connection    │
   │  Decoder       │
   └───────┬───────┘
           ▼
   ┌───────────────┐
   │  Reverse       │       ← 反向引导（GRA × 3 级）
   │  Guidance      │
   │  (RS5/RS4/RS3) │
   └───────┬───────┘
           ▼
    多级输出 (S_g, S_5, S_4, S_3)
```

> 📐 从 `网络结构图.py` 可生成完整网络结构可视化

---

## 📊 实验结果

在四个标准 COD 数据集上评估（全部指标使用 `py_sod_metrics` 计算）：

| 数据集 | 样本数 | 特点 |
|:---|:---|:---|
| **COD10K** | 10,000 | 最大规模 COD 数据集，100 类别 |
| **CAMO** | 2,500 | 8 类别，物体大小变化大 |
| **CHAMELEON** | 76 | 高质量标注，自然场景 |
| **NC4K** | 4,121 | 大规模测试集 |

### 消融实验

| 配置 | Backbone | CBAM | 说明 |
|:---|:---|:---|:---|
| Baseline | Res2Net-50 | ✗ | 原始 SINet-V2 |
| +PVTv2 | PVTv2-B2 | ✗ | 仅替换主干 |
| +CBAM | PVTv2-B2 | ✓ | **本模型（完整版）** |

> 📈 详细指标数值见下方 [性能评估](#-性能评估) 章节

---

## 🚀 快速开始

### 环境要求

- Python ≥ 3.8
- PyTorch ≥ 1.10
- CUDA ≥ 11.3（GPU 训练推荐）
- 单张 GPU 显存 ≥ 8GB（batch_size=8, input=352²）

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/jdytwagn/camouflaged-object-detection-improve.git
cd camouflaged-object-detection-improve

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载预训练权重（二选一）
# 方式 A：百度网盘
# 链接: https://pan.baidu.com/s/1bqAies8SlG_hGT8b4WTbqg?pwd=ke2e 提取码: ke2e
# 方式 B：官方下载（推荐）
wget https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth -P lib/

# 4. 下载数据集
# 百度网盘: https://pan.baidu.com/s/159VrPMukt1Zii8ckx4rAuA?pwd=kduw 提取码: kduw
# 解压到 ./Dataset/ 目录
```

### 目录结构

```
.
├── Network_PVTv2.py    # 模型定义（CBAM + RFB + NCD + GRA）
├── MyTrain.py           # 训练脚本
├── MyTest.py            # 测试/推理脚本
├── dataloader.py        # 数据加载与增强
├── utils.py             # 工具函数（梯度裁剪、参数/FLOPs 统计）
├── 新metric.py           # 性能指标评估（MAE/F-measure/E-measure/S-measure/W-Fmeasure）
├── GUI.py               # 图形界面（图片/视频/摄像头实时检测）
├── 网络结构图.py          # 网络可视化生成
├── lib/
│   └── pvtv2.py         # PVTv2 预训练模型加载
├── Dataset/
│   ├── TrainValDataset/ # 训练集（Imgs/ + GT/）
│   └── TestDataset/     # 测试集（COD10K/CAMO/CHAMELEON/NC4K）
└── model_pth/           # 模型保存目录
```

### 训练

```bash
# 默认参数训练（PVTv2-B2, AdamW, lr=1e-5, batch=8, epoch=50, size=352）
python MyTrain.py

# 自定义参数
python MyTrain.py \
    --model PVTv2-B2 \
    --epoch 100 \
    --lr 1e-4 \
    --batchsize 16 \
    --trainsize 384 \
    --gpu_id 0 \
    --train_root ./Dataset/TrainValDataset/ \
    --test_root ./Dataset/TestDataset/CAMO/ \
    --save_root ./model_pth/OUR/
```

### 测试（生成预测图）

```bash
# 修改 MyTest.py 中的 --pth_path 为你的权重路径
python MyTest.py --pth_path ./model_pth/OUR/KKK_best_47.pth --gpu_id 0
```

### 性能评估

```bash
# 修改 新metric.py 中的 data_root 为你的预测结果路径
python 新metric.py
```

评估指标包括：
- **MAE**（Mean Absolute Error）— 越小越好
- **S-measure**（Sα）— 结构相似度
- **E-measure**（Eφ）— 增强对齐度量
- **F-measure**（Fβ）— 综合查准率与查全率
- **W-Fmeasure**（Fβ^w）— 加权 F-measure

### GUI 演示

```bash
python GUI.py
```

支持功能：
- 🔐 用户注册/登录
- 🖼️ 单张图片检测
- 🎬 视频文件检测
- 📹 摄像头实时检测
- 💾 检测结果保存

---

## 🔬 论文参考

本项目的基线模型参考以下论文：

> **SINet-V2** — *Concealed Object Detection*  
> Fan, D.P., Ji, G.P., Cheng, M.M., Shao, L.  
> IEEE TPAMI, 2021. [[Paper]](https://arxiv.org/abs/2109.09043) [[Code]](https://github.com/GewelsJI/SINet-V2)

> **PVTv2** — *Pyramid Vision Transformer V2*  
> Wang, W., Xie, E., Li, X., et al.  
> CVMJ, 2022. [[Paper]](https://arxiv.org/abs/2106.13797) [[Code]](https://github.com/whai362/PVT)

> **CBAM** — *Convolutional Block Attention Module*  
> Woo, S., Park, J., Lee, J.Y., Kweon, I.S.  
> ECCV, 2018. [[Paper]](https://arxiv.org/abs/1807.06521)

---

## 📦 预训练模型

| 模型 | 百度网盘 | 提取码 |
|:---|:---|:---|
| PVTv2-B2 预训练权重 | [下载](https://pan.baidu.com/s/1bqAies8SlG_hGT8b4WTbqg) | `ke2e` |
| 训练好的完整模型 | [下载](https://pan.baidu.com/s/1nz_E-tiGLIE8u1NN3rixJA) | `mgc1` |
| 四数据集（COD10K/CAMO/CHAMELEON/NC4K） | [下载](https://pan.baidu.com/s/159VrPMukt1Zii8ckx4rAuA) | `kduw` |

---

## 📂 文件说明

| 文件 | 功能 |
|:---|:---|
| `Network_PVTv2.py` | 完整模型定义：CBAM、RFB_modified、NeighborConnectionDecoder、GRA、ReverseStage、Network |
| `MyTrain.py` | 训练主程序：数据加载→多尺度训练→Hybrid E-loss→验证→模型保存 |
| `MyTest.py` | 推理脚本：加载模型→生成 COD10K/CAMO/NC4K/CHAMELEON 四数据集预测图 |
| `dataloader.py` | 数据加载器：随机翻转/裁剪/旋转/颜色增强/噪声 → ImageNet 归一化 |
| `utils.py` | 工具集：梯度裁剪、学习率衰减、训练指标滑动平均、参数/FLOPs 计算 |
| `新metric.py` | 性能评估：MAE/S-measure/E-measure/F-measure/W-Fmeasure + 可视化 |
| `GUI.py` | Tkinter 图形界面：用户系统 + 图片/视频/摄像头检测 + 结果保存 |
| `网络结构图.py` | 网络结构可视化工具 |
| `lib/pvtv2.py` | PVTv2 实现（含官方预训练权重 URL） |

---

## 🔧 引用

如果本项目对你的研究或工作有帮助，请考虑引用：

```bibtex
@misc{camouflaged-detection-improved,
  author       = {jdytwagn},
  title        = {Camouflaged Object Detection — Improved SINet-V2 with PVTv2 and CBAM},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/jdytwagn/camouflaged-object-detection-improve}}
}
```

同时也请引用原始 SINet-V2 和 PVTv2 论文。

---

## 📄 License

MIT License © 2026

---

<div align="center">
  <sub>Built with ❤️ by jdytwagn · Powered by PyTorch & PVTv2</sub>
</div>
