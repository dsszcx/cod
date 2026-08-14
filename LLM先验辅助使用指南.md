# LLM 辅助伪装目标检测 — 使用指南

本项目在 SINet-V2 改进版基础上，引入 **Qwen2.5-VL-3B-Instruct** 生成语义标签和位置先验，
以 [原图 + 位置先验图 + 语义嵌入] 作为网络输入，实现先验引导的伪装目标分割。

---

## 一、整体流程

```
Step 1 (服务器): Qwen2.5-VL-3B 离线批量标注
   训练/测试图片 -> llm_annotator.py -> prior_annotations.json

Step 2 (本地): 先验转导图
   prior_annotations.json -> prior_encoder.py -> prior/*.npz

Step 3 (本地): 训练
   MyTrain_Prior.py -> model_pth/PRIOR/*.pth

Step 4 (本地): 推理
   MyTest_Prior.py -> results_prior/CAMO/*.png
```

---

## 二、环境准备

### 服务器（跑 Qwen2.5-VL-3B）
```bash
pip install transformers>=4.47 qwen-vl-utils accelerate torch
# 下载模型（也可自动从 HuggingFace 拉取）
# 建议先下载到本地：
#   huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./Qwen2.5-VL-3B
```

### 本地（训练/推理）
```bash
pip install torch torchvision timm opencv-python numpy pillow tqdm
# 已有: pretrained/pvt_v2_b2.pth (PVTv2-B2 骨干预训练权重)
```

---

## 三、使用步骤

### Step 1: 服务器上批量生成先验标注

```bash
# 训练集（1000张）
python llm_annotator.py \
    --data_root ./Dataset/TrainValDataset/Imgs/ \
    --out prior_annotations.json --split train

# 测试集（250张）
python llm_annotator.py \
    --data_root ./Dataset/TestDataset/CAMO/Imgs/ \
    --out prior_annotations.json --split test
```

输出格式（每张图）:
```json
{
  "train": {
    "camourflage_00001.jpg": {
      "tags": ["白鸟", "翅膀"],
      "bbox": [52, 30, 288, 215],
      "confidence": 0.87
    }
  },
  "test": { ... }
}
```

支持断点续跑（--no_resume 可关闭），每 20 张自动保存检查点。

### Step 2: 本地生成先验张量

```bash
python prior_encoder.py --ann prior_annotations.json --out prior/ --size 352
```

生成内容:
- `prior/vocab.json` — 标签词表（词袋编码用）
- `prior/train_xxx.npz` — 每张训练图的 (pos_map, tag_bow, region, confidence)
- `prior/test_xxx.npz` — 每张测试图的先验

调试工具（可视化先验叠加图）:
```bash
python -c "from prior_encoder import visualize_prior; visualize_prior('test.jpg', 'prior/test_xxx.npz')"
```

### Step 3: 训练

```bash
python MyTrain_Prior.py \
    --epoch 50 --batchsize 8 --trainsize 352 \
    --prior_root ./prior/ \
    --vocab_size <与 vocab.json 长度一致> \
    --save_root ./model_pth/PRIOR/
```

模型保存: `model_pth/PRIOR/KKK_best_*.pth`（MAE 最低）

### Step 4: 推理

```bash
python MyTest_Prior.py \
    --pth_path ./model_pth/PRIOR/KKK_best_2.pth \
    --prior_root ./prior/ --vocab_size 256
```

输出: `results_prior/PRIOR/CAMO/*.png`

---

## 四、网络改动说明（Network_Prior.py）

输入从 3 通道变为 5 通道:

```
通道1-3: 原图 RGB (ImageNet 归一化)
通道4:   位置先验图 (bbox 高斯热力图, 1x352x352)
通道5:   语义嵌入图 (tag_bow 经 SemanticProjector 投影, 全图常量)
```

关键实现:
1. `PriorBackbone` — PVTv2-B2 以 in_chans=5 实例化；前 3 通道加载 ImageNet 预训练权重，
   额外 2 通道 kaiming 初始化。
2. `SemanticProjector` — tag_bow (V,) -> 单标量 -> 广播为空间图。
3. 解码器（RFB/NCD/ReverseStage）完全复用原结构。

---

## 五、文件清单

| 文件 | 说明 |
|:---|:---|
| `llm_annotator.py` | Qwen2.5-VL-3B 封装 + 离线批量标注 |
| `prior_encoder.py` | bbox->高斯热力图 / tags->词袋 / 组装 npz |
| `Network_Prior.py` | 5 通道输入网络 + 先验注入 |
| `dataloader_prior.py` | 训练数据加载（含先验对齐增强） |
| `MyTrain_Prior.py` | 先验引导训练脚本 |
| `MyTest_Prior.py` | 先验引导推理脚本 |
| `prior_annotations.json` | LLM 标注结果（待生成） |
| `prior/*.npz` | 先验张量（待生成） |

---

## 六、消融实验建议

| 配置 | 说明 |
|:---|:---|
| Baseline | 原始 3 通道 (Network_PVTv2.py) |
| +位置先验 | 只加 pos_map (通道4=pos, 通道5=0) |
| +语义先验 | 只加 tag_bow (通道4=0, 通道5=sem) |
| **完整版** | 位置+语义双先验 (默认) |

通过 `SemanticProjector` 或 dataloader 传零图即可实现消融。