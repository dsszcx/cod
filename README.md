# SAM3 先验引导的伪装目标检测

当前研究范围更新（2026-10-26）：后续训练、验证与测试仅保留CAM图像，排除NonCAM；方法主线为“多语义—多区域—主辅关系—COD”。参见[当前CAM-only数据范围与入口](docs/project/CAM_only数据范围调整_20261026.md)。下面的8通道方法、混合数据指标与旧运行配置保留为历史记录，不能作为新方案的效果证据。最新CAM-only短训练和限制见下方“当前方案”。

快速了解现行方法与实验入口：[当前方案](docs/project/当前方案_CAM_only.md)。

2026-10-27进展：正式RGB基线80轮训练、最佳权重逐图复核和标准四指标测试已完成；全量语义区域先验已启动生成，前8张仅通过工程验收，尚未完成正式关系训练，也未证明主线方法的性能收益。最新结果见[RGB验收](docs/experiments/RGB基线验收与标准测试_20261027.md)，运行边界见[全量先验生成](docs/experiments/全量语义区域先验生成_20261027.md)。GitHub发布副本使用示例路径，部署前按[发布说明](docs/deployment/GitHub发布说明_20261027.md)配置本机资源。

用 **SAM3 + Qwen2.5-VL** 自动生成「目标区域 + 辅助区域」先验，作为额外通道喂给
**PVTv2-B2** 分割网络，在伪装目标检测（COD）上做像素级分割。

| 数据集 | 指标 | 值 |
|:---|:---|:---:|
| CAMO + COD10K 全量（约 8,275 张，**含训练图**） | **MAE ↓** | **0.0697** |

> **口径**：0.0697 是 CAMO 与 COD10K 的**全部有 GT 图**（训练 CAM 子集 4,040 +
> 测试集 4,250，名义合计 8,290；实际参与统计 **8,275** 张，个别图缺先验/GT）
> 合并算出的**一个总 mean MAE**。分母里有模型训练时见过的图，所以**偏乐观**，
> 不能当作纯测试集的泛化精度。
>
> 对照：无先验基线 **0.0632** 是 **CAMO 250 张纯测试集**口径 —— 两者**不可直接比较**。

> 本仓库是**源码 + 文档 + 测试**分享包：不含数据集、模型权重、大型运行产物或外部完整源码检出。
> 权重与数据按下面「快速开始」自行准备。

---

## 一、先说实话（重要）

1. **0.0697 达到了项目设定的「MAE ≤ 0.07」目标线**，但**口径必须一起讲清楚**：
   它是 **CAMO + COD10K 全量约 8,275 张（训练 CAM 子集 4,040 + 测试集 4,250，
   含训练图）合并算出的一个总 MAE**，不是纯测试集精度 —— 分母里有模型训练时
   见过的图，所以**偏乐观**，不能当作泛化指标对外宣传。
   同脚本下的**无先验基线 0.0632 是「CAMO 250 张纯测试集」口径**，两者
   **不可直接比较**，因此「先验是否带来提升」目前**没有结论**；要回答它，需在
   同一 4,250 张测试集上分别跑 `--prior_mode real` 与 `zero` 两组再比。
   Sm / Fβw 等指标同样需要用该权重重算，本 README 不填未验证的数。
2. 早期文档里的「MAE 0.0673，相对 0.18 提升 63%」**已作废**：那个 0.18 是把
   **未归一化的 GT（0/255）**直接送进 MAE 算出来的「前景像素占比 × 255」，
   与预测好坏无关。详见 `docs/experiments/Layer0证据报告.md`。
3. 那次「SAM3 先验」实验的 `prior_root` 指向一个**不存在的目录**，
   `dataloader_sam3` 对缺失 npz 静默返回全 0 先验 —— 等于训练了一条死分支。
   现在 `check_prior_coverage()` 会在 `--prior_mode real` 且覆盖率为 0 时直接报错。

结论：**这套代码的价值在「可复现、可诊断的完整流水线」**，
精度上还有一段要走（见「已知限制」）。

---

## 二、方法

### 2.1 三段式先验生成（服务器端，大模型）

```
原图 → Step1  Qwen2.5-VL 预筛标注（OpenCV 显著性候选框 → Qwen 命名）→ prior_annotations.json
     → Step2  SAM3 区域生成（box / text / hybrid prompt）        → sam3_regions/*.npz
     → Step3  区域选择（heuristic / Qwen）+ 质量闸门              → prior_sam3_c448/*.npz
                                                                    （448×448 uint8 0/1）
```

先验被压缩成 **448×448 二值 npz**（`savez_compressed`）：8275 个 npz 从 **78 GB** 降到
**0.3 GB**，下游代码零改动（`_load_prior` 本来就接受 uint8 0/1）。

### 2.2 分割网络（`Network_SAM3.py`）

```
8 通道输入 [RGB(3) | target(1) | aux(3) | neg_flag(1)]
        │
        ├─ PriorFusion：4→16→3 通道注意力，rgb * (1 + scale * attn)，scale 初始为 0
        │                （先验一开始几乎不起作用，由训练自己决定用多少）
        │
        └─ PVTv2-B2 (in_chans=8)
              ├─ stage2 → RFB_modified ─┐
              ├─ stage3 → RFB_modified ─┤→ NeighborConnectionDecoder
              └─ stage4 → RFB_modified ─┘
                                          ↓
                              ReverseStage ×3（RS5/RS4/RS3）
                                          ↓
                       4 路输出：edge / c2 / c3 / c4（深监督）
```

- 损失：**Hybrid E-loss**（加权 BCE + 增强对齐项 + 加权 IoU），
  并乘上 **Sobel 边缘权重**（边界像素权重 10、非边界 1），逼出更锐利的边界；
- 深监督权重：`0.4·edge + 0.25·c2 + 0.2·c3 + 0.15·c4`；
- 训练时每步跑 **2 个尺度**（`--size_rates 0.75,1`），验证/推理用 `c4`；
- 第 8 通道 `neg_flag`：COD10K 的 `NonCAM` 图（2960 张）按设计**没有目标**，
  GT 全黑、`neg_flag=1`，规则统一在 `dataloader_sam3.is_noncam()`，
  训练 / 推理 / Web 服务共用同一份判断。


---

## 三、目录结构（分享包内容）

```
.
├── MyTrain*.py / MyTest*.py   训练与推理入口（保留根目录，兼容原命令）
├── Network*.py                3/8 通道网络定义
├── dataloader*.py / utils.py  数据加载与公共函数
├── llm_annotator.py           Step1：Qwen 标注
├── sam3_regions.py            Step2：SAM3 候选区域
├── region_selector.py         Step3：区域选择与质量闸门
├── run_server_pipeline.py     Step1–5 一键流水线
├── configs/                   冻结实验配置
├── tools/                     E0–E4、压缩与测速工具
├── tests/                     全部免 GPU/回归测试
├── docs/
│   ├── project/               当前项目说明、进度、要求与交付文档
│   ├── experiments/           E0–E4、Layer-0、SAM3 实验记录
│   ├── deployment/            部署、分享、权重和未推送文件说明
│   ├── references/            论文、研究计划和结构图
│   └── archive/               旧版 README、问题日志和历史交接快照
├── releases/                  发布包及 sha256 校验文件
├── data_archives/             数据集压缩副本，不参与直接训练
├── Dataset/ / COD10K-v2/      训练数据与官方原始数据（保持原路径）
├── model_pth/ / pretrained/   模型权重（保持原路径）
└── outputs/ / results*/       实验输出（可重新生成）
```

完整的目录职责、入口命令和新增文件放置规则见 `PROJECT_STRUCTURE.md`。

---

## 四、快速开始

### 4.1 环境

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # 训练 / 推理
pip install -r requirements-server.txt                 # 服务器（Step1/2 需要 Qwen + SAM3）
```

Python 3.10、PyTorch ≥ 2.1（按服务器 CUDA 版本自行安装）。

### 4.2 准备「包外」的三样东西

| 需要什么 | 放到哪里 | 说明 |
|:---|:---|:---|
| PVTv2-B2 ImageNet 预训练 | `pretrained/pvt_v2_b2.pth` | 101 MB，取自 PVTv2 官方 release（whai362/PVT）；网络 `in_chans=8`，加载时其余层自动匹配 |
| 数据集 | `Dataset/TrainValDataset/{Imgs,GT}`、`Dataset/TestDataset/{CAMO,COD10K}/{Imgs,GT}` | 7000 张训练（CAMO 1000 + COD10K-CAM 3040 + COD10K-NonCAM 2960） |
| 先验 | `prior_sam3_c448/*.npz` | 自跑 Step1-3 得到；或向分享方单独索取（约 0.3 GB） |

SAM3 权重（`sam3-main/sam3-main/checkpoints/sam3.pt`）与 Qwen 模型
（`Qwen2.5-VL-3B-Instruct/`）只在 Step1/2 需要；**只做训练/推理不需要它们**。

### 4.3 先跑测试（都不需要 GPU、不需要数据）

```bash
python tests/test_train_args.py          # 参数默认值 + DP 闸门
python tests/test_prior_consistency.py   # 先验二值化 / train==test / neg_flag / 7ch 权重拒载
python tests/test_compact_priors.py      # 先验压实与 IoU 闸门
python tests/test_region_selector.py     # Step3 选择逻辑（网络不通时自动走 heuristic）
python tests/test_step1_annotator.py     # Step1 解析与兜底（47 项）
python verify_pipeline.py          # 目录/先验覆盖率自检
```

### 4.4 生成先验（已有 `prior_sam3_c448/` 就跳过）

```bash
python run_server_pipeline.py --root /srv/cod-workspace --fresh --yes
# 等价：bash run_server_pipeline.sh --fresh --yes
```

### 4.5 训练

```bash
unset CUDA_VISIBLE_DEVICES     # 见 §5.2
python MyTrain_SAM3.py --epoch 80 --batchsize 6 --gpus 1 \
  --prior_root prior_sam3_c448 --save_root model_pth/SAM3_1018 \
  --workers 8 --aug_mode fast 2>&1 | tee run_step4.log
```

### 4.6 推理 + 评估

```bash
python MyTest_SAM3.py --pth_path model_pth/SAM3_1018/KKK_best_XX.pth \
  --data_root ./Dataset/TestDataset/ --datasets CAMO \
  --prior_root prior_sam3_c448 --gpu_id 0
# 预测图 → results_sam3/SAM3_1018/CAMO/

python eval_metrics.py \
  --pair sam3:results_sam3/SAM3_1018/CAMO:Dataset/TestDataset/CAMO/GT
```

---

## 五、两个一定会踩的坑

### 5.1 DataParallel 会让训练慢 **70 倍**（本机实测）

| 配置 | s/step（bs=6，352px，2 个 size_rate） |
|:---|:---:|
| **单卡**（默认） | **0.271 s** |
| DP 2 卡（`--gpus 1,2 --dp`） | **19.06 s** |

这台机器**没有 NVLink**，`DataParallel.replicate()` 只有约 **2.2 GB/s**；
模型又小（batch3@352 的前向+反向仅 92 ms），每步 ~2834 个 kernel launch 在 GIL 上排队，
并行收益被通信 + 调度彻底吃掉。

**所以 `--dp` 改成默认关闭**：只写 `--gpus 1,2` 不带 `--dp` 时，会自动只用第一张卡并打印原因。
想确认自己的机器是不是也这样：

```bash
python tools/bench_step.py --mode step --synthetic --batchsize 6            # 单卡
python tools/bench_step.py --mode step --synthetic --gpus 1,2 --dp          # 对比
```

其余提速手段：`--amp`（bf16 前向，损失仍 fp32）、`--size_rates 1`（单尺度）、
`--workers 8`、`--aug_mode fast`。

### 5.2 `--epoch N` 训练 **1..N**，且 `CUDA_VISIBLE_DEVICES` 会二次过滤

```python
for epoch in range(start_epoch, opt.epoch + 1):   # --epoch 2 → 跑 2 轮
```

- `--epoch 80` 实际训练 epoch **1..80**（80 轮），与 `CosineAnnealingLR(T_max=opt.epoch)`
  和 `T_max` 警告里的轮数完全对齐；`--epoch 2` 冒烟会跑 **2** 轮（旧版左闭右开只跑 1 轮）；
- 3ch 基线 `MyTrain.py` 仍是 `range(start_epoch, opt.epoch)`（左闭右开），
  两个脚本的 `--epoch` 语义不同，不要拿同一套轮数预期去读日志；
- `MyTrain_SAM3` 内部 `os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus`，
  如果 shell 里残留 `export CUDA_VISIBLE_DEVICES=2,3`，再传 `--gpus 0`
  实际用的是**物理 2 号卡**。排查前先 `unset CUDA_VISIBLE_DEVICES`。

其它硬闸门（在分配显存前就退出）：

| 条件 | 原因 |
|:---|:---|
| `--batchsize 1` | CBAM 的 `BatchNorm2d` 收到 `[1,4,1,1]`，ValueError |
| `--dp` 且 `batchsize % n_gpus != 0` | `torch.chunk` 静默丢弃余数（6 张分 4 卡有一卡空转） |
| `--dp` 且每卡 < 2 张 | 同样触发 bs=1 的 BN 崩溃 |
| 最小 `size_rate × trainsize` < 64 | 插值后尺寸退化 |

---

## 六、复现 0.0697 的配方

| 项 | 值 |
|:---|:---|
| 权重 | `model_pth/SAM3_1017/`（服务器，**不随包发**） |
| 训练集 | `Dataset/TrainValDataset/` 全量 7000 张（含 NonCAM 2960 张，`neg_flag=1`） |
| 先验 | `prior_sam3_c448`（448×448 uint8 0/1，`--prior_size 448`） |
| 输入 | 352×352，8 通道 |
| 优化器 | AdamW，lr 2e-4，CosineAnnealing（T_max=80, eta_min=1e-6），clip 0.5 |
| batch / 卡 | 6 / **单卡** |
| 多尺度 | `--size_rates 0.75,1`（每步 2 次前向） |
| 增强 | `--aug_mode fast`（先缩到 1.25×trainsize 再裁剪/旋转/翻转） |
| 损失 | Hybrid E-loss × Sobel 边缘权重；深监督 0.4/0.25/0.2/0.15 |
| 选模 | 每个 epoch 在 CAMO 上算 MAE，取最小 → `KKK_best_{epoch}.pth` |

> **这个配方产出 `SAM3_1017` 权重**。正文里的 0.0697 是该权重在 **CAMO + COD10K 全量
> 约 8,275 张（训练 CAM 子集 4,040 + 测试集 4,250，含训练图）**上合并评测的结果；
> 训练期的选模是在 CAMO 250 张上做的，两者不是同一件事（见 §七）。

> 注意：**seed 固定（默认 0）但 cudnn.benchmark=True**，同机重跑能到 ±0.002 量级，
> 换卡/换驱动不要期待逐位一致。

---

## 七、已知限制

1. **0.0697 含训练图 → 偏乐观**：分母是 CAMO + COD10K 全量约 8,275 张
   （训练 CAM 子集 4,040 + 测试集 4,250），模型在 4,040 张上见过 GT。
   对外只能作为「达到目标线」的进度汇报，并且必须注明口径。
2. **与 0.0632 不可比**：基线是 CAMO 250 张纯测试集口径，分母不同，相减没有意义。
   「先验到底有没有用」要在同一 4,250 张测试集上跑 `--prior_mode real` vs `zero`
   才能回答 —— 这是当前主线待办，不是已有结论。
3. **分数据集（CAMO / COD10K 各自）尚未拆开重算**，Sm/Fβw 也未用该权重重跑。
4. **Step1 依赖 Qwen2.5-VL-3B**，中文能力弱（已全改英文 prompt），纯背景图幻觉明显。
5. **训练集里 2960 张 NonCAM 图 GT 全黑**，它们只贡献「别乱分割」的负样本，
   不代表模型在目标检测上的能力。
6. `model_pth/` 里的旧权重（`SAM3/KKK_best_41.pth` 等）是 **7 通道 / 无 prior_fusion** 的，
   `check_ckpt_channels.assert_ckpt_channels` 会直接拒绝加载 —— 那不是 bug。

---

## 八、文档

| 文件 | 内容 |
|:---|:---|
| `docs/project/项目文档.md` | 完整技术文档：架构、先验格式、超参、结果、逐条已修复问题 |
| `docs/project/项目进度.md` | v1→v5 演进 + 逐日更新日志（含本轮 DP 与 0.0697） |
| `docs/deployment/服务器部署说明.md` | 服务器落地手册；**§14 = 单卡优先 + DP 实测 + 参数闸门** |
| `docs/deployment/分享说明.md` | 本包边界（含/不含什么）与交接清单 |
| `docs/experiments/Layer0证据报告.md` | 0.18 / 0.0673 的口径问题取证 |
| `docs/experiments/E0E4执行记录.md` | E0–E4 本轮：Gate A 实测、冻结参数、服务器执行顺序与产物 |

---

## 九、E0–E4 纯推理对照（`tools/e0e4/`，2026-10 冻结）

目的：**不训练、不让 GT 进入任何生成阶段**，量出「3ch 基线 → ROI → Qwen 文本 → SAM3 → 固定融合」
这条链上每一步各自的价值。配置冻结在 `configs/e0e4_v1.yaml`，改任一数值即作废本轮
（必须新版本号、重跑 Gate C）。

| 阶段 | 做什么 | 输入 | 产物 |
|:---|:---|:---|:---|
| E0 | 3ch 基线推理（`Net_epoch_40.pth`，raw sigmoid，无 min-max） | 原图 | `e0/<ds>/<stem>/M0_soft.npy`、`M0_binary.png` |
| ROI | M0_binary 连通域 → ≤3 个 ROI（面积门槛 + 12% padding） | M0 | `rois/*.json`、`qwen_composites/*.png` |
| E1 | SAM3 box prompt，**每条 prompt 独立 `set_image`** | 原图 + ROI | `predictions/E1/` |
| E2 | Qwen 看原图 → 固定 JSON | 原图 | `captions/e2.jsonl` |
| E3 | Qwen 看三栏拼图（原图｜M0 叠加｜编号 ROI 裁剪） | 拼图 | `captions/e3.jsonl` |
| E2/E3→SAM3 | 每个 `sam_text` 一条 text prompt，独立 state | **原图**（不是拼图） | `predictions/E2`、`E3` |
| E4 | 固定非 GT 融合：mask IoU≥0.30 一对一匹配，`R = B ∪ (T ∩ dilate(B, 10% 长边))`；无匹配回退 E0 | E1+E3+E0 | `predictions/E4/`、`fusions/*.json` |
| Eval | 4 指标（MAE / Sα / meanE / Fβw，与 `eval_metrics.py` 同一后端）+ bootstrap 配对差 | GT（仅此处） | `metrics/*` |

```bash
# 服务器，仓库根目录；preflight 只要有一项 BLOCKED 就退出码 2，不要往下跑
python tools/e0e4/preflight.py --out outputs/E0_E4_ZEROTRAIN_V1/p0
python tools/e0e4/run_e0e4.py --split p0 --stage all --gpu_id 0   # Gate C：30 张（10/10/10）
python tools/e0e4/run_e0e4.py --split p1 --stage all --gpu_id 0   # Gate D/E：200 张（50/100/50）
python tests/test_e0e4.py                                               # 本机免 GPU 回归
```

硬约束（`guards.*` 必须全为 `false`，否则 preflight / run 直接拒绝）：GT 不进生成阶段；
文件名与 dataset 名不进 Qwen prompt（禁止 `llm_annotator.extract_context`）；NonCAM 不跳过
（只用于分层报告）；不做 min-max；不按测试集调阈值或融合规则；解析失败的 Qwen 回答不得人工改写后计入主结果。

**`e0_ckpt` 为什么是 `Net_epoch_40.pth`**：`model_pth/OUR/log.log` 打印
`MAE=46.076…` 与 `Best Epoch: 0`（GT 在 0–255、预测在 0–1；46.076/255 = 0.1807，恰为 CAMO 前占比），
其 best 选择不可信。因此预注册「训练跑满」的第 40 轮权重，sha256 随每张预测一起落盘
（`meta.json` / `E0_RUN.json`）。

**`e0.use_ca: false` 又是什么**：`model_pth/` 里所有 3ch 权重都只有 **740 key**，而当前
`Network_PVTv2`（`GRA` 里后加了 `ChannelAttention`）需要 **758**；缺的 18 个恰好是
`RS*.{weak,medium,strong}_gra.ca.fc.*`。也就是说，`MyTest.py` 的 strict 加载对这些权重本来就会失败。
`Network_PVTv2` 因此新增 `use_ca` 开关（**默认 True，训练与旧入口行为不变**），E0 用
`use_ca=False` 精确重建训练时架构、740/740 严格加载；`preflight` 同时验证
`use_ca=True` 必须判 BLOCKED，防止以后有人用随机初始化的 CA 层冒充基线。

产物树（`outputs/E0_E4_ZEROTRAIN_V1/p0|p1/`）：`INPUT_MANIFEST.csv`、`manifest_p0.txt`、
`manifest_p1.txt`、`e0/`、`rois/`、`qwen_composites/`、`captions/`、`sam3/`、`fusions/`、
`predictions/E0..E4/`、`metrics/`、`REVIEW_SHEETS/`、`TEXT_QUALITY_REVIEW.csv`、
`FAILURE_ANALYSIS.md`、`RUN_LOGS/`（`CONFIG_LOCKED.yaml`、`PREFLIGHT_REPORT.md`、
`ENVIRONMENT.md`、`<stage>.log`）、`CHECKPOINT_SHA256.txt`、`CODE_COMMIT.txt`。

---

## 十、致谢

- **PVTv2**：Wang et al., *PVT v2: Improved Baselines with Pyramid Vision Transformer*
  （`lib/pvtv2.py`，权重需自行下载）
- **SAM3**：Segment Anything Model 3（Step2 区域生成，权重与源码不在本包内）
- **Qwen2.5-VL**：阿里通义千问视觉语言模型（Step1 标注 / Step3 判别）
- 网络结构中的 RFB / NCD / ReverseStage 参考 camouflaged object detection 社区实现
  （SINet / PFNet 系列），`Network_PVTv2.py` 内已注明来源。
- 数据集：CAMO、COD10K、NC4K，请遵守各自 license，**不要随代码二次分发**。
