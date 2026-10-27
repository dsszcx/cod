# Layer-0 证据报告：SAM3 先验到底有没有用？

> 日期：2026-10-14（第 9 节为当晚补充；第 10 节为 2026-10-15 的 P0 守卫补记）｜ 数据：CAMO 测试集（250 张）+ 训练集（7000 张）
> 工具：`eval_metrics.py`、`prior_quality_report.py`、`_repro_v3/prior_bruteforce.py`（第 9 节）、`tests/test_prior_consistency.py`（第 10 节）
> 本文所有数字都可用文末命令一键复现，产物在 `outputs/metrics/`、`outputs/prior_quality/`。

## 0. 结论摘要

1. **历史宣传 “MAE 0.18 → 0.0673（↓63%）” 不成立。**
   “0.18” 从来不是 MAE：旧评测把**未归一化的 GT（0/255）**送进 MAE，算出来的是
   “GT 前景占比 × 255”（本机复现 **33.47**；旧日志的 **46.08** 是同一个量按图平均的版本）。
   它只反映数据集的前景比例，与预测好坏完全无关。
2. **统一口径重算后，SAM3 先验模型在 5 项主指标上全部低于 baseline**：
   MAE 0.0632 → 0.0673（**变差 6.5%**），Sm、Fβw、adpF1、adpIoU 无一例外。
3. **最根本的原因：那次实验的先验分支大概率从未收到过有效先验。**
   训练日志 `model_pth/SAM3/log.log` 的配置是 `prior_root='./prior_sam3/'`，
   而该目录在本机**不存在**；`dataloader_sam3._load_prior` / `MyTest_SAM3.load_test_prior`
   对缺失文件**静默返回全 0**，没有任何告警。于是“先验模型”实质是
   “多了一条恒为 0 的死分支的 baseline”。
4. **即使把先验做出来，现有区域质量也不足以支撑该设计**（第 3 节）：
   测试集 **37.4%** 的图**任何候选**都达不到 IoU 0.3（不可救），
   启发式选出的先验 IoU **中位数仅 0.001**（均值 0.292），31.6% 被质量门控清零；
   训练侧的原始候选只覆盖 **1000/7000 = 14.3%** 的图（缺的 6000 张全是 COD10K）。
5. **（第 9 节，2026-10-14 晚补）那次测试图是“带着真先验”跑出来的，而且可以逐图复现——
   所谓“无法复现的残差”主要是度量伪影 + 选目标策略差异，不是模型/代码问题。**
   `results_sam3/SAM3/CAMO` 与用**同一份权重**（`KKK_best_41.pth`）+ **同一份 HEAD 代码**
   （`_repro_v3/` 快照，`fc /b` 与 HEAD 无差异）重建的 250 张对比：**178 张完全一致**，
   72 张差 ≥0.01；对这 72 张逐图穷举 **7 种目标策略 × 4 种辅助策略**后 **69/72 复现到 <0.010**
   （62 张 <0.005，41 张 <0.002）。归因：(a) 度量伪影——旧口径拿 float 渲染比 uint8 PNG，
   天然带 ~0.1–0.2 灰阶底，26/72 张因此被误判为“复现不了”，真实噪声底只有 ~0.001–0.003；
   (b) 历史那次选的**目标候选不同**（胜出组合 45/72 是 `idx0|order0814`，即 08-14
   `qwen_select` 匹配失败回退 `tags[0]`），辅助装配规则则始终是 08-14 的 `order0814`。
   这条修正第 3 条的适用范围：**“死分支”是训练侧事实**（训练时 `prior_root` 缺失 → 先验恒 0）；
   测试侧先验是活的（HIST vs 全 0 先验渲染：均值 4.82；HIST vs 重建先验渲染：中位数 0.0039、
   178/250 完全一致），只是模型在“恒 0 先验”上训练、没见过真先验，所以用不好。

> 诚实声明：本机没有 `prior_sam3/`，无法 100% 复盘当时那台机器的目录状态。
> 但两种可能都指向同一结论：(a) 目录不存在 → 先验全 0；
> (b) 目录存在 → 它的输入源（`sam3_regions/`）只有 1250 张图的候选项，
> 训练时至少 86% 样本先验为 0，且被选中的先验中位 IoU 只有 0.001。
> 无论哪种情况，“先验带来 63% 提升”都不可能由这份数据产生。
>
> （2026-10-14 晚补）上述“目录存在与否”的疑问现在有了新证据：**本页结论不受影响，但
> “那次测试的先验也是 0”需要修正**——历史测试图能与本机重建在 178/250 张上完全一致，
> 剩下 72 张可被“换一个候选”复现到 <0.01（第 9 节），说明**历史测试确实消费了真实先验**；
> 只是那份先验文件已不在本机磁盘上（现存 `prior_sam3_0814/` 存的是启发式选择的结果，
> 例如 `00061` 的目标是 candidate #1 `tree bark`，而 HIST 对应的是 candidate #0）。
> 结论方向不变：训练侧先验恒 0 + 先验质量差 ⇒ “先验模型”必然打不过 baseline。

## 1. 指标口径修复（`eval_metrics.py`）

| 问题（修复前） | 现象 | 修复（修复后） |
|:---|:---|:---|
| `metrics` 字典为空 | 打印/落盘指标缺失、外部读不到 | 恢复 `metrics` 字典（`outputs/metrics/layer0_camo_std.json`） |
| GT 未归一化 | MAE 采样值 ≈ 33.5 / 46.08（假数字） | 仅保留为诊断字段 `mae_bug_unnormalized_gt`，不再当 MAE 用 |
| 没有前景占比参照 | 无法解释上述假数字 | 新增 `GTfg_ratio`（按图平均 18.12%）与 `GTfgPx_ratio`（按像素加权 13.16%） |
| 历史 per-image min-max 拉伸混在预测图里 | 评测口径不明 | 新增 `--minmax` 开关；现网预测图**已内含**该拉伸，故 on/off 数值相同 |

- `MAE_bug` 的成因：`mean |pred/255 - gt(0~255)|`。预测是 0~1（再除 255≈0），
  GT 是 0/255，于是结果 ≈ 平均 GT 强度 × 255。
  按图平均占比 0.1807 × 255 = **46.08**（旧日志）；本机按像素加权 0.1313 × 255 = **33.47**。
- 结论：这个字段只能是“数据集前景比例”的度量，**任何用 0.18 → 0.0673 的对比都是无效对比**。

## 2. 统一口径的模型对比（CAMO 测试集 250 张）

预测目录：`results/OUR/CAMO`（baseline）vs `results_sam3/SAM3/CAMO`（先验模型）。
`--minmax` 打开/关闭结果完全相同（说明预测图已含拉伸）。

| 方案 | MAE ↓ | Sm ↑ | Eφ ↑ | Fβw ↑ | adpFβ ↑ | adpF1 ↑ | adpIoU ↑ | adpDice ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline（无先验） | **0.063226** | **0.835056** | **0.890664** | **0.757220** | **0.783086** | **0.798896** | **0.693904** | **0.798896** |
| SAM3 先验 | 0.067318 | 0.823695 | 0.887857 | 0.746741 | 0.777919 | 0.785115 | 0.677568 | 0.785115 |
| 差值 | +0.00409（差 6.5%） | −0.0114 | −0.0028 | −0.0105 | −0.0052 | −0.0138 | −0.0163 | −0.0138 |

参照量：`GTfg_ratio = 0.181167`，`GTfgPx_ratio = 0.131554`，`MAE_bug ≈ 33.47`。

## 3. SAM3 先验质量实测（`prior_quality_report.py`）

先验链路：`llm_annotator.py`（Qwen 命名）→ `sam3_regions.py`（候选掩码）→
`region_selector.py`（选 1 个主目标 + 最多 3 个辅助 + 面积/置信度门控）。
“oracle 上界”= 在所有候选里取 IoU 最高者，即**选择策略完美时的天花板**；
“selected”= 现有选择策略真正喂给网络的那个先验。

### 3.1 测试集 CAMO（250 张）

| 指标 | 数值 |
|:---|:---|
| 有 npz 的图 | 250/250（100%） |
| 候选为空的 npz | 52（20.8%） |
| 候选数 / 图（均值/中位/p75/最大） | 1.51 / 2 / 2 / 4 |
| oracle 上界（n=198 有 GT） | 均值 **0.519**、中位 0.699、p25 0.029、p75 0.902；IoU≥0.5 占 **54.5%** |
| **任何候选都 < IoU 0.3（不可救）** | **74 张（37.4%）** |
| selected（真正喂网络的） | 均值 **0.292**、**中位 0.001**、p75 0.771；IoU≥0.5 仅 **31.6%** |
| selected 中 IoU<0.3（误导） | **64.8%** |
| 被门控清零的先验 | 79（31.6%）｜面积门控 4、置信度门控 24 |

### 3.2 训练集（7000 张，含 CAMO+COD10K）

| 指标 | 数值 |
|:---|:---|
| 有 npz 的图 | **1000/7000（14.3%）**；缺的 6000 张**全部是 COD10K** |
| 候选为空的 npz | 240（24.0%） |
| oracle 上界（n=760 有 GT） | 均值 **0.604**、中位 0.821；IoU≥0.5 占 64.7% |
| 任何候选都 < IoU 0.3（不可救） | 201（26.4%） |
| selected | 均值 **0.309**、**中位 0.001**；IoU≥0.5 占 **32.8%** |
| 被门控清零的先验 | 301（30.1%）｜面积门控 12、置信度门控 52 |

读法：
- **候选不是没有**（oracle 均值 0.52~0.60 说明 SAM3 常能框住目标），
  但**选择策略把一半以上的好候选弄丢了**（oracle 0.519 → selected 0.292，中位数直接掉到 0.001）。
- 训练集覆盖只有 14.3%，而 `_load_prior` 对缺失文件补零 ⇒ **训练时至少 86% 的样本先验为全 0**；
  先验分支在绝大多数步数上学到的是“忽略这个输入”。
- 常见标签（top tags）：bird、tree bark、rock、leaf、owl、spider、camouflage、person
  —— 含相当比例的**背景类**标签（tree bark / rock / ground），说明 Qwen 命名质量本身也是噪声源。

## 4. 根因链

```
sam3_regions 只覆盖 1250/7250 张图（COD10K 全缺）
        +
prior_sam3/ 目录缺失（本机事实）→ _load_prior 静默补零，无任何告警
        ↓
训练样本 ≥86% 先验为全 0；测试样本同样走同一条补零代码路径
        ↓
先验分支 = 常数 0 输入 → 梯度学不到任何与先验相关的信息（退化成一条死分支）
        ↓
与 baseline 相比只多了无用参数/无用输入通道 → CAMO 上 5 项指标全面下降
        ↓
旧评测又用未归一化 GT，把 46.08 当成“baseline MAE” → 伪造出 “↓63%” 的假结论
```

## 5. 已加入的防护与新开关（默认行为不变，除非显式传参）

| 位置 | 入口 | 作用 |
|:---|:---|:---|
| `dataloader_sam3.check_prior_coverage()` | 被训练/测试脚本自动调用 | 打印 `prior_root` 是否存在、`n_have/n_total` 覆盖率；**`--prior_mode real` 且覆盖率为 0 时直接抛 `FileNotFoundError`**，杜绝再次静默训练死分支 |
| `dataloader_sam3.py` | `--prior_mode real/zero/oracle/shuffle` | 消融：真实先验 / 全 0 / 用 GT 当先验（分支上界）/ 打乱先验（负对照） |
| `dataloader_sam3.py` | `--img_filter`、`--seed` | 子集训练（如只训 CAMO）、数据增强与 shuffle 可复现 |
| `MyTrain_SAM3.py` | `--minmax`、`--gpus`、`--seed` | 验证口径随评测口径统一、多卡、可复现 |
| `MyTest_SAM3.py` | `--minmax`、`--save_name`、`--prior_mode` | 关掉历史拉伸；消融结果写到 `results_sam3/<save_name>/`，**不会覆盖 baseline 结果** |
| `eval_metrics.py` | `--minmax` | 唯一指标口径，兼容历史预测图 |
| `prior_quality_report.py` | 见文末命令 | 覆盖率 / 候选数 / oracle 上界 / 实际先验 IoU / 清零比例 |

## 6. 建议的下一步（按优先级）

1. **先补数据，再谈模型**：把 prior 生成链路在全部 7000 张训练图 + 目标测试集上跑通
   （`llm_annotator.py` → `sam3_regions.py` → `region_selector.py` → `prior_sam3/`），
   否则先验分支永远是死分支。补齐前后都用 `check_prior_coverage()` 验收。
2. **做 3 个廉价消融确认“分支是否有信息通路”**（每个 run 只改一个参数）：
   `--prior_mode zero`（下界）vs `real`（现状）vs `oracle`（上界）。
   若 `oracle` 不明显优于 `zero` ⇒ 网络结构用不上先验，应直接砍掉该分支；
   若 `oracle` 明显更好但 `real` 不行 ⇒ 问题在先验质量/选择策略，值得继续优化
   （当前 oracle vs selected 的巨大差距已经指向选择策略）。
3. **修选择策略**：现策略按 `sam3_score × sqrt(area) × objectiveness` 排序，
   中位 IoU 只有 0.001；至少应加入“与图像显著性/背景对比度”的一致性检验，
   或直接取面积占比最接近前景先验分布（≪0.18）的候选。
4. **重训与复评**：任何结论都必须用 `eval_metrics.py`（统一口径、`--minmax` 一致）报告，
   并同时给出 `outputs/prior_quality/*.json` 的覆盖率，避免再次出现“先验没接上却报提升”。

> （2026-10-14 晚补）第 9 节给第 3 条补上了直接证据：2026-08-16 那次测试用的是 08-14 的
> `qwen_select`，它因为 bug 会大比例退化成“回退候选 #0”（72 张残差图里 45 张的目标就是候选 #0），
> 也就是说**当时喂进网络的先验并不是按启发式分数选出来的**。所以“修选择策略”这件事不仅仍然必要，
> 而且**在 `0c76dca` 之前根本没生效过**——任何“先验模型不行”的结论都应先在当前 HEAD 代码上重跑一次先验。

## 7. 复现命令

```bat
:: 1) 统一口径评测（baseline vs SAM3，含历史 min-max 口径）
python eval_metrics.py ^
  --pair baseline:results/OUR/CAMO:Dataset/TestDataset/CAMO/GT ^
  --pair sam3:results_sam3/SAM3/CAMO:Dataset/TestDataset/CAMO/GT ^
  --minmax --out outputs/metrics/layer0_camo_minmax.json --csv outputs/metrics/summary.csv

:: 2) 先验质量：测试集
python prior_quality_report.py --regions sam3_regions --splits test ^
  --gt_dir Dataset/TestDataset/CAMO/GT --img_dir Dataset/TestDataset/CAMO/Imgs ^
  --out outputs/prior_quality/test.json

:: 3) 先验质量：训练集（覆盖率会暴露 COD10K 全缺）
python prior_quality_report.py --regions sam3_regions --splits train ^
  --gt_dir Dataset/TrainValDataset/GT --img_dir Dataset/TrainValDataset/Imgs ^
  --out outputs/prior_quality/train.json

:: 4) 有真实 prior_sam3/ 时（测“真正喂进网络的先验”）
python prior_quality_report.py --regions sam3_regions --prior prior_sam3 --splits test ^
  --gt_dir Dataset/TestDataset/CAMO/GT --img_dir Dataset/TestDataset/CAMO/Imgs

:: 5) 消融（不覆盖 baseline 结果；权重名按实际最优替换，
::    model_pth/SAM3/ 下有 KKK_best_{1,2,3,4,6,7,8,11,12,16,22,27,30,39,40,41}.pth）
python MyTest_SAM3.py --pth_path ./model_pth/SAM3/KKK_best_41.pth ^
  --prior_mode zero --save_name SAM3_zero --datasets CAMO
python MyTest_SAM3.py --pth_path ./model_pth/SAM3/KKK_best_41.pth ^
  --prior_mode oracle --save_name SAM3_oracle --datasets CAMO
```

## 8. 产物清单

| 文件 | 内容 |
|:---|:---|
| `outputs/metrics/layer0_camo_minmax.json` / `layer0_camo_std.json` | 两种口径的完整指标（含 `metrics_all` 的 13 项指标） |
| `outputs/metrics/summary.csv` | 上述结果的扁平表 |
| `outputs/metrics/*.log` | 评测过程日志 |
| `outputs/prior_quality/test.json` / `train.json` | 先验质量结构化结果（覆盖率、oracle、selected、门控统计） |
| `outputs/prior_quality/test.log` / `train.log` | 人类可读的报表 |
| `outputs/prior_quality/_repro_v3/` | 复现工程：HEAD 逐字节快照（`Network_SAM3.py`/`Network_PVTv2.py`/`MyTest_SAM3.py`）、`residual.json`（72 张残差图清单）、`prior_bruteforce.py`（逐图策略穷举） |
| `outputs/prior_quality/_bf_result.json` / `.log` | 穷举原始结果（每图 28 种组合的 `diff`/`d_float`/`matches`） |
| `outputs/prior_quality/_bf_summary_png.txt` | 穷举汇总：headline 数字、胜出组合分布、未复现清单、度量污染清单（第 9 节表格来源） |
| `outputs/prior_quality/_probe_png.txt` | 度量探针：float 口径 vs PNG 口径对照 + 与 `results_sam3_heursrc` 的一致性 |
| `outputs/prior_quality/_prior_match.log` | 盘上两份先验（`prior_sam3_0814` / `prior_sam3_sel`）各自选了哪个候选（250 张全量、CPU） |
| `outputs/prior_quality/_head_snapshot_check.log` | `fc /b` 证明 `_repro_v3/` 快照 == HEAD（`0c76dca`）代码 |
| `outputs/prior_quality/prior_attrib.csv` / `.log` | HIST vs 全 0 先验 / vs 重建先验 的逐图距离（判断“先验分支是否活”） |

## 9. 残差归因：逐图策略穷举（2026-10-14 晚补）

> 起因：第 2 节的 SAM3 预测目录 `results_sam3/SAM3/CAMO`（2026-08-16 那次测试）无法在本机“完全重建”——
> 同一份 `KKK_best_41.pth` + 同一份 08-14 选择器，重建结果（`results_sam3_heursrc/`）与之在 **72/250** 张图上
> 有 ≥0.01 的 PNG 差（最大 28 灰阶）。这份差到底是模型/代码问题，还是先验/度量问题？

### 9.1 方法

- 脚本：`outputs/prior_quality/_repro_v3/prior_bruteforce.py`。
- 代码与权重：`_repro_v3/Network_SAM3.py`、`_repro_v3/Network_PVTv2.py` 与 **HEAD（`0c76dca`）逐字节相同**
  （`fc /b` 无差异，见 `_head_snapshot_check.log`），权重 `model_pth/SAM3/KKK_best_41.pth`（7 通道）。
- 穷举空间：每张图 **7 种目标策略 × 4 种辅助策略 = 28 种组合**，全部走真实前向。
  - 目标策略：`idx0` / `idx1` / `idx2`（直接取第 0/1/2 个候选）、`max_score`、`max_area`、
    `heur`（08-14 的 `score×√area×objectiveness`）、`prev`；
  - 辅助策略：`order0814`（08-14 的候选顺序装配）、`iourank`、`nofilter`、`zeros`。
- 先验来源：`sam3_regions/*.npz` 的**原始候选掩码**，逐图现场装配（不使用盘上任何 `prior_sam3*` 目录），
  避免“盘上先验已被覆盖”的干扰。
- 样本：72 张残差图 + 6 张控制图（历史与重建完全一致、`d_heur0814 ≈ 0.0006–0.0031`）。
- 度量：**PNG vs PNG**（`cv2.imencode/imdecode` 后再比，与历史图同一数值空间），float 口径另存为 `d_float`。

### 9.2 第一次穷举“搜不到匹配”的原因：度量错了

第一版脚本把 `float(res)*255` 直接和 uint8 的 HIST 相减。这个口径天然带 **~0.1–0.2 灰阶**的底
（`CV_32F → CV_8U` 舍入 + 编码差异），比要追的残差还大，于是“永远搜不到匹配”。`00061` 是最干净的对照：

| 口径 | 最优组合 | 数值 | 结论 |
|:---|:---|---:|:---|
| float `res*255` vs HIST（旧） | `idx0\|order0814` | 0.0105 | 被 ~0.1 的底抬高 → 误判“没匹配上” |
| **PNG vs PNG（现）** | `idx0\|order0814` | **0.0004** | 真匹配 |
| PNG vs PNG | `heur\|order0814` | 4.4260 | = 盘上 `d_heur0814`（4.4258），度量自洽 |

同类被旧口径误判的图共 **26 张**（PNG 距离 <0.01 而 float 口径 ≥0.05），例如
`01064` 0.0084 vs 0.1501、`00285` 0.0080 vs 0.1491、`01159` 0.0073 vs 0.1323。
也就是说：**“残差无法复现”里有 26/72 张是度量伪影，真实噪声底只有 ~0.001–0.003。**

### 9.3 结果：69/72 可复现

| 项 | 数量 |
|:---|---:|
| 残差图（`d_hist_heur ≥ 0.01`） | 72（其余 178/250 完全一致） |
| 控制图：最优组合 = `heur\|order0814`（6/6），且数值 ≈ 盘上 `d_heur0814` | 6 |
| 复现到 `diff < 0.002` / `< 0.005` / **`< 0.010`** | 41 / 62 / **69** |
| 逐字节相同（`diff == 0`） | 0 |
| 残差图中 `n_cand ≤ 1`（先验与策略无关） | 5（2 张 `n_cand=0` + 3 张 `n_cand=1`） |
| 残差图中 `n_cand ≥ 2`（策略能改变先验） | 67 |

胜出组合分布（72 张，`_bf_summary_png.txt`）：

```text
idx0|order0814        45
idx1|order0814        10
max_score|order0814    6
max_area|order0814     5
heur|order0814         5   <- 恰好是那 5 张 n_cand<=1 的图（换策略先验也不变）
idx2|order0814         1
```

两条结论直接从这张表读出来：

1. **辅助策略 72/72 都是 08-14 的 `order0814`** → 辅助装配规则从未变过
   （09-24 新增的 `iourank` / `nofilter` / `zeros` 一次也没赢）；
2. **目标策略以 `idx0` 为绝对多数（45/72）** → 历史那次选的**不是**启发式那个候选，而是**候选 #0**。

### 9.4 剩下 3 张：噪声底，不是策略问题

| 图 | n_cand | d_heur0814 | 最好 diff | 备注 |
|:---|---:|---:|---:|:---|
| `01141` | 2 | 9.4513 | 0.0117 | 唯一真正的“擦边球”（阈值 0.010） |
| `00924` | 0 | 0.0113 | 0.0101 | `n_cand=0`：任何策略下先验都是全 0 |
| `01030` | 0 | 0.0104 | 0.0100 | 同上 |

5 张 `n_cand ≤ 1` 的图最好的 diff 落在 **0.0059–0.0101** —— 也就是说**即使先验逐字节相同**，
“前向 + 编码”仍会留下 ~0.006–0.010 的灰阶差（远超 float 口径暗示的 0.011 底，但不是策略能消除的）。
真正“无法解释”的只有 `01141` 一张擦边球。

> 补记：早先的直觉是“`n_cand ≤ 1` 的图先验唯一 ⇒ 必须逐字节复现”，而实测差 0.010–0.015，
> 一度被视为矛盾。现在定性为 **float 口径的 ~0.1 底 + 真实灰阶噪声底 ~0.006–0.010**，矛盾消失。

### 9.5 为什么 HIST 对应“候选 #0”

三条独立证据：

1. **代码**：08-14 的 `region_selector.py` 快照（`outputs/prior_quality/_rs_0814.py`）里
   `qwen_select()` 的 `target_idx` **初值就是 `0`**，只有 Qwen 的回答与某个 tag 互相包含时才会改；
   `docs/project/项目进度.md` §4.5 明确记录这条路径**“之前 100% fallback”**，
   该 bug 直到 `0c76dca`（2026-09-24，也就是现在的 HEAD）才修——**晚于** 2026-08-16 那次测试。
2. **穷举**：45/72 张的胜出目标是候选 #0（其余 27 张为 `idx1`/`idx2`/`max_score`/`max_area`/`heur`），
   辅助策略 100% 是 `order0814`——与“Qwen 偶尔模糊命中别的 tag，否则回退候选 #0”完全吻合。
3. **盘上先验**（`_prior_match.log`，250 张全量、纯 CPU）：两份现存先验的选择分布都不是“以 #0 为主，
   且都不是 HIST 用的那份”：

   | 先验目录 | 目标 == candidate 0 / 1 / 2 / 3 | 全 0 | 与 HIST 的关系 |
   |:---|:---|---:|:---|
   | `prior_sam3_0814/`（`results_sam3_heursrc` 的生成源） | 134 / 50 / 12 / 2 | 52 | 例如 `00061` 目标 = candidate #1（`tree bark`），而 HIST 对应 candidate #0 |
   | `prior_sam3_sel/` | 115 / 44 / 10 / 2 | 79 | 同样 `00061` = candidate #1 |

   两者目标候选一致的只有 **223/250**。也就是说：**HIST 消费的那份先验不在这两份盘上文件里**
   （当时应是 08-14 `qwen_select` 路径现场产出的，之后被覆盖/清理）。

结论：72 张残差的可复现解释 = **先验来源/选目标不同**，不是模型、代码或评测口径问题。

### 9.6 与第 3 节“死分支”结论的关系（修正）

| 阶段 | 先验实际值 | 证据 |
|:---|:---|:---|
| **训练** | **恒 0**（`prior_root='./prior_sam3/'` 缺失 → `_load_prior` 静默返回全 0） | 训练配置 + 数据加载代码 |
| **测试**（2026-08-16） | **真实先验（活的）** | HIST vs 全 0 先验渲染：均值 4.82、仅 61/250 一致；HIST vs 重建先验渲染：中位数 0.0039、178/250 完全一致；`d2_zero_vs_prior.log`：全 0 vs 重建先验 0/250 一致、均值 4.30 |

即“死分支”**只在训练侧成立**：网络在“先验恒 0”的分布上训练，测试时却收到真实先验，
属于训练/推理分布不一致——这正解释了“有真先验也没用”（real 口径 MAE 0.0673 vs zero 口径 0.0712，
两者都差于 baseline 0.0632）。第 3 节的结论（训练侧缺先验 + 先验质量差）仍然成立，
但**不能**再写成“那次测试的先验也是 0”。

### 9.7 代码/权重口径（本次复现的前置条件）

| 项 | HEAD（`0c76dca`） | 工作区（v4/v5 未提交） |
|:---|:---|:---|
| `Network_SAM3.py` | **7 通道**（`in_chans=7`，无 `neg_flag`） | **8 通道**（`+neg_flag`，新增 `ChannelAttention`） |
| patch_embed 形状 | `64×7×7×7` | `64×8×7×7`（与 7ch 权重不兼容） |
| 可用权重 | `model_pth/SAM3/KKK_best_41.pth`（7ch） | **本机没有 8ch checkpoint** |

- 7ch 与 8ch 的 `patch_embed` 权重形状不同 ⇒ `KKK_best_41.pth` **只能**装进 HEAD 那版 7 通道代码；
  8ch 代码在本机无法加载任何已有权重（这正是 8ch 全量消融被阻塞的原因）。
- 所以本次穷举用的是 `_repro_v3/` 下的 7ch 快照——已用 `fc /b` 证明它与 HEAD 逐字节一致
  （`_head_snapshot_check.log`），即“跑的确实是已提交的代码”。
- `9007b13`（`docs: SAM3 prior results report + fix test prior resize`）只改了
  `MyTrain_SAM3.py`（+6 行）和一份文档，**不触碰测试前向**，无法解释残差。

### 9.8 复现命令（第 9 节新增产物）

```bat
:: 逐图策略穷举（需要 GPU；72 残差 + 6 控制 × 28 组合）
::   -> outputs/prior_quality/_bf_result.log / _bf_result.json
.venv\Scripts\python.exe outputs\prior_quality\_repro_v3\prior_bruteforce.py

:: 汇总（读 _bf_result.json；本机终端抓取不稳，一律重定向到文件）
.venv\Scripts\python.exe outputs\prior_quality\_bf_summary.py > outputs\prior_quality\_bf_summary_png.txt

:: 度量探针：同一个前向，同时报 float 口径 / PNG 口径 / 与 heursrc 的距离
.venv\Scripts\python.exe outputs\prior_quality\_probe_png.py > outputs\prior_quality\_probe_png.txt

:: 盘上两份先验各自选了哪个候选（纯 CPU，250 张全量）-> _prior_match.log
.venv\Scripts\python.exe outputs\prior_quality\_prior_match.py

:: 代码快照一致性（HEAD 的 7ch 代码 == _repro_v3/ 快照）
git --no-pager show HEAD:Network_SAM3.py > outputs\prior_quality\_h_ns.py
fc /b outputs\prior_quality\_h_ns.py outputs\prior_quality\_repro_v3\Network_SAM3.py
git --no-pager show HEAD:Network_PVTv2.py > outputs\prior_quality\_h_pvt.py
fc /b outputs\prior_quality\_h_pvt.py outputs\prior_quality\_repro_v3\Network_PVTv2.py
```

## 10. 第二条“先验静默归零”路径 + P0 守卫（2026-10-15）

第 3 / 9.6 节的“训练侧先验恒 0”由 **`prior_root` 目录缺失**造成。构建 P0 守卫时又发现
**另一条独立路径**，它与目录是否存在完全无关，因此即便 `prior_sam3/` 齐全也会静默归零：

| 现象 | 根因 | 修复 |
|:---|:---|:---|
| 同一份 prior 目录：MyTest 报 `40/40`，MyTrain 验证循环报 `0/40` | `dataloader.test_dataset.__init__` 用字符串拼接后取 `path.split('/')[-1]` 当文件名；`MyTrain_SAM3.py` 用 `os.sep` 拼 `image_root`，Windows 下是反斜杠 → `split('/')` 切不动 → `name` 变成**整条路径** → `test_<name>.npz` 永远不存在 | ① 根目录一律用 `/`（`os.path.join(...).replace(os.sep, "/") + "/"`）；② `load_test_prior` 先归一化成 basename 再解析 |
| `x.png` 的图匹配不到已有的 `test_x_jpg.npz` | 旧单点拼接只试 `.jpg`（`name.replace('.png','.jpg')`），后缀一变就落空 | 新增 `_prior_npz_candidates()`：basename + `.jpg`/`.png` 两种后缀，全部候选按序探测；训练/推理/Web 共用 |

同一份 40 张 fixture（其中 10 张的先验 npz **故意删除**作阴性对照）上的实测：

| 入口 | 修复前 | 修复后 |
|:---|:---|:---|
| MyTrain 验证循环 | `prior 0/40 (missing 40)`，MAE 0.3933 | `prior 30/40 (missing 10)`，MAE 0.2965 |
| MyTest | （同目录能得到正确数，但与训练侧不同源） | `prior 30/40 (missing 10)` |

报告的 30/40 与盘上实际存在的 npz 数逐一对齐；MAE 变化说明先验**真的进了网络**
（该数字来自 8 张图的 smoke fixture + 随机权重，只用于判断“分支是否活”，不代表性能）。

### 10.1 守卫清单（`tests/test_prior_consistency.py`，纯 CPU，秒级）

| 检查 | 内容 | 现状 |
|:---|:---|:---|
| [1] 先验二值化 | `prepare_prior_at` 必须输出 0/1（旧路径残留 1830 个灰阶像素） | PASS |
| [2] 单一先验解析 | 训练/推理/Web 同一份解析与 `prepare_prior_at`；含**全路径名**、`.png` npz、**缺文件 → `found=False`** | PASS |
| [3] `neg_flag` 规则 | COD10K `NonCAM` → 1（训练集 2960/7000，COD10K-test 1974/4000） | PASS |
| [4] 权重通道守卫 | 7ch 必须**拒绝**（旧代码静默装载）；8ch + `prior_fusion` 才可加载 | PASS |

各入口还都能显式报告覆盖率，`--strict_prior` 则把“缺任一先验”从 WARN 升级为中止
（`check_prior_coverage` 原本只在 0% 覆盖率时才报错，部分缺失会被静默吞掉）：

```bat
:: 纯 CPU 一致性回归（无需 GPU）
.venv\Scripts\python.exe test_prior_consistency.py

:: 训练 / 推理：报告先验覆盖率，缺任一先验即中止
.venv\Scripts\python.exe MyTrain_SAM3.py ... --strict_prior
.venv\Scripts\python.exe MyTest_SAM3.py ... --strict_prior
```
