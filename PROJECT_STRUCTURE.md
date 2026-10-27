# 项目目录框架

本文件是仓库的“地图”。整理原则是：运行入口稳定、代码与资产分离、实验材料可追溯。

## 1. 日常使用从哪里开始

| 目标 | 入口 |
|:---|:---|
| 了解项目现状 | `docs/project/项目交付文档.md` |
| 最新方法主线与运行入口 | `docs/project/当前方案_CAM_only.md` |
| 查看当前要求及CAM-only数据范围 | `docs/project/CAM_only数据范围调整_20261026.md`、`configs/relation_cam_only_20261026.yaml` |
| 查看历史E0–E4任务书（原只读文件） | `docs/project/现行要求.md` |
| 查看新主线接口设计 | `docs/project/多语义多区域主辅关系_COD接口设计_V1.md` |
| 新主线关系网络（已完成小样本短训练，尚无正式效果结论） | `Network_Relation.py`、`lib/region_fusion.py` |
| 2026-10-26短训练、复现与先验作用诊断 | `docs/experiments/本周短训练与复现性诊断_20261026.md` |
| CAM-only两小时融合对照（128/32、两种子） | `docs/experiments/CAM_only融合可学习性对照_20261026.md` |
| 候选可靠性与关系作用研究（两种子10轮，直接监督门控的负结果） | `docs/experiments/候选可靠性与关系学习_20261026.md` |
| 训练 8 通道模型 | `MyTrain_SAM3.py` |
| 推理 8 通道模型 | `MyTest_SAM3.py` |
| 运行 Step1–5 | `run_server_pipeline.py` |
| 运行 E0–E4 | `tools/e0e4/run_e0e4.py` |
| 执行回归测试 | `tests/` |
| 查看服务器部署方法 | `docs/deployment/服务器部署说明.md` |

## 2. 目录职责

| 路径 | 内容 | 是否建议手工修改 |
|:---|:---|:---:|
| `configs/` | 冻结实验配置 | 谨慎 |
| `tools/` | 实验编排、压缩、测速工具 | 是 |
| `tests/` | CPU 回归测试 | 是 |
| `docs/project/` | 当前有效的项目级文档 | 是 |
| `docs/experiments/` | 实验方案、证据和结果记录 | 是 |
| `docs/deployment/` | 部署与交付说明 | 是 |
| `docs/references/` | 论文和研究参考资料 | 通常否 |
| `docs/archive/` | 初始来源说明与排障记录，仅供追溯 | 否 |
| `releases/` | zip/tar 发布包及校验文件 | 由脚本生成 |
| `data_archives/` | 临时数据传输包；当前仅保留说明文件 | 否 |
| `Dataset/` | 训练与测试采用的标准数据布局 | 否 |
| `COD10K-v2/` | COD10K 官方原始数据 | 否 |
| `model_pth/` | 训练权重 | 否 |
| `pretrained/` | 预训练骨干权重 | 否 |
| `sam3_regions/`、`prior_sam3*/` | 中间先验产物 | 否 |
| `outputs/`、`results*/` | 日志、指标和预测结果 | 否 |

根目录中的 Python 文件是现有训练/推理链直接依赖的模块。为避免破坏原命令和服务器脚本，本轮不强行迁入 `src/`。

## 3. 常用命令

```bash
# 快速回归
python tests/test_train_args.py
python tests/test_prior_consistency.py
python tests/test_compact_priors.py
python tests/test_region_selector.py
python tests/test_step1_annotator.py
python tests/test_layer0_fixes.py
python tests/test_e0e4.py

# 数据与流水线自检
python verify_pipeline.py

# 只查看服务器包清单，不生成压缩包
python make_server_bundle.py --list --no_weights
```

## 4. 新文件放置规则

- 新的实验配置放入 `configs/`，不要散落在根目录。
- 新的回归测试放入 `tests/`，文件名采用 `test_*.py`。
- 新的实验工具放入 `tools/`；属于 E0–E4 的模块放入 `tools/e0e4/`。
- 当前有效说明放入 `docs/project/`；实验记录放入 `docs/experiments/`。
- 历史文档只移入 `docs/archive/`，不直接删除。
- 新生成的交付包放入 `releases/`，数据压缩包放入 `data_archives/`。
- 数据、权重和实验输出不提交 Git；提交前先检查 `git status --short`。
