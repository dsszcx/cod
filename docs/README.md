# 文档索引

- [预测辅助门控双种子结果](experiments/预测门控双种子结果_20261113.md)保留复验反向结果；[同模块控制、空间负结果与计算开销](experiments/RGB预测辅助门控执行结果_20261112.md)同时列出raw/PNG四指标。

- [2026-11-10 融合研究进展](experiments/fusion-progress-20261110.md)汇总 CAM-only 80 轮四指标、前向开销、候选几何诊断和空间融合对照。
- [2026-11-11 独立种子 80 轮结果](experiments/独立种子80轮复核结果_20261111.md)与[空间融合 10 轮结果](experiments/候选内部空间融合结果_20261111.md)包含 raw、标准 PNG、分组退化和后续安排。
- [候选提示词对照](experiments/候选提示词对照结果_20261111.md)、[弱角色与背景候选干预](experiments/弱角色与背景候选干预结果_20261111.md)和[SAM3 四例阈值诊断](experiments/SAM阈值四例诊断_20261111.md)记录生成与融合的正负结果。
- [CamoFormer／ESCNet 阅读与接入建议](references/融合模块阅读与接入建议_20261111.md)记录已核对的模块操作和下一轮对照设计。

- `project/`：当前有效的项目说明、进度、要求和交付清单。
- `experiments/`：Layer-0、E0–E4 和 SAM3 的实验记录与证据。
- `deployment/`：服务器部署、分享、权重和大文件说明。
- `references/`：论文、研究计划和网络结构图。
- `archive/`：保留初始来源说明和排障记录，仅供追溯。

首次阅读建议顺序：

1. `project/项目交付文档.md`
2. `project/现行要求.md`
3. `experiments/E0E4执行记录.md`
4. `deployment/服务器部署说明.md`

历史文档中的指标可能采用旧口径。对外报告时以 `project/项目交付文档.md` 的口径警示为准。
