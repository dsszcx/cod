# CAMO轮廓退化与NonCAM误检分析（2026-10-22）

本轮使用已保存的P0预测开展本地分析，未运行服务器任务或训练。脚本：`tools/server/analyze_cod_failure_modes.py`；输出：`outputs/COD_FAILURE_ANALYSIS_20261022/`。

## 1. CAMO的主要问题不只是多ROI并集

| 分组（各10张） | baseline平均IoU | 全部ROI Mask并集 | 最大ROI对应Mask | GT挑选最佳单个已保存候选 |
|---|---:|---:|---:|---:|
| CAMO | 0.75713 | 0.36145 | 0.31766 | 0.34377 |
| COD10K-CAM | 0.43818 | 0.43246 | 0.45570 | 0.45570 |

最佳单个候选只从每次框提示已保存的最高分Mask中选择，属于GT参与的事后诊断，不能用于推理；也不包含SAM3内部未保存的其他候选。

- CAMO：2/10张的并集劣于最佳单个候选；8/10张所有已保存单个候选都不如baseline。只保留最大ROI不能修复整体问题。
- COD10K-CAM：5/10张并集劣于最佳单个候选；有6/10张的已保存候选能超过baseline。该分组确实存在可研究的合并/选择问题。
- 多ROI也可能共同覆盖目标，所以最佳单个候选平均IoU不一定高于并集；不能因为使用GT选择就把它解释为整个系统的性能上界。

代表案例`CAMO/camourflage_00430`：只有一个ROI，不存在多框合并。baseline IoU=0.8807，SAM3结果约0.0002；框覆盖目标，但SAM3选中背景纹理，候选分数为0.96875。ROI框由归一化中心x/y/宽/高构成，与现有SAM3 API约定相符。

代表案例`CAMO/camourflage_00800`：baseline IoU=0.9293，框提示结果为0；新版文本分割能得到目标外形，但与框分支不匹配而被融合回退。

## 2. 高分和框/文本一致性不足以排除NonCAM

新版E3在CAMO上4/10张非空，在COD10K-CAM上5/10张非空，在NonCAM上10/10张非空。

若仅观察已过滤记录中最高分≥0.9的图片：CAMO为2/10、COD10K-CAM为3/10，NonCAM却有4/10。提高分数要求会同时丢弃大量伪装图，同时保留部分普通物体误检。

新版E4在NonCAM有4/10张框/文本匹配成功；因此要求两分支一致也不能直接解释为“存在伪装目标”。两个分支可能同时分割普通物体。

这里的分数是原实验0.5过滤后保存的结果，EMPTY记录中没有原始低分。本分析不据此拟合分类阈值。

NonCAM的GT为全黑，但画面可能包含清晰的牛、鸟等普通物体。Qwen正确命名普通物体、SAM3正确分割其轮廓，仍会形成COD任务误检。这是任务定义与通用分割之间的区别，不能用文件名中的NonCAM标签作为推理开关。

## 3. 当前融合也会损失baseline正确像素

新版E4相对baseline有3张MAE退化，具体像素变化如下（占整图比例）：

- `CAMO/camourflage_00169`：丢失baseline正确前景0.012706，恢复前景0.005986；另外移除误检0.005123、新增误检0.000458。最终MAE从0.034473升至0.036529。
- `CAMO/camourflage_01188`：新增误检0.015215大于移除误检0.004287，最终MAE从0.018251升至0.023934。
- 牛的NonCAM样本：误检扩大，MAE从0.628227升至0.667330。

原因是E4有匹配时输出“匹配的SAM3框Mask与局部文本Mask的融合”，并没有保留baseline的所有正确区域。只有无匹配时才完整回退baseline。

## 4. 下一轮应保留粗Mask的空间信息

本次实际输入SAM3的是粗Mask生成的外接框。外接框丢掉了前景形状，也把背景纳入了提示范围；当前配置还未开启SAM3的实例交互模块。

仓库中的SAM3提供`enable_inst_interactivity=True`和`model.predict_inst(state, ...)`接口，可研究直接用baseline粗Mask、前景点/背景点提示来约束同一目标。需要先核对实例交互权重是否能从现有checkpoint正确加载，再跑少量样本。该方案应独立成新分支，与baseline及当前框提示比较，而非直接覆盖当前结果。

建议下一次先做“baseline粗Mask → 自动前景/背景点 → SAM3实例分割”的免训练对照，点位只由粗Mask生成，GT只用于最后评价；同时记录伪装图改善与NonCAM误检。它可能改善轮廓，但仍不能单独解决普通物体与伪装目标的任务区分。

已有P0被用于多次诊断，后续开发结果需明确标注探索性；确定方案后再用未参与诊断的样本验证。

## 5. 本地材料

- 每图诊断：`outputs/COD_FAILURE_ANALYSIS_20261022/per_image_diagnosis.csv`
- 每个框已保存候选：`outputs/COD_FAILURE_ANALYSIS_20261022/box_candidate_diagnosis.csv`
- 分组摘要：`outputs/COD_FAILURE_ANALYSIS_20261022/SUMMARY.json`
- 20张伪装图对比：`outputs/COD_FAILURE_ANALYSIS_20261022/REVIEW/`
