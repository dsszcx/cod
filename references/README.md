# 外部代码与模型来源

本仓库发布项目自有代码、配置、测试和实验记录，不重复上传外部完整源码、数据集或权重。本机的外部目录仍保留，不因Git忽略规则而删除。

- SINet-V2：原始仓库 https://github.com/GewelsJI/SINet-V2 ，本次官方评估链路使用commit `5b9844c94d10a03b815c7c7039c194166a9a8ba3`。本地检出放在`references/upstream/SINet-V2/`；复核入口见`tools/official_sinetv2/README.md`。权重另行按作者说明获取，不随代码发布。
- SAM3：原始仓库 https://github.com/facebookresearch/sam3 。本地`sam3-main/`及其checkpoints不随本次项目快照发布；按现有部署说明准备兼容`build_sam3_image_model`、`Sam3Processor`的环境和本地权重，不把示例媒体或数据当作项目训练数据提交。
- Qwen2.5-VL-3B-Instruct：使用单独准备的本地模型目录。模型文件及访问凭据不进入Git。

外部资源继续遵守各自许可证和数据/模型使用条款。本次排除仅影响新发布分支的当前文件树，不删除本机文件，也不改写仓库已有历史。
