# 官方 SINet-V2 权重 / 评估链路核验

状态（2026-10-26）：**CAMO 250 张和 COD10K CAM 2026 张的官方权重推理与四指标 Python 协议复核已全部完成，退出码 0，出现 `OFFICIAL_SINETV2_PYTHON_PROTOCOL_CHECK_DONE`。两个数据集全部 8 个指标按论文报告精度（小数点后三位）与论文一致。** 用户提供的权重原在 `C:/Users/USER/Downloads/Net_epoch_best.pth`，现已按用户要求移动到项目 `model_pth/SINet_V2/Net_epoch_best.pth`，Downloads 原文件不再保留；文件为 108,330,636 bytes，严格加载全部 880 个参数张量成功；移动前后及服务器 SHA256 一致：`460188b602f2b88951918006255a70df829a55512386d8ae74f84423881ddbbb`。6 项公式/边界单元测试、官方网络 CPU 前向兼容性检查、完整数据配对/尺寸/二值性检查通过。完整结果见 `docs/reports/官方SINetV2评估链路核验_20261026.md`。这不是自研模型的结果，也不是原版 MATLAB 执行结果。

## 固定来源与范围

- 作者仓库：https://github.com/GewelsJI/SINet-V2
- 源码提交：`5b9844c94d10a03b815c7c7039c194166a9a8ba3`，本地 `references/upstream/SINet-V2`。
- 作者 README 的权重链接：https://drive.google.com/file/d/1D3RKQ8Nzd0ArV_c47StVKEuaoYTwnclR/view 。文件 `Net_epoch_best.pth`；建议放在 `model_pth/SINet_V2/`。取得后记录 SHA256；哈希仅用于后续一致性记录，不等于作者提供了签名。
- 网络使用作者原始 `Network_Res2Net_GRA_NCD.Network(imagenet_pretrained=False)`，严格加载全模型权重，不替换为项目的 PVTv2/CBAM 网络，不再下载 ImageNet 权重。
- 官方测试集分别计算：CAMO 250 张、COD10K 中 `-CAM-` 图像 2026 张。COD10K 目录实际有 4000 张，NonCAM/背景图不进入这次标准 CAM 评估。
- CHAMELEON 76 张目前未发现，不声称完成全部基准复现。不使用训练子集/32 张内部验证集替代官方测试集。

## 协议

RGB → PIL/torchvision resize 352×352 → ImageNet Normalize → 原始网络最后一个输出 → 双线性恢复原图分辨率（align_corners=False）→ sigmoid → 逐图 minmax → uint8 PNG。

官方旧 SciPy `misc.imsave` 在新环境不可用；本适配器复刻其 2D float bytescale 舍入步骤，不修改作者网络。评估读取 PNG，再按作者 MATLAB `main.m` 对每图 minmax；GT 使用 `>128`。四指标为 S-measure、256 阈值 **strict >** 的 mean E、weighted F(beta=1)、MAE。不用 max/adaptive E 替代 mean E，不混用旧实验 raw 概率协议。

`metrics.py` 是独立 Python 公式复核：保留双精度、常量图 mapminmax 不变、MATLAB 一基坐标和半向上取整。修正直接套 PySOD 时质心 .5 的取整差别，E 不重新量化为 uint8 直方图。SciPy EDT 与 MATLAB bwdist 的最近点并列选择可能不同，因此 **Python 通过不等于原版 MATLAB 已执行或逐位一致**。原版 MATLAB 核验入口见 `run_matlab.m`；当前两台机器都未找到 MATLAB/Octave，尚未执行该入口。

论文参考值（表 II，https://arxiv.org/html/2102.10274v2）：

| 数据集 | S↑ | mean E↑ | weighted F↑ | MAE↓ |
|---|---:|---:|---:|---:|
| CAMO-Test | .820 | .882 | .743 | .070 |
| COD10K-Test | .815 | .887 | .680 | .037 |
| CHAMELEON | .888 | .942 | .816 | .030 |

参考值是论文报告，不是本次实测。取得权重后逐项报告实测值、差值及环境，不自动用某个误差阈值宣称“复现成功”。

## 使用

本地只检查数据，不加载权重、不进行训练：

```powershell
.venv/Scripts/python.exe tools/official_sinetv2/verify.py --phase check --data-root Dataset/TestDataset --output outputs/OFFICIAL_SINETV2_VERIFY_20261026
.venv/Scripts/python.exe -m unittest discover -s tests -p test_official_sinetv2_protocol.py -v
```

取得官方权重后，使用新输出目录运行（推理每 100 张输出进度，不附加进程监控）：

```powershell
.venv/Scripts/python.exe tools/official_sinetv2/verify.py --phase all --data-root Dataset/TestDataset --weights model_pth/SINet_V2/Net_epoch_best.pth --output outputs/OFFICIAL_SINETV2_VERIFY_20261026
```

也支持 `--phase infer` / `--phase evaluate` 分开运行。仅完整覆盖每个数据集才生成四指标汇总。输出包含 `provenance.json`、`predictions/`、逐图指标和 `metrics_python.json`；不覆盖旧预测图。

有 MATLAB 环境后，`addpath('tools/official_sinetv2')`，调用 `run_matlab(predictionRoot, testDatasetRoot, upstreamRoot, outputJson)`，在同一批 PNG 上执行作者原始评估函数。作者 `Enhancedmeasure.m` 的主函数名写成 Emeasure，若 MATLAB 版本因此报错，应原样记录兼容性错误，不把未经运行的结果标为 MATLAB 实测。

服务器：项目 `/srv/cod-project`；测试集 `/srv/cod-workspace/Dataset/TestDataset`。独立部署目录 `releases/official_sinetv2_20261026_v1`；结果目录 `outputs/OFFICIAL_SINETV2_VERIFY_20261026_V1`；日志在部署目录 `RUN.log`。运行 GPU 1，四个 CPU 评估进程；没有附加进程监控。Python 使用 cx1，评估依赖复用先前私有 `metric_runtime`，不修改全局环境。已核对运行包版本为 PyTorch 2.10.0+cu128、torchvision 0.25.0+cu128、NumPy 2.0.2、SciPy 1.13.1、Pillow 13.0.0。原始数据、已有实验结果均不改动。
