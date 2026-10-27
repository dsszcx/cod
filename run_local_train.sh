#!/bin/bash
# =============================================================================
# 本地端训练+推理脚本 — SAM3 box-prompt pipeline
# 用法: bash run_local_train.sh [--step train|test|all] [--pth path] [--epoch N] [--batch N] [--prior dir]
# =============================================================================

set -e

# ======================== 配置区域 ========================
# 先验数据路径（从服务器下载后的本地路径）
PRIOR_DIR="./prior_sam3"

# 训练配置（与 MyTrain_SAM3.py 的默认值保持一致）
EPOCH=80
BATCH=6
TEST_DATASETS="CAMO,COD10K"

# 权重路径（推理时使用）
PTH_PATH=""

# ======================== 参数解析 ========================
STEP="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --step) STEP="$2"; shift 2 ;;
        --pth)  PTH_PATH="$2"; shift 2 ;;
        --epoch) EPOCH="$2"; shift 2 ;;
        --batch) BATCH="$2"; shift 2 ;;
        --prior) PRIOR_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  SAM3 Training Pipeline (Local)"
echo "  Step: $STEP | Epoch: $EPOCH | Batch: $BATCH"
echo "============================================"

# ======================== 训练 ========================
if [ "$STEP" = "all" ] || [ "$STEP" = "train" ]; then
    echo ""
    echo ">>> [Train] Starting training..."
    echo ">>> Prior dir: $PRIOR_DIR"
    echo ""

    python MyTrain_SAM3.py \
        --epoch $EPOCH \
        --batchsize $BATCH \
        --test_datasets $TEST_DATASETS \
        --prior_root "$PRIOR_DIR"

    echo ""
    echo ">>> [Train] Done! Weights saved in model_pth/SAM3/"
fi

# ======================== 推理 ========================
if [ "$STEP" = "all" ] || [ "$STEP" = "test" ]; then
    if [ -z "$PTH_PATH" ]; then
        # 自动找最新权重
        PTH_PATH=$(ls -t model_pth/SAM3/*.pth 2>/dev/null | head -1)
        if [ -z "$PTH_PATH" ]; then
            echo ">>> [Test] No weights found! Use --pth to specify."
            exit 1
        fi
    fi

    echo ""
    echo ">>> [Test] Running inference..."
    echo ">>> Weights: $PTH_PATH"
    echo ""

    python MyTest_SAM3.py --pth_path "$PTH_PATH" \
        --datasets $TEST_DATASETS \
        --prior_root "$PRIOR_DIR"

    echo ""
    echo ">>> [Test] Done! Results saved in results_sam3/"
fi

echo ""
echo "============================================"
echo "  训练流程完成！"
echo "  查看预测图: results_sam3/"
echo "============================================"
