#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
OUT="$PWD/outputs/QWEN_NO_EXAMPLE_20261022"
trap 'result=$?; printf "%s\n" "$result" > "$OUT/sam3_v2.exitcode"' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="/srv/cod-workspace/sam3-main/sam3-main${PYTHONPATH:+:$PYTHONPATH}"
echo "Revised Qwen text -> SAM3 started: $(date -Is)"
/srv/cod-workspace/envs/cx1/bin/python -u tools/server/qwen_no_example_segment.py
echo "QWEN_V2_SAM3_DONE: $(date -Is)"
