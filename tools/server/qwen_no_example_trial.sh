#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
OUT="$PWD/outputs/QWEN_NO_EXAMPLE_20261022"
mkdir -p "$OUT"
trap 'result=$?; printf "%s\n" "$result" > "$OUT/trial.exitcode"' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# Same thread setting as the old Qwen run for this prompt-only comparison.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "Qwen no-category-example trial started: $(date -Is)"
/srv/cod-workspace/envs/cx2/bin/python -u tools/server/qwen_no_example_trial.py
echo "QWEN_NO_EXAMPLE_DONE: $(date -Is)"
