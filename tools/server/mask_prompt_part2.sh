#!/usr/bin/env bash
# Part 2: Qwen descriptions for original images and baseline ROI composites.
set -e
cd "$(dirname "$0")/../.."
ROOT="$PWD"
PYTHON=/srv/cod-workspace/envs/cx2/bin/python
CONFIG=configs/mask_prompt_20261022.yaml
OUT="$ROOT/outputs/MASK_PROMPT_P0_20261022"
trap 'result=$?; printf "%s\n" "$result" > "$OUT/part2.exitcode"' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "Part 2 started: $(date -Is)"
echo "Loading Qwen; E2=original, E3=baseline ROI composite, 30 images each."
"$PYTHON" -u tools/e0e4/run_e0e4.py --config "$CONFIG" --out "$OUT" --split p0 --stage qwen --gpu_id 0
echo "PART2_DONE: Qwen descriptions ready for SAM3. $(date -Is)"
