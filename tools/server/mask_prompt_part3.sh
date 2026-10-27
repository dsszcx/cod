#!/usr/bin/env bash
# Part 3: SAM3 box/text branches, then fixed E4 fusion.
set -e
cd "$(dirname "$0")/../.."
ROOT="$PWD"
PYTHON=/srv/cod-workspace/envs/cx1/bin/python
CONFIG=configs/mask_prompt_20261022.yaml
OUT="$ROOT/outputs/MASK_PROMPT_P0_20261022"
trap 'result=$?; printf "%s\n" "$result" > "$OUT/part3.exitcode"' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="/srv/cod-workspace/sam3-main/sam3-main${PYTHONPATH:+:$PYTHONPATH}"
echo "Part 3 started: $(date -Is)"
echo "Loading SAM3 once; E1=baseline ROI boxes, E2=original-image text, E3=ROI-composite text."
"$PYTHON" -u tools/e0e4/run_e0e4.py --config "$CONFIG" --out "$OUT" --split p0 --stage sam3 --gpu_id 0 --overwrite
echo "SAM3 branches finished; fusing E1 and E3. $(date -Is)"
"$PYTHON" -u tools/e0e4/run_e0e4.py --config "$CONFIG" --out "$OUT" --split p0 --stage e4 --gpu_id 0
echo "PART3_DONE: SAM3 predictions and E4 fusion ready for evaluation. $(date -Is)"
