#!/usr/bin/env bash
# Part 1: frozen baseline -> coarse masks -> ROI/composite (30 images).
set -e
cd "$(dirname "$0")/../.."
ROOT="$PWD"
PYTHON=/srv/cod-workspace/envs/cx1/bin/python
CONFIG=configs/mask_prompt_20261022.yaml
OUT="$ROOT/outputs/MASK_PROMPT_P0_20261022"
mkdir -p "$OUT"
trap 'result=$?; printf "%s\n" "$result" > "$OUT/part1.exitcode"' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
echo "Part 1 started: $(date -Is)"
echo "Loading baseline Python environment..."
"$PYTHON" -u -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)'
"$PYTHON" -u tools/e0e4/build_manifest.py --config "$CONFIG" --out "$OUT"
"$PYTHON" -u tools/e0e4/infer_e0.py --config "$CONFIG" --out "$OUT" --ids "$OUT/manifest_p0.txt" --gpu_id 0
"$PYTHON" -u tools/e0e4/roi_from_mask.py --config "$CONFIG" --out "$OUT" --ids "$OUT/manifest_p0.txt"
echo "PART1_DONE: 30 baseline masks and ROI composites ready. $(date -Is)"
