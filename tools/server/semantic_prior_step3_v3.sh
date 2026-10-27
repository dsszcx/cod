#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
OUT="$PWD/outputs/SEMANTIC_PRIOR_PILOT_V1"
trap 'result=$?; printf "%s\n" "$result" > "$OUT/step3_v3.exitcode"' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "Semantic-prior selection v3 started: $(date -Is)"
/srv/cod-workspace/envs/cx2/bin/python -u tools/semantic_prior/select_candidates_v3.py
echo "SEMANTIC_SELECTION_V3_DONE: $(date -Is)"
