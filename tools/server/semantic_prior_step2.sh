#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
OUT="$PWD/outputs/SEMANTIC_PRIOR_PILOT_V1"
trap 'result=$?; printf "%s\n" "$result" > "$OUT/step2.exitcode"' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="/srv/cod-workspace/sam3-main/sam3-main${PYTHONPATH:+:$PYTHONPATH}"
echo "Semantic-prior SAM3 candidates started: $(date -Is)"
/srv/cod-workspace/envs/cx1/bin/python -u tools/semantic_prior/segment_tags.py
echo "SEMANTIC_CANDIDATES_DONE: $(date -Is)"
