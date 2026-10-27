#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
OUT="$PWD/outputs/RELATION_TRAIN_PRIORS_PILOT_V1"
trap 'result=$?; printf "%s\n" "$result" > "$OUT/sam3.exitcode"' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="/srv/cod-workspace/sam3-main/sam3-main${PYTHONPATH:+:$PYTHONPATH}"
echo "Training pilot per-tag SAM3 started: $(date -Is)"
/srv/cod-workspace/envs/cx1/bin/python -u tools/semantic_prior/segment_tags.py --config configs/relation_train_priors_pilot_v1.yaml
echo "TRAIN_PRIOR_CANDIDATES_DONE: $(date -Is)"
