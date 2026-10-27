#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
OUT="$PWD/outputs/RELATION_TRAIN_PRIORS_PILOT_V1"
mkdir -p "$OUT"
trap 'result=$?; printf "%s\n" "$result" > "$OUT/tags.exitcode"' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "Training pilot component tags started: $(date -Is)"
/srv/cod-workspace/envs/cx2/bin/python -u tools/semantic_prior/generate_manifest_tags.py
echo "TRAIN_PRIOR_TAGS_DONE: $(date -Is)"
