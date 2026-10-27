#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
OUT="$PWD/outputs/EMPTY_DIAG_20261022"
mkdir -p "$OUT"
trap 'result=$?; printf "%s\n" "$result" > "$OUT/diag.exitcode"' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="/srv/cod-workspace/sam3-main/sam3-main${PYTHONPATH:+:$PYTHONPATH}"
echo "EMPTY diagnostic started: $(date -Is)"
/srv/cod-workspace/envs/cx1/bin/python -u tools/server/diagnose_empty_sam3.py
echo "EMPTY_DIAG_DONE: $(date -Is)"
