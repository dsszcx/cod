#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1
OUT="$PWD/outputs/RELATION_SMOKE_20261026_V1"
trap 'result=$?; printf "%s\n" "$result" > "$PWD/outputs/relation_smoke_20261026.exitcode"' EXIT
/srv/cod-workspace/envs/cx1/bin/python -u tools/semantic_prior/train_pilot.py
echo "RELATION_SMOKE_WORK_DONE"
