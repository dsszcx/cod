#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/srv/cod-workspace/envs/cx1/bin/python
$PY tools/semantic_prior/make_cam_sweep.py
ROOT=outputs/CAM_FUSION_SWEEP_20261026_V1
for seed in 20261022 20261023; do
  for condition in raw_zero raw_small scaled_small; do
    cfg="$ROOT/s${seed}_${condition}.yaml"
    $PY -u tools/semantic_prior/train_pilot.py --config "$cfg"
  done
done
echo CAM_FUSION_SWEEP_TRAINING_DONE
