#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/srv/cod-project/releases/quality_decoupling_20261026_v1/metric_runtime:$PWD
exec /srv/cod-workspace/envs/cx1/bin/python -u tools/semantic_prior/train_rgb_full.py \
  --config configs/rgb_cam_full_20261026.yaml "$@"
