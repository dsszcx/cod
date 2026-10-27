#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/srv/cod-project/releases/quality_decoupling_20261026_v1/metric_runtime:$PWD
PY=/srv/cod-workspace/envs/cx1/bin/python
OUT=/srv/cod-project/outputs/RGB_FULL_DATA_ACCEPTANCE_20261026_V1
$PY tests/test_rgb_full_manifest.py
$PY tests/test_pilot_metrics.py
$PY tools/semantic_prior/make_rgb_full_acceptance.py --out "$OUT" \
  --manifests /srv/cod-project/outputs/RELATION_CAM_ONLY_20261026_V1 \
  --imagenet /srv/cod-workspace/pretrained/pvt_v2_b2.pth
$PY -u tools/semantic_prior/train_pilot.py --config "$OUT/CONFIG.yaml"
$PY -u tools/semantic_prior/report_rgb_full_acceptance.py --root "$OUT"
echo RGB_FULL_ACCEPTANCE_BATCH_DONE
