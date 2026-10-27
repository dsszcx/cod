#!/usr/bin/env bash
set -euo pipefail
RELEASE=/srv/cod-project/releases/official_sinetv2_20261026_v1
OUTPUT=/srv/cod-project/outputs/OFFICIAL_SINETV2_VERIFY_20261026_V1
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/srv/cod-project/releases/quality_decoupling_20261026_v1/metric_runtime
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "$RELEASE"
/srv/cod-workspace/envs/cx1/bin/python -u official_sinetv2/verify.py \
  --phase all --data-root /srv/cod-workspace/Dataset/TestDataset \
  --upstream "$RELEASE/SINet-V2" --weights "$RELEASE/Net_epoch_best.pth" \
  --output "$OUTPUT" --device cuda:0 --workers 4 2>&1 | tee "$RELEASE/RUN.log"
