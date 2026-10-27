#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/srv/cod-workspace/envs/cx1/bin/python
OUT=/srv/cod-project/outputs/QUALITY_DECOUPLING_20261026_V1
# Verify all four metrics can run BEFORE spending GPU training time.
PYTHONPATH="$PWD/metric_runtime:$PWD" $PY tests/test_pilot_metrics.py
$PY tests/test_quality_decoupling.py
$PY tools/semantic_prior/make_quality_experiment.py --out "$OUT" \
  --manifests /srv/cod-project/outputs/RELATION_CAM_EXTENDED_20261026_V1 \
  --imagenet /srv/cod-workspace/pretrained/pvt_v2_b2.pth
for seed in 20261022 20261023; do
  for condition in control observer feedback shared; do
    $PY -u tools/semantic_prior/train_pilot.py --config "$OUT/s${seed}_${condition}.yaml"
  done
done
$PY -u tools/semantic_prior/evaluate_candidate_experiment.py --plan "$OUT/PLAN.json"
PYTHONPATH="$PWD/metric_runtime:$PWD" $PY -u tools/semantic_prior/report_quality_experiment.py --root "$OUT"
echo QUALITY_DECOUPLING_EXPERIMENT_DONE
