#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/srv/cod-workspace/envs/cx1/bin/python
OUT=/srv/cod-project/outputs/CANDIDATE_RELIABILITY_20261026_V1
$PY tools/semantic_prior/make_candidate_experiment.py --out "$OUT" \
  --manifests /srv/cod-project/outputs/RELATION_CAM_EXTENDED_20261026_V1 \
  --imagenet /srv/cod-workspace/pretrained/pvt_v2_b2.pth --epochs 10 --minutes 50
for seed in 20261022 20261023; do
  for condition in control purity; do
    $PY -u tools/semantic_prior/train_pilot.py --config "$OUT/s${seed}_${condition}.yaml"
  done
done
$PY -u tools/semantic_prior/evaluate_candidate_experiment.py --plan "$OUT/PLAN.json"
echo CANDIDATE_RELIABILITY_EXPERIMENT_DONE
