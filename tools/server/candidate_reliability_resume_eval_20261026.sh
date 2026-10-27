#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/srv/cod-workspace/envs/cx1/bin/python
PLAN=/srv/cod-project/outputs/CANDIDATE_RELIABILITY_20261026_V1/PLAN.json
$PY tests/test_evaluation_runtime.py
$PY -u tools/semantic_prior/evaluate_candidate_experiment.py --plan "$PLAN"
$PY -u tools/semantic_prior/probe_relation_utility.py --plan "$PLAN" --max-edges 12
echo CANDIDATE_RELIABILITY_EXPERIMENT_DONE
