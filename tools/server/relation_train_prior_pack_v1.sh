#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
OUT="$PWD/outputs/RELATION_TRAIN_PRIORS_PILOT_V1"
trap 'result=$?; printf "%s\n" "$result" > "$OUT/pack.exitcode"' EXIT
export CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
/srv/cod-workspace/envs/cx1/bin/python -u tools/semantic_prior/pack_regions.py --config configs/relation_train_priors_pilot_v1.yaml --output outputs/RELATION_TRAIN_PREP_V1/priors
/srv/cod-workspace/envs/cx1/bin/python -u tools/semantic_prior/check_pilot_prior_batches.py
echo "TRAIN_PRIOR_PACK_AND_BATCH_DONE"
