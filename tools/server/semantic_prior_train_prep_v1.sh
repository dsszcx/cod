#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
/srv/cod-workspace/envs/cx1/bin/python tests/test_relation_training_data.py
/srv/cod-workspace/envs/cx1/bin/python -u tools/semantic_prior/prepare_training.py
/srv/cod-workspace/envs/cx1/bin/python -u tools/semantic_prior/check_training_data.py
echo "RELATION_TRAIN_PREPARATION_V1_DONE"
