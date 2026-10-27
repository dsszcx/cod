#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
OUT="$PWD/outputs/SEMANTIC_PRIOR_PILOT_V1"
trap 'result=$?; printf "%s\n" "$result" > "$OUT/network_check_v1.exitcode"' EXIT
export CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
/srv/cod-workspace/envs/cx1/bin/python tests/test_region_network.py
/srv/cod-workspace/envs/cx1/bin/python -u tools/semantic_prior/check_region_network.py
echo "REGION_NETWORK_V1_DONE"
