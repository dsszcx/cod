#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="/srv/cod-workspace/sam3-main/sam3-main${PYTHONPATH:+:$PYTHONPATH}"
CFG=configs/relation_cam_extra_priors_20261026.yaml
/srv/cod-workspace/envs/cx1/bin/python tools/semantic_prior/extend_cam_pilot.py
timeout 1800 /srv/cod-workspace/envs/cx2/bin/python -u tools/semantic_prior/generate_manifest_tags.py --config "$CFG"
timeout 1800 /srv/cod-workspace/envs/cx1/bin/python -u tools/semantic_prior/segment_tags.py --config "$CFG"
/srv/cod-workspace/envs/cx1/bin/python tools/semantic_prior/pack_regions.py --config "$CFG" --output outputs/RELATION_CAM_EXTENDED_20261026_V1/priors
echo CAM_EXTRA_PRIORS_DONE
