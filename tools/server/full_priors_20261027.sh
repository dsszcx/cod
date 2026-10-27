#!/usr/bin/env bash
set -euo pipefail
cd /srv/cod-project/releases/relation_full_priors_20261027_v1
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/srv/cod-workspace/sam3-main/sam3-main
CFG=configs/relation_full_priors_20261027.yaml
PLAN=/srv/cod-project/outputs/RELATION_FULL_PRIOR_PLAN_20261027_V1
EXTRA=()
if [[ "${1:-}" == "smoke" ]]; then EXTRA=(--limit 8); fi
echo "STAGE_TAGS $(date -Is)"
/srv/cod-workspace/envs/cx2/bin/python tools/semantic_prior/generate_manifest_tags.py --config "$CFG" --resume --repair-errors "${EXTRA[@]}"
echo "STAGE_SAM3 $(date -Is)"
/srv/cod-workspace/envs/cx1/bin/python tools/semantic_prior/segment_tags.py --config "$CFG" --resume "${EXTRA[@]}"
echo "STAGE_PACKAGES $(date -Is)"
/srv/cod-workspace/envs/cx1/bin/python tools/semantic_prior/pack_regions.py --config "$CFG" --output "$PLAN/priors" --resume "${EXTRA[@]}"
if [[ "${1:-}" == "smoke" ]]; then
  echo "FULL_PRIORS_SMOKE_DONE $(date -Is)"
else
  /srv/cod-workspace/envs/cx1/bin/python tools/semantic_prior/verify_full_priors.py --plan "$PLAN" --original /srv/cod-project/outputs/RELATION_CAM_ONLY_20261026_V1
fi
