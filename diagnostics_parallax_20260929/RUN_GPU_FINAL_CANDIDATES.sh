#!/usr/bin/env bash
set -euo pipefail

python -B diagnostics_parallax_20260929/scripts/generate_roma_candidates.py \
  --manifest diagnostics_parallax_20260929/configs/delivery_area_final5.json \
  --output diagnostics_parallax_20260929/runs/roma_delivery_area_final5 \
  --device cuda:0 --spacing 8

python -B diagnostics_parallax_20260929/scripts/generate_roma_candidates.py \
  --manifest diagnostics_parallax_20260929/configs/electro_final5.json \
  --output diagnostics_parallax_20260929/runs/roma_electro_final5 \
  --device cuda:0 --spacing 8
