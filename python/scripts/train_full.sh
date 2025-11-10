#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CPU_CORES="$(python - <<'PY'
import os
try:
    print(os.cpu_count() or 1)
except Exception:
    print(1)
PY
)"

export OMP_NUM_THREADS="$CPU_CORES"
export MKL_NUM_THREADS="$CPU_CORES"
export NUMEXPR_NUM_THREADS="$CPU_CORES"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m fx_intraday_ai.cli train \
  --config configs/default.yaml \
  --predictions-path data/cache/predictions.parquet \
  --model-dir artifacts/models \
  "$@"
