#!/usr/bin/env bash
# Run the largest mixed part: parallel English DeBERTa seeds, then InfoXLM.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
gpu="${GPU:-0}"

DATASET_CONFIG="data/processed/text_pair/discogem/english/multilevel/dataset.json" \
RUN_NAME="english" \
OUT_ROOT="outputs/discogem/multilevel" \
HEAD_TYPE="multilevel_classification" \
MODEL_SPECS="microsoft/deberta-v3-large" \
RUN_SPECS="soft rel" \
SEEDS_OVERRIDE="42 43 44" \
GPU="$gpu" \
MAX_PARALLEL="${MAX_PARALLEL:-3}" \
AUTO_PARALLEL="${AUTO_PARALLEL:-1}" \
bash "$repo_root/scripts/run_text_pair_sweep_parallel.sh"

bash "$script_dir/run_part.sh" \
  'multilingual|microsoft/infoxlm-base|soft mse|43'
