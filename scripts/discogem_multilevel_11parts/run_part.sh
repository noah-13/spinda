#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; failed=0
for task in "$@"; do
  IFS='|' read -r variant model spec seed <<< "$task"
  data="data/processed/text_pair/discogem/$variant/multilevel/dataset.json"
  if ! MODEL_SPECS="$model" RUN_SPECS="$spec" SEEDS_OVERRIDE="$seed" GPU="${GPU:-0}" FORCE="${FORCE:-0}" DATASET_CONFIG="$data" RUN_NAME="$variant" OUT_ROOT="outputs/discogem/multilevel" HEAD_TYPE="multilevel_classification" bash "$root/scripts/run_text_pair_sweep.sh"; then failed=1; fi
done
exit "$failed"
