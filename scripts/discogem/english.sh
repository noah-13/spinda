#!/usr/bin/env bash
# Prepare and train paper-compatible English DiscoGeM with the generic sweep.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/interrupt_cleanup.sh"
source "$SCRIPT_DIR/../lib/paper_experiment_defaults.sh"
cd "$SCRIPT_DIR/../.."

OUT_ROOT="${OUT_ROOT:-outputs/discogem/english}"
TRAINING_CONFIG="${TRAINING_CONFIG:-$PAPER_TRAINING_CONFIG}"
MODEL_SPECS="${MODEL_SPECS:-$PAPER_ENGLISH_MODEL_SPECS}"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-scripts/run_text_pair_sweep.sh}"
if [[ -n "${LEVEL:-}" ]]; then LEVELS=("$LEVEL"); else LEVELS=(level1 level2 level3); fi

for level in "${LEVELS[@]}"; do
  DATASET_CONFIG="data/datasets/text_pair/discogem/english/$level/dataset.json"
  if [[ ! -s "$DATASET_CONFIG" || ! -s "${DATASET_CONFIG%/dataset.json}/train.json" || ! -s "${DATASET_CONFIG%/dataset.json}/dev.json" || ! -s "${DATASET_CONFIG%/dataset.json}/test.json" ]]; then
    uv run python -m hlv_toolkits.scripts.prepare_discogem_annotation_labels --variants english
  fi
  DATASET_CONFIG="$DATASET_CONFIG" RUN_NAME="$level" OUT_ROOT="$OUT_ROOT" TRAINING_CONFIG="$TRAINING_CONFIG" MODEL_SPECS="$MODEL_SPECS" \
    SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-}" GPU="${GPU:-0}" FORCE="${FORCE:-0}" \
    bash "$SWEEP_SCRIPT"
done
