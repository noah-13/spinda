#!/usr/bin/env bash
# Prepare and train the official all-language MultiPICo split.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/interrupt_cleanup.sh"
source "$SCRIPT_DIR/../lib/paper_experiment_defaults.sh"
cd "$SCRIPT_DIR/../.."

DATA_DIR="${MULTIPICO_OUTPUT_DIR:-data/datasets/text_pair/multipico/multilingual}"
INPUT_DIR="${MULTIPICO_INPUT_DIR:-data/raw/multipico}"
OUT_ROOT="${OUT_ROOT:-outputs/multipico/multilingual}"
TRAINING_CONFIG="${TRAINING_CONFIG:-$PAPER_TRAINING_CONFIG}"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-scripts/run_text_pair_sweep.sh}"
# Multilingual encoders only; intentionally distinct from English-only models.
MODEL_SPECS="${MODEL_SPECS:-$PAPER_MULTILINGUAL_MODEL_SPECS}"

if [[ "${FORCE_PREPARE:-0}" == "1" || ! -s "$DATA_DIR/dataset.json" || ! -s "$DATA_DIR/train.json" || ! -s "$DATA_DIR/dev.json" || ! -s "$DATA_DIR/test.json" ]]; then
  uv run python -m hlv_toolkits.scripts.prepare_multipico_annotation_labels \
    --input-dir "$INPUT_DIR" --output-dir "$DATA_DIR"
fi

DATASET_CONFIG="$DATA_DIR/dataset.json" RUN_NAME="${RUN_NAME:-default}" OUT_ROOT="$OUT_ROOT" \
  TRAINING_CONFIG="$TRAINING_CONFIG" MODEL_SPECS="$MODEL_SPECS" \
  RUN_SPECS="${RUN_SPECS:-}" SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-}" \
  GPU="${GPU:-0}" FORCE="${FORCE:-0}" bash "$SWEEP_SCRIPT"
