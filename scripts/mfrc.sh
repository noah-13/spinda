#!/usr/bin/env bash
# Prepare and train MFRC with the generic single-text multi-label sweep.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/interrupt_cleanup.sh"
cd "$SCRIPT_DIR/.."

DATA_DIR="${MFRC_OUTPUT_DIR:-data/processed/single_text/mfrc}"
OUT_ROOT="${OUT_ROOT:-outputs/mfrc}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.json}"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-scripts/run_single_text_sweep.sh}"
RUN_NAME="${RUN_NAME:-default}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"

if [[ "$FORCE_PREPARE" == "1" || ! -s "$DATA_DIR/dataset.json" ]]; then
  uv run python -m hlv_toolkits.scripts.prepare_mfrc_annotation_labels \
    --output-dir "$DATA_DIR"
fi

DATASET_CONFIG="$DATA_DIR/dataset.json" RUN_NAME="$RUN_NAME" OUT_ROOT="$OUT_ROOT" \
  TRAINING_CONFIG="$TRAINING_CONFIG" GPU="${GPU:-0}" FORCE="${FORCE:-0}" \
  \
  MODEL_SPECS="${MODEL_SPECS:-}" RUN_SPECS="${RUN_SPECS:-}" SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-}" \
  bash "$SWEEP_SCRIPT"
