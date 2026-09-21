#!/usr/bin/env bash
# Prepare and train the official English-only MultiPICo split.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/interrupt_cleanup.sh"
cd "$SCRIPT_DIR/../.."

DATA_DIR="${MULTIPICO_OUTPUT_DIR:-data/datasets/text_pair/multipico/english}"
INPUT_DIR="${MULTIPICO_INPUT_DIR:-data/raw/multipico}"
OUT_ROOT="${OUT_ROOT:-outputs/multipico/english}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.json}"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-scripts/run_text_pair_sweep.sh}"
# English-only encoders; do not mix these with the multilingual screen.
MODEL_SPECS="${MODEL_SPECS:-microsoft/deberta-v3-large;roberta-base;bert-base-uncased;Twitter/twhin-bert-base}"

if [[ "${FORCE_PREPARE:-0}" == "1" || ! -s "$DATA_DIR/dataset.json" || ! -s "$DATA_DIR/train.json" || ! -s "$DATA_DIR/dev.json" || ! -s "$DATA_DIR/test.json" ]]; then
  uv run python -m hlv_toolkits.scripts.prepare_multipico_annotation_labels \
    --input-dir "$INPUT_DIR" --output-dir "$DATA_DIR" --language en
fi

DATASET_CONFIG="$DATA_DIR/dataset.json" RUN_NAME="${RUN_NAME:-default}" OUT_ROOT="$OUT_ROOT" \
  TRAINING_CONFIG="$TRAINING_CONFIG" MODEL_SPECS="$MODEL_SPECS" \
  RUN_SPECS="${RUN_SPECS:-}" SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-}" \
  GPU="${GPU:-0}" FORCE="${FORCE:-0}" bash "$SWEEP_SCRIPT"
