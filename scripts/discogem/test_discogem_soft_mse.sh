#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

MODEL="${MODEL:-roberta-base}"
SAFE_MODEL="${MODEL//\//_}"
SEEDS="${SEEDS:-42}"
DISCOGEM_LABEL_LEVEL="${DISCOGEM_LABEL_LEVEL:-level2}"
RUN_DIR="${RUN_DIR:-outputs/discogem/runs/single/${DISCOGEM_LABEL_LEVEL}/${SAFE_MODEL}__classification__soft_label_loss_mse}"
MODEL_DIR="${MODEL_DIR:-$RUN_DIR/seed_${SEEDS}/final_model}"
PREDICTIONS="${PREDICTIONS:-$RUN_DIR/seed_${SEEDS}/test/predictions.jsonl}"
RESULTS="${RESULTS:-$RUN_DIR/seed_${SEEDS}/test/evaluation.json}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LENGTH="${MAX_LENGTH:-128}"

uv run python -m hlv_toolkits.scripts.predict \
  --model_path "$MODEL_DIR" \
  --data_source discogem \
  --split test \
  --discogem_label_level "$DISCOGEM_LABEL_LEVEL" \
  --discogem_label_mode soft \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --max_length "$MAX_LENGTH" \
  --output_file "$PREDICTIONS"

uv run python -m hlv_toolkits.scripts.evaluate \
  --predictions "$PREDICTIONS" \
  --ground_truth_source discogem \
  --ground_truth_split test \
  --discogem_label_level "$DISCOGEM_LABEL_LEVEL" \
  --discogem_label_mode soft \
  --output_file "$RESULTS" \
  --no-plot \
  --no-ternary_browser

echo "Test results saved to: $RESULTS"
