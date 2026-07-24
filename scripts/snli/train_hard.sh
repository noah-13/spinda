#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."


MODEL="${MODEL:-roberta-base}"
DEVICE="${DEVICE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/snli/hard}"
SEEDS="${SEEDS:-42}"

uv run python -m hlv_toolkits.scripts.train \
  --device "$DEVICE" \
  --data_source processed \
  --model "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --num_epochs "${NUM_EPOCHS:-3}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --train_batch_size "${TRAIN_BATCH_SIZE:-32}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-64}" \
  --max_length "${MAX_LENGTH:-128}" \
  --seeds $SEEDS
