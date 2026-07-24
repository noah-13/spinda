#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."


MODEL="${MODEL:-roberta-base}"
DEVICE="${DEVICE:-auto}"
SEEDS="${SEEDS:-42}"
DISCOGEM_LABEL_LEVEL="${DISCOGEM_LABEL_LEVEL:-level2}"
SAFE_MODEL="${MODEL//\//_}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/discogem/runs/single/${DISCOGEM_LABEL_LEVEL}/${SAFE_MODEL}__classification__hard}"

uv run python -m hlv_toolkits.scripts.train \
  --device "$DEVICE" \
  --data_source processed \
  --processed_task discogem \
  --discogem_label_mode hard \
  --head_type "${HEAD_TYPE:-classification}" \
  --discogem_label_level "$DISCOGEM_LABEL_LEVEL" \
  --model "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --num_epochs "${NUM_EPOCHS:-20}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --train_batch_size "${TRAIN_BATCH_SIZE:-32}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-64}" \
  --max_length "${MAX_LENGTH:-128}" \
  --seeds $SEEDS
