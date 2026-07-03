#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-roberta-base}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/discogem/hard}"
PROCESSED_DATA_FILE="${PROCESSED_DATA_FILE:-data/processed/discogem.jsonl}"
SEEDS="${SEEDS:-42}"

uv run python -m hlv_toolkits.scripts.train \
  --data_source processed \
  --processed_task discogem \
  --processed_data_dir "$PROCESSED_DATA_FILE" \
  --head_type "${HEAD_TYPE:-classification}" \
  --discogem_label_level "${DISCOGEM_LABEL_LEVEL:-level2}" \
  --model "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --num_epochs "${NUM_EPOCHS:-20}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --train_batch_size "${TRAIN_BATCH_SIZE:-32}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-64}" \
  --max_length "${MAX_LENGTH:-128}" \
  --seeds $SEEDS
