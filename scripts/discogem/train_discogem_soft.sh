#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-roberta-base}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/discogem/soft}"
PROCESSED_DATA_FILE="${PROCESSED_DATA_FILE:-data/processed/discogem.jsonl}"
SEEDS="${SEEDS:-42}"
SOFT_LABEL_LOSS="${SOFT_LABEL_LOSS:-cross_entropy}"
SOFT_LABEL_METRIC="${SOFT_LABEL_METRIC:-tvd}"

uv run python -m hlv_toolkits.scripts.train \
  --data_source processed \
  --processed_task discogem \
  --processed_data_dir "$PROCESSED_DATA_FILE" \
  --head_type "${HEAD_TYPE:-classification}" \
  --use_soft_labels \
  --soft_label_loss "$SOFT_LABEL_LOSS" \
  --soft_label_metric_for_best_model "$SOFT_LABEL_METRIC" \
  --discogem_label_level "${DISCOGEM_LABEL_LEVEL:-level2}" \
  --model "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --num_epochs "${NUM_EPOCHS:-20}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --train_batch_size "${TRAIN_BATCH_SIZE:-32}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-64}" \
  --max_length "${MAX_LENGTH:-128}" \
  --seeds $SEEDS
