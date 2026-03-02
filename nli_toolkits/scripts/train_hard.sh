#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-roberta-base}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/hard_snli}"
SEEDS="${SEEDS:-42}"

python -m nli_toolkits.scripts.train \
  --data_source snli \
  --model "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --num_epochs "${NUM_EPOCHS:-3}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --train_batch_size "${TRAIN_BATCH_SIZE:-32}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-64}" \
  --max_length "${MAX_LENGTH:-128}" \
  --seeds $SEEDS
