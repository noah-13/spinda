#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-roberta-base}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/soft_chaosnli}"
SEEDS="${SEEDS:-42}"
CHAOS_TRAIN_JSONL="${CHAOS_TRAIN_JSONL:-chaosNLI_v1.0/chaosNLI_snli_train.jsonl}"
CHAOS_DEV_JSONL="${CHAOS_DEV_JSONL:-chaosNLI_v1.0/chaosNLI_snli_dev.jsonl}"
SOFT_LABEL_LOSS="${SOFT_LABEL_LOSS:-cross_entropy}"
SOFT_LABEL_METRIC="${SOFT_LABEL_METRIC:-tvd}"


python -m nli_toolkits.scripts.train \
  --data_source chaosnli \
  --chaosnli_train_path "$CHAOS_TRAIN_JSONL" \
  --chaosnli_dev_path "$CHAOS_DEV_JSONL" \
  --use_soft_labels \
  --soft_label_loss "$SOFT_LABEL_LOSS" \
  --soft_label_metric_for_best_model "$SOFT_LABEL_METRIC" \
  --model "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --num_epochs "${NUM_EPOCHS:-20}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --train_batch_size "${TRAIN_BATCH_SIZE:-32}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-64}" \
  --max_length "${MAX_LENGTH:-128}" \
  --seeds $SEEDS \
  --use_wandb \
