#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."


MODEL="${MODEL:-roberta-base}"
DEVICE="${DEVICE:-auto}"
SEEDS="${SEEDS:-42}"
DISCOGEM_LABEL_LEVEL="${DISCOGEM_LABEL_LEVEL:-level2}"
SOFT_LABEL_LOSS="${SOFT_LABEL_LOSS:-mse}"
SOFT_LABEL_METRIC="${SOFT_LABEL_METRIC:-tvd}"
SAFE_MODEL="${MODEL//\//_}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/discogem/runs/single/${DISCOGEM_LABEL_LEVEL}/${SAFE_MODEL}__classification__soft_label_loss_${SOFT_LABEL_LOSS}}"

uv run python -m hlv_toolkits.scripts.train \
  --device "$DEVICE" \
  --data_source processed \
  --processed_task discogem \
  --discogem_label_mode soft \
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
  --seeds $SEEDS \
  --use_wandb \
  --wandb_project "${WANDB_PROJECT:-discogem}" \
  --wandb_entity "${WANDB_ENTITY:-noah103374955-ludwig-maximilianuniversity-of-munich}" \
  --wandb_group "${WANDB_GROUP:-single}" \
  --wandb_job_type "${WANDB_JOB_TYPE:-train}" \
  --wandb_run_name "${WANDB_RUN_NAME:-discogem__${DISCOGEM_LABEL_LEVEL:-level2}__${MODEL//\//_}__classification__soft_label_loss_${SOFT_LABEL_LOSS}__seed_${SEEDS}}"
