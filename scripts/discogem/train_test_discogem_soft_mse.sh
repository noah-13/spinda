#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

MODEL="${MODEL:-roberta-base}"
SEEDS="${SEEDS:-42}"
DEVICE="${DEVICE:-auto}"
DISCOGEM_LABEL_LEVEL="${DISCOGEM_LABEL_LEVEL:-level2}"
SAFE_MODEL="${MODEL//\//_}"
RUN_DIR="${RUN_DIR:-outputs/discogem/runs/single/${DISCOGEM_LABEL_LEVEL}/${SAFE_MODEL}__classification__soft_label_loss_mse}"

if [[ -e "$RUN_DIR/seed_${SEEDS}" ]]; then
  echo "Output directory already exists: $RUN_DIR/seed_${SEEDS}" >&2
  echo "Set RUN_DIR to a new directory or remove it explicitly before rerunning." >&2
  exit 1
fi

MODEL="$MODEL" DISCOGEM_LABEL_LEVEL="$DISCOGEM_LABEL_LEVEL" OUTPUT_DIR="$RUN_DIR" SEEDS="$SEEDS" DEVICE="$DEVICE" bash "$SCRIPT_DIR/train_discogem_soft_mse.sh"

RUN_DIR="$RUN_DIR" SEEDS="$SEEDS" DISCOGEM_LABEL_LEVEL="$DISCOGEM_LABEL_LEVEL" MODEL_DIR="$RUN_DIR/seed_${SEEDS}/final_model" PREDICTIONS="$RUN_DIR/seed_${SEEDS}/test/predictions.jsonl" RESULTS="$RUN_DIR/seed_${SEEDS}/test/evaluation.json" DEVICE="$DEVICE" bash "$SCRIPT_DIR/test_discogem_soft_mse.sh"
