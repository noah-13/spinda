#!/usr/bin/env bash
# Train TGeGUM genre/topic tasks separately and with a shared three-head model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/interrupt_cleanup.sh"
cd "$SCRIPT_DIR/.."

INPUT_DIR="${HUMANS_AND_DOMAINS_INPUT_DIR:-data/raw/humans_and_domains}"
SINGLE_TEXT_ROOT="${HUMANS_AND_DOMAINS_SINGLE_TEXT_OUTPUT_ROOT:-data/datasets/single_text}"
OUT_ROOT="${OUT_ROOT:-outputs/humans_and_domains}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.json}"
SINGLE_SWEEP="${SINGLE_SWEEP:-scripts/run_single_text_sweep.sh}"
TASKS="${TASKS:-genre topic1 topic2}"
MODEL_SPECS="${MODEL_SPECS:-microsoft/deberta-v3-large;roberta-base;bert-base-uncased;Twitter/twhin-bert-base}"
RUN_SPECS="${RUN_SPECS:-soft ce;soft mse;soft jsd;soft rel;soft_to_hard ce}"

dataset_config="$SINGLE_TEXT_ROOT/humans_and_domains/genre/dataset.json"
if [[ "${FORCE_PREPARE:-0}" == "1" || ! -s "$dataset_config" ]]; then
  uv run python -m hlv_toolkits.scripts.prepare_humans_and_domains_annotation_labels \
    --input-dir "$INPUT_DIR" \
    --single-text-output-root "$SINGLE_TEXT_ROOT"
fi

for task in $TASKS; do
  DATASET_CONFIG="$SINGLE_TEXT_ROOT/humans_and_domains/$task/dataset.json" RUN_NAME="$task" OUT_ROOT="$OUT_ROOT" \
    TRAINING_CONFIG="$TRAINING_CONFIG" MODEL_SPECS="$MODEL_SPECS" RUN_SPECS="$RUN_SPECS" \
    SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-}" GPU="${GPU:-0}" FORCE="${FORCE:-0}" bash "$SINGLE_SWEEP"
done

DATASET_CONFIG="$SINGLE_TEXT_ROOT/humans_and_domains/multilevel/dataset.json" RUN_NAME="multilevel" OUT_ROOT="$OUT_ROOT" \
  TRAINING_CONFIG="$TRAINING_CONFIG" MODEL_SPECS="$MODEL_SPECS" RUN_SPECS="$RUN_SPECS" \
  SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-}" GPU="${GPU:-0}" FORCE="${FORCE:-0}" \
  bash "$SINGLE_SWEEP"
