#!/usr/bin/env bash
# Train paper-compatible DiscoGeM multilevel models for English and multilingual data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/interrupt_cleanup.sh"
source "$SCRIPT_DIR/../lib/paper_experiment_defaults.sh"
cd "$SCRIPT_DIR/../.."

TRAINING_CONFIG="${TRAINING_CONFIG:-$PAPER_TRAINING_CONFIG}"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-scripts/run_text_pair_sweep.sh}"
OUT_ROOT="${OUT_ROOT:-outputs/discogem/multilevel}"
# Space-separated subset: english, multilingual, or both (the default).
VARIANTS="${VARIANTS:-english multilingual}"
RUN_SPECS="${RUN_SPECS:-soft ce;soft mse;soft jsd;soft rel;soft_to_hard ce}"

for variant in $VARIANTS; do
  case "$variant" in
    english)
      dataset_config="data/datasets/text_pair/discogem/english/multilevel/dataset.json"
      default_models="$PAPER_ENGLISH_MODEL_SPECS"
      ;;
    multilingual)
      dataset_config="data/datasets/text_pair/discogem/multilingual/multilevel/dataset.json"
      default_models="$PAPER_MULTILINGUAL_MODEL_SPECS"
      ;;
    *)
      echo "Unknown VARIANTS value: $variant (expected english and/or multilingual)" >&2
      exit 2
      ;;
  esac

  if [[ ! -s "$dataset_config" || ! -s "${dataset_config%/dataset.json}/train.json" || ! -s "${dataset_config%/dataset.json}/dev.json" || ! -s "${dataset_config%/dataset.json}/test.json" ]]; then
    uv run python -m hlv_toolkits.scripts.prepare_discogem_annotation_labels --variants "$variant"
  fi
  variant_models="${MODEL_SPECS:-$default_models}"
  DATASET_CONFIG="$dataset_config" RUN_NAME="$variant" OUT_ROOT="$OUT_ROOT" \
    TRAINING_CONFIG="$TRAINING_CONFIG" MODEL_SPECS="$variant_models" RUN_SPECS="$RUN_SPECS" \
    SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-}" GPU="${GPU:-0}" FORCE="${FORCE:-0}" \
    bash "$SWEEP_SCRIPT"
done
