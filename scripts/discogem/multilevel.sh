#!/usr/bin/env bash
# Train paper-compatible DiscoGeM multilevel models for English and multilingual data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/interrupt_cleanup.sh"
cd "$SCRIPT_DIR/../.."

TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.json}"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-scripts/run_text_pair_sweep.sh}"
OUT_ROOT="${OUT_ROOT:-outputs/discogem/multilevel}"
# Space-separated subset: english, multilingual, or both (the default).
VARIANTS="${VARIANTS:-english multilingual}"
RUN_SPECS="${RUN_SPECS:-soft ce;soft mse;soft jsd;soft rel;hard ce}"

for variant in $VARIANTS; do
  case "$variant" in
    english)
      dataset_config="data/processed/text_pair/discogem/english/multilevel/dataset.json"
      default_models="microsoft/deberta-v3-large;roberta-base;bert-base-uncased;xlm-roberta-base;Twitter/twhin-bert-base;bert-base-multilingual-cased"
      ;;
    multilingual)
      dataset_config="data/processed/text_pair/discogem/multilingual/multilevel/dataset.json"
      default_models="xlm-roberta-base;bert-base-multilingual-cased;microsoft/infoxlm-base"
      ;;
    *)
      echo "Unknown VARIANTS value: $variant (expected english and/or multilingual)" >&2
      exit 2
      ;;
  esac

  if [[ ! -s "$dataset_config" ]]; then
    uv run python -m hlv_toolkits.scripts.prepare_discogem_annotation_labels --variants "$variant"
  fi
  variant_models="${MODEL_SPECS:-$default_models}"
  DATASET_CONFIG="$dataset_config" RUN_NAME="$variant" OUT_ROOT="$OUT_ROOT" \
    TRAINING_CONFIG="$TRAINING_CONFIG" MODEL_SPECS="$variant_models" RUN_SPECS="$RUN_SPECS" \
    HEAD_TYPE="multilevel_classification" SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-}" GPU="${GPU:-0}" FORCE="${FORCE:-0}" \
    bash "$SWEEP_SCRIPT"
done
