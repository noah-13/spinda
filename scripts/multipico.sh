#!/usr/bin/env bash
# Prepare and train every paper MultiPICo setting.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/interrupt_cleanup.sh"
source "$SCRIPT_DIR/lib/paper_experiment_defaults.sh"
cd "$SCRIPT_DIR/.."

MULTIPICO_DATA_ROOT="${MULTIPICO_DATA_ROOT:-data/datasets/text_pair/multipico}"
MULTIPICO_INPUT_DIR="${MULTIPICO_INPUT_DIR:-data/raw/multipico}"
OUT_ROOT="${OUT_ROOT:-outputs/multipico}"
TRAINING_CONFIG="${TRAINING_CONFIG:-$PAPER_TRAINING_CONFIG}"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-scripts/run_text_pair_sweep.sh}"
# Space-separated subset: english, multilingual, or both (the default).
VARIANTS="${VARIANTS:-english multilingual}"

for variant in $VARIANTS; do
  case "$variant" in
    english)
      prepare_args=(--language en)
      default_models="$PAPER_ENGLISH_MODEL_SPECS"
      ;;
    multilingual)
      prepare_args=()
      default_models="$PAPER_MULTILINGUAL_MODEL_SPECS"
      ;;
    *)
      echo "Unknown VARIANTS value: $variant (expected english and/or multilingual)" >&2
      exit 2
      ;;
  esac

  data_dir="$MULTIPICO_DATA_ROOT/$variant"
  dataset_config="$data_dir/dataset.json"
  if [[ "${FORCE_PREPARE:-0}" == "1" || ! -s "$dataset_config" || ! -s "$data_dir/train.json" || ! -s "$data_dir/dev.json" || ! -s "$data_dir/test.json" ]]; then
    uv run python -m spinda.scripts.prepare_multipico_annotation_labels \
      --input-dir "$MULTIPICO_INPUT_DIR" --output-dir "$data_dir" "${prepare_args[@]}"
  fi

  variant_models="${MODEL_SPECS:-$default_models}"
  DATASET_CONFIG="$dataset_config" RUN_NAME="$variant" OUT_ROOT="$OUT_ROOT" \
    TRAINING_CONFIG="$TRAINING_CONFIG" MODEL_SPECS="$variant_models" \
    RUN_SPECS="${RUN_SPECS:-}" SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-}" \
    GPU="${GPU:-0}" FORCE="${FORCE:-0}" bash "$SWEEP_SCRIPT"
done
