#!/usr/bin/env bash
# Prepare and train every paper DiscoGeM setting: separate and joint levels.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/interrupt_cleanup.sh"
source "$SCRIPT_DIR/lib/paper_experiment_defaults.sh"
cd "$SCRIPT_DIR/.."

DISCOGEM_DATA_ROOT="${DISCOGEM_DATA_ROOT:-data/datasets/text_pair/discogem}"
OUT_ROOT="${OUT_ROOT:-outputs/discogem}"
TRAINING_CONFIG="${TRAINING_CONFIG:-$PAPER_TRAINING_CONFIG}"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-scripts/run_text_pair_sweep.sh}"
# Space-separated subsets: english, multilingual, or both (the default).
VARIANTS="${VARIANTS:-english multilingual}"
# Space-separated training modes: separate, joint, or both (the default).
MODES="${MODES:-separate joint}"
RUN_SPECS="${RUN_SPECS:-soft ce;soft mse;soft jsd;soft rel;soft_to_hard ce}"

if [[ -n "${LEVEL:-}" && "$MODES" == *joint* ]]; then
  echo "LEVEL only applies to separate training; set MODES=separate to select one level." >&2
  exit 2
fi
if [[ -n "${LEVEL:-}" ]]; then
  LEVELS=("$LEVEL")
else
  LEVELS=(level1 level2 level3)
fi

for variant in $VARIANTS; do
  case "$variant" in
    english) default_models="$PAPER_ENGLISH_MODEL_SPECS" ;;
    multilingual) default_models="$PAPER_MULTILINGUAL_MODEL_SPECS" ;;
    *)
      echo "Unknown VARIANTS value: $variant (expected english and/or multilingual)" >&2
      exit 2
      ;;
  esac
  variant_models="${MODEL_SPECS:-$default_models}"

  for mode in $MODES; do
    case "$mode" in
      separate)
        for level in "${LEVELS[@]}"; do
          dataset_config="$DISCOGEM_DATA_ROOT/$variant/$level/dataset.json"
          if [[ "${FORCE_PREPARE:-0}" == "1" || ! -s "$dataset_config" || ! -s "${dataset_config%/dataset.json}/train.json" || ! -s "${dataset_config%/dataset.json}/dev.json" || ! -s "${dataset_config%/dataset.json}/test.json" ]]; then
            uv run python -m spinda.scripts.prepare_discogem_annotation_labels --variants "$variant"
          fi
          DATASET_CONFIG="$dataset_config" RUN_NAME="$level" OUT_ROOT="$OUT_ROOT/$variant" \
            TRAINING_CONFIG="$TRAINING_CONFIG" MODEL_SPECS="$variant_models" RUN_SPECS="$RUN_SPECS" \
            SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-}" GPU="${GPU:-0}" FORCE="${FORCE:-0}" \
            bash "$SWEEP_SCRIPT"
        done
        ;;
      joint)
        dataset_config="$DISCOGEM_DATA_ROOT/$variant/multilevel/dataset.json"
        if [[ "${FORCE_PREPARE:-0}" == "1" || ! -s "$dataset_config" || ! -s "${dataset_config%/dataset.json}/train.json" || ! -s "${dataset_config%/dataset.json}/dev.json" || ! -s "${dataset_config%/dataset.json}/test.json" ]]; then
          uv run python -m spinda.scripts.prepare_discogem_annotation_labels --variants "$variant"
        fi
        DATASET_CONFIG="$dataset_config" RUN_NAME="$variant" OUT_ROOT="$OUT_ROOT/joint" \
          TRAINING_CONFIG="$TRAINING_CONFIG" MODEL_SPECS="$variant_models" RUN_SPECS="$RUN_SPECS" \
          SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-}" GPU="${GPU:-0}" FORCE="${FORCE:-0}" \
          bash "$SWEEP_SCRIPT"
        ;;
      *)
        echo "Unknown MODES value: $mode (expected separate and/or joint)" >&2
        exit 2
        ;;
    esac
  done
done
