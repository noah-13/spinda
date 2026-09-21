#!/usr/bin/env bash
# Prepare and train ChaosNLI with the generic text-pair sweep.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/interrupt_cleanup.sh"
cd "$SCRIPT_DIR/.."

DATA_DIR="${DATA_DIR:-data/datasets/text_pair/chaosnli}"
INPUT_DIR="${INPUT_DIR:-data/raw/chaosnli}"
OUT_ROOT="${OUT_ROOT:-outputs/chaosnli}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.json}"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-scripts/run_text_pair_sweep.sh}"
FOLD="${FOLD:-0}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
SUBSETS="${SUBSETS:-snli mnli_m}"
EVALUATE="${EVALUATE:-1}"
FORCE_EVAL="${FORCE_EVAL:-0}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-32}"
PREDICT_MAX_LENGTH="${PREDICT_MAX_LENGTH:-0}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

evaluate_completed_runs() {
  local dataset_dir="$1" run_dir="$2"
  local model_config model_dir seed_dir test_dir predictions evaluation
  # Match Evaluator's default categorical-distribution metrics. The legacy
  # pojsd and soft-F1 metrics are opt-in and are not emitted by this command.
  local required_metrics='accuracy tvd jsd kl ce l2 entropy_correlation distance_correlation'

  while IFS= read -r -d '' model_config; do
    model_dir="${model_config%/config.json}"
    seed_dir="${model_dir%/final_model}"
    test_dir="$seed_dir/test"
    predictions="$test_dir/predictions.json"
    evaluation="$test_dir/evaluation.json"

    if [[ "$FORCE_EVAL" != "1" && -f "$evaluation" ]] && uv run python -c '
import json, sys
required = set(sys.argv[2:])
with open(sys.argv[1], encoding="utf-8") as file:
    metrics = json.load(file)
raise SystemExit(not required.issubset(metrics))
' "$evaluation" $required_metrics; then
      echo "Skipping completed evaluation: $seed_dir"
      continue
    fi
    mkdir -p "$test_dir"
    if [[ ! -s "$predictions" ]]; then
      echo "Generating test predictions: $seed_dir"
      uv run python -m hlv_toolkits.scripts.predict \
        --model_path "$model_dir" --data_dir "$dataset_dir" --split test \
        --device cuda:0 --batch_size "$PREDICT_BATCH_SIZE" --max_length "$PREDICT_MAX_LENGTH" \
        --output_file "$predictions"
    else
      echo "Reusing test predictions: $predictions"
    fi

    echo "Evaluating test predictions: $seed_dir"
    uv run python -m hlv_toolkits.scripts.evaluate \
      --predictions "$predictions" --data_dir "$dataset_dir" --ground_truth_split test \
      --output_file "$evaluation" --no-plot
  done < <(find "$run_dir" -type f -path '*/seed_*/final_model/config.json' -print0 | sort -z)
}

for subset in $SUBSETS; do
  dataset_dir="$DATA_DIR/$subset/$FOLD"
  dataset_config="$DATA_DIR/$subset/$FOLD/dataset.json"
  if [[ "$FORCE_PREPARE" == "1" || ! -s "$dataset_config" || ! -s "$dataset_dir/train.json" || ! -s "$dataset_dir/dev.json" || ! -s "$dataset_dir/test.json" ]]; then
    uv run python -m hlv_toolkits.scripts.prepare_chaosnli_annotation_labels --input_dir "$INPUT_DIR" --output_dir "$DATA_DIR" --subsets "$subset" --fold "$FOLD"
  fi
  DATASET_CONFIG="$dataset_config" RUN_NAME="$subset/fold_$FOLD" OUT_ROOT="$OUT_ROOT" TRAINING_CONFIG="$TRAINING_CONFIG" SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-}" GPU="${GPU:-0}" FORCE="${FORCE:-0}" \
    bash "$SWEEP_SCRIPT"
  if [[ "$EVALUATE" == "1" ]]; then
    evaluate_completed_runs "$dataset_dir" "$OUT_ROOT/$subset/fold_$FOLD"
  fi
done
