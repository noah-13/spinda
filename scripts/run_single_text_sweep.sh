#!/usr/bin/env bash
# Train one single-text dataset across the shared model/strategy sweep.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/interrupt_cleanup.sh"
cd "$SCRIPT_DIR/.."

DATASET_CONFIG="${DATASET_CONFIG:?Set DATASET_CONFIG to a dataset.json path.}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME to the dataset-specific output name.}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT to the output root.}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.json}"
GPU="${GPU:-0}"
FORCE="${FORCE:-0}"
STATUS_FILE_NAME="run-status.txt"
if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then
  read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
else
  mapfile -t SEEDS < <(uv run python -c 'import json, sys; [print(seed) for seed in json.load(open(sys.argv[1]))["seeds"]]' "$TRAINING_CONFIG")
fi
export CUDA_VISIBLE_DEVICES="$GPU"
DEVICE="cuda:0"

MODEL_SPECS="${MODEL_SPECS:-microsoft/deberta-v3-large;roberta-base;bert-base-uncased;xlm-roberta-base;Twitter/twhin-bert-base;bert-base-multilingual-cased}"
RUN_SPECS="${RUN_SPECS:-soft ce;soft mse;soft jsd;soft rel;soft_to_hard ce}"
IFS=";" read -r -a MODELS <<< "$MODEL_SPECS"
IFS=";" read -r -a RUNS <<< "$RUN_SPECS"

run() {
  local model="$1" label_mode="$2" strategy="$3" seed="$4"
  local safe_model="${model//\//_}"
  local output_dir="$OUT_ROOT/$RUN_NAME/${safe_model}__${label_mode}__${strategy}"
  local seed_output_dir="$output_dir/seed_${seed}"
  local final_model_dir="$seed_output_dir/final_model"
  local status_file="$seed_output_dir/$STATUS_FILE_NAME"
  local resume_checkpoint=""; local -a resume_args=()
  if [[ "$FORCE" == "1" ]]; then
    rm -rf "$seed_output_dir"
  elif [[ -f "$final_model_dir/config.json" ]]; then
    printf 'skipped completed %s\n' "$(date --iso-8601=seconds)" | tee -a "$status_file"
    echo "Skipping completed run: $output_dir/seed_$seed"; return 0
  elif [[ -d "$seed_output_dir" ]]; then
    resume_checkpoint="$(find "$seed_output_dir" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\t%p\n' | sort -V | tail -n 1 | cut -f 2-)"
  fi
  mkdir -p "$seed_output_dir"
  printf 'started %s\n' "$(date --iso-8601=seconds)" >> "$status_file"
  [[ -z "$resume_checkpoint" ]] || resume_args=(--resume_from_checkpoint "$resume_checkpoint")
  if uv run python -m hlv_toolkits.scripts.train \
    --config "$TRAINING_CONFIG" "$DATASET_CONFIG" \
    --label_mode "$label_mode" --label_training_strategy "$strategy" \
    --model "$model" --device "$DEVICE" --output_dir "$output_dir" --seeds "$seed" "${resume_args[@]}"; then
    printf 'completed %s\n' "$(date --iso-8601=seconds)" >> "$status_file"
  else
    printf 'failed %s\n' "$(date --iso-8601=seconds)" >> "$status_file"; return 1
  fi
}

failed_runs=()
for model in "${MODELS[@]}"; do
  for run_spec in "${RUNS[@]}"; do
    read -r label_mode strategy <<< "$run_spec"
    for seed in "${SEEDS[@]}"; do
      run "$model" "$label_mode" "$strategy" "$seed" || failed_runs+=("$model/$label_mode/$strategy/seed_$seed")
    done
  done
done

if ((${#failed_runs[@]})); then
  printf 'Failed runs for %s:\n  %s\n' "$RUN_NAME" "${failed_runs[@]}"
  exit 1
fi
