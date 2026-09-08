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
HEAD_TYPE="${HEAD_TYPE:-}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
AUTO_PARALLEL="${AUTO_PARALLEL:-1}"
AUTO_MIN_FREE_MB="${AUTO_MIN_FREE_MB:-6144}"
AUTO_TARGET_GPU_UTIL="${AUTO_TARGET_GPU_UTIL:-101}"
AUTO_WARMUP_SECONDS="${AUTO_WARMUP_SECONDS:-20}"
AUTO_POLL_SECONDS="${AUTO_POLL_SECONDS:-10}"
# Reserve 125% of the largest observed task allocation for the next task.
AUTO_PEAK_SAFETY_PCT="${AUTO_PEAK_SAFETY_PCT:-125}"
AUTO_PEAK_JOB_MB=0
AUTO_LAST_LAUNCH=0
STATUS_FILE_NAME="run-status.txt"
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_PARALLEL must be positive." >&2; exit 2; }
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
  local resume_checkpoint=""; local -a resume_args=() head_args=()
  [[ -z "$HEAD_TYPE" ]] || head_args=(--head_type "$HEAD_TYPE")
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
    --model "$model" --device "$DEVICE" --output_dir "$output_dir" --seeds "$seed" "${head_args[@]}" "${resume_args[@]}"; then
    printf 'completed %s\n' "$(date --iso-8601=seconds)" >> "$status_file"
  else
    printf 'failed %s\n' "$(date --iso-8601=seconds)" >> "$status_file"; return 1
  fi
}

skip_completed() {
  local model="$1" label_mode="$2" strategy="$3" seed="$4"
  local safe_model="${model//\//_}"
  local seed_output_dir="$OUT_ROOT/$RUN_NAME/${safe_model}__${label_mode}__${strategy}/seed_${seed}"
  local final_model_dir="$seed_output_dir/final_model"
  local status_file="$seed_output_dir/$STATUS_FILE_NAME"
  if [[ "$FORCE" != "1" && -f "$final_model_dir/config.json" ]]; then
    printf 'skipped completed %s\n' "$(date --iso-8601=seconds)" | tee -a "$status_file"
    echo "Skipping completed run: $seed_output_dir"
    return 0
  fi
  return 1
}

wait_for_slot() {
  local active free util metrics estimate required now
  while :; do
    active=$(jobs -pr | wc -l)
    if (( active >= MAX_PARALLEL )); then
      wait -n || failed=1
      continue
    fi
    [[ "$AUTO_PARALLEL" == "1" ]] || return 0
    (( active == 0 )) && return 0
    now=$(date +%s)
    if (( now - AUTO_LAST_LAUNCH < AUTO_WARMUP_SECONDS )); then
      sleep "$AUTO_POLL_SECONDS"
      continue
    fi
    metrics=$(nvidia-smi -i "$GPU" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null) || { AUTO_PARALLEL=0; return 0; }
    IFS="," read -r free util <<< "$metrics"
    free=${free//[[:space:]]/}; util=${util//[[:space:]]/}
    estimate=$(nvidia-smi -i "$GPU" --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk 'BEGIN { m=0 } /^[0-9]+/ { if ($1>m) m=$1 } END { print m }')
    estimate=${estimate:-0}
    if (( estimate > AUTO_PEAK_JOB_MB )); then AUTO_PEAK_JOB_MB=$estimate; fi
    required=$(( AUTO_MIN_FREE_MB + AUTO_PEAK_JOB_MB * AUTO_PEAK_SAFETY_PCT / 100 ))
    if (( free >= required && util < AUTO_TARGET_GPU_UTIL )); then return 0; fi
    sleep "$AUTO_POLL_SECONDS"
  done
}

failed=0
for model in "${MODELS[@]}"; do
  for run_spec in "${RUNS[@]}"; do
    read -r label_mode strategy <<< "$run_spec"
    for seed in "${SEEDS[@]}"; do
      # Do not make an already-completed run wait for an available GPU slot.
      if skip_completed "$model" "$label_mode" "$strategy" "$seed"; then
        continue
      fi
      wait_for_slot
      echo "Launching ($(( $(jobs -pr | wc -l) + 1 ))/$MAX_PARALLEL): $model/$label_mode/$strategy/seed_$seed"
      (run "$model" "$label_mode" "$strategy" "$seed") &
      AUTO_LAST_LAUNCH=$(date +%s)
    done
  done
done
while (( $(jobs -pr | wc -l) > 0 )); do wait -n || failed=1; done
(( failed == 0 )) || exit 1
