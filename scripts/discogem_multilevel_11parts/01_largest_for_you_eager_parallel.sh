#!/usr/bin/env bash
# One GPU-aware queue across English and multilingual tasks.  A freed slot is
# filled immediately, including by the multilingual InfoXLM task.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
source "$repo_root/scripts/lib/interrupt_cleanup.sh"
cd "$repo_root"

gpu="${GPU:-0}"
max_parallel="${MAX_PARALLEL:-3}"
auto_parallel="${AUTO_PARALLEL:-1}"
min_free_mb="${AUTO_MIN_FREE_MB:-4096}"
warmup_seconds="${AUTO_WARMUP_SECONDS:-20}"
poll_seconds="${AUTO_POLL_SECONDS:-10}"
safety_pct="${AUTO_PEAK_SAFETY_PCT:-125}"
export CUDA_VISIBLE_DEVICES="$gpu"

tasks=(
  'english|microsoft/deberta-v3-large|soft rel|42'
  'english|microsoft/deberta-v3-large|soft rel|43'
  'english|microsoft/deberta-v3-large|soft rel|44'
  'multilingual|microsoft/infoxlm-base|soft mse|43'
)
next=0
failed=0
peak_job_mb=0
last_launch=0

run_task() {
  local variant="$1" model="$2" run_spec="$3" seed="$4"
  local label_mode strategy safe_model output_dir seed_dir final_dir status_file resume=""
  read -r label_mode strategy <<< "$run_spec"
  safe_model="${model//\//_}"
  output_dir="outputs/discogem/multilevel/$variant/${safe_model}__${label_mode}__${strategy}"
  seed_dir="$output_dir/seed_$seed"
  final_dir="$seed_dir/final_model"
  status_file="$seed_dir/run-status.txt"
  if [[ -f "$final_dir/config.json" && "${FORCE:-0}" != "1" ]]; then
    printf 'skipped completed %s\n' "$(date --iso-8601=seconds)" | tee -a "$status_file"
    return 0
  fi
  if [[ "${FORCE:-0}" == "1" ]]; then rm -rf "$seed_dir"; fi
  if [[ -d "$seed_dir" ]]; then
    resume="$(find "$seed_dir" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\t%p\n' | sort -V | tail -n 1 | cut -f 2-)"
  fi
  mkdir -p "$seed_dir"
  printf 'started %s\n' "$(date --iso-8601=seconds)" >> "$status_file"
  local -a resume_arg=()
  [[ -z "$resume" ]] || resume_arg=(--resume_from_checkpoint "$resume")
  uv run python -m hlv_toolkits.scripts.train --config configs/training.json \
    "data/processed/text_pair/discogem/$variant/multilevel/dataset.json" \
    --label_mode "$label_mode" --label_training_strategy "$strategy" \
    --model "$model" --device cuda:0 \
    --output_dir "$output_dir" --seeds "$seed" "${resume_arg[@]}"
  local result=$?
  if (( result == 0 )); then printf 'completed %s\n' "$(date --iso-8601=seconds)" >> "$status_file"; else printf 'failed %s\n' "$(date --iso-8601=seconds)" >> "$status_file"; fi
  return "$result"
}

can_launch() {
  local active now free estimate required
  active=$(jobs -pr | wc -l)
  (( active < max_parallel )) || return 1
  [[ "$auto_parallel" == "1" && "$active" -gt 0 ]] || return 0
  now=$(date +%s)
  (( now - last_launch >= warmup_seconds )) || return 1
  free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9') || return 0
  estimate=$(nvidia-smi -i "$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk 'BEGIN {m=0} /^[0-9]+/ {if ($1>m) m=$1} END {print m}')
  estimate=${estimate:-0}; (( estimate > peak_job_mb )) && peak_job_mb=$estimate
  required=$((min_free_mb + peak_job_mb * safety_pct / 100))
  (( free >= required ))
}

while (( next < ${#tasks[@]} )) || (( $(jobs -pr | wc -l) > 0 )); do
  if (( next < ${#tasks[@]} )) && can_launch; then
    IFS='|' read -r variant model run_spec seed <<< "${tasks[$next]}"
    echo "Launching $variant / $model / $run_spec / seed_$seed"
    (run_task "$variant" "$model" "$run_spec" "$seed") &
    last_launch=$(date +%s)
    ((next += 1))
  elif (( $(jobs -pr | wc -l) > 0 )); then
    if (( $(jobs -pr | wc -l) >= max_parallel )); then wait -n || failed=1; else sleep "$poll_seconds"; fi
  fi
done

exit "$failed"
