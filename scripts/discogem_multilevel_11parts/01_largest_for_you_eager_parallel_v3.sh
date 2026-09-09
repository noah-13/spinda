#!/usr/bin/env bash
# Cross-dataset queue.  Never launch a second job before the first has created
# a CUDA allocation, so simultaneous NFS/model initialisation cannot flood one GPU.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
source "$repo_root/scripts/lib/interrupt_cleanup.sh"
cd "$repo_root"

mkdir -p outputs/discogem/multilevel/logs
exec 9>outputs/discogem/multilevel/logs/01_largest_for_you_eager_parallel.lock
flock -n 9 || { echo "This eager part is already running." >&2; exit 1; }

gpu="${GPU:-0}"; max_parallel="${MAX_PARALLEL:-3}"
min_free_mb="${AUTO_MIN_FREE_MB:-4096}"; poll_seconds="${AUTO_POLL_SECONDS:-5}"
safety_pct="${AUTO_PEAK_SAFETY_PCT:-125}"
export CUDA_VISIBLE_DEVICES="$gpu"
tasks=(
  'english|microsoft/deberta-v3-large|soft rel|42'
  'english|microsoft/deberta-v3-large|soft rel|43'
  'english|microsoft/deberta-v3-large|soft rel|44'
  'multilingual|microsoft/infoxlm-base|soft mse|43'
)
next=0; failed=0; peak_job_mb=0

run_task() {
  local variant="$1" model="$2" spec="$3" seed="$4"
  local label_mode strategy safe_model output_dir seed_dir final_dir status_file log_file resume=""
  read -r label_mode strategy <<< "$spec"
  safe_model="${model//\//_}"
  output_dir="outputs/discogem/multilevel/$variant/${safe_model}__${label_mode}__${strategy}"
  seed_dir="$output_dir/seed_$seed"; final_dir="$seed_dir/final_model"
  status_file="$seed_dir/run-status.txt"; log_file="$seed_dir/train.log"
  if [[ "${FORCE:-0}" != "1" && -f "$final_dir/config.json" ]]; then
    printf 'skipped completed %s\n' "$(date --iso-8601=seconds)" | tee -a "$status_file"; return 0
  fi
  [[ "${FORCE:-0}" != "1" ]] || rm -rf "$seed_dir"
  if [[ -d "$seed_dir" ]]; then
    resume="$(find "$seed_dir" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\t%p\n' | sort -V | tail -n 1 | cut -f 2-)"
  fi
  mkdir -p "$seed_dir"; printf 'started %s\n' "$(date --iso-8601=seconds)" >> "$status_file"
  local -a resume_arg=(); [[ -z "$resume" ]] || resume_arg=(--resume_from_checkpoint "$resume")
  if uv run python -m hlv_toolkits.scripts.train --config configs/training.json \
    "data/processed/text_pair/discogem/$variant/multilevel/dataset.json" \
    --label_mode "$label_mode" --label_training_strategy "$strategy" \
    --head_type multilevel_classification --model "$model" --device cuda:0 \
    --output_dir "$output_dir" --seeds "$seed" "${resume_arg[@]}" 2>&1 | tee -a "$log_file"; then
    printf 'completed %s\n' "$(date --iso-8601=seconds)" >> "$status_file"
  else
    printf 'failed %s\n' "$(date --iso-8601=seconds)" >> "$status_file"; return 1
  fi
}

can_launch() {
  local active free observed required
  active=$(jobs -pr | wc -l); (( active < max_parallel )) || return 1
  (( active == 0 )) && return 0
  observed=$(nvidia-smi -i "$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk 'BEGIN {m=0} /^[0-9]+/ {if ($1>m) m=$1} END {print m}')
  observed=${observed:-0}
  # No CUDA allocation yet: keep exactly one initializer alive.
  (( observed > 0 )) || return 1
  (( observed > peak_job_mb )) && peak_job_mb=$observed
  free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9') || return 1
  required=$((min_free_mb + peak_job_mb * safety_pct / 100))
  (( free >= required ))
}

while (( next < ${#tasks[@]} )) || (( $(jobs -pr | wc -l) > 0 )); do
  if (( next < ${#tasks[@]} )) && can_launch; then
    IFS='|' read -r variant model spec seed <<< "${tasks[$next]}"
    echo "Launching $variant / $model / $spec / seed_$seed"
    (run_task "$variant" "$model" "$spec" "$seed") &
    ((next += 1))
  elif (( $(jobs -pr | wc -l) > 0 )); then
    if (( $(jobs -pr | wc -l) >= max_parallel )); then wait -n || failed=1; else sleep "$poll_seconds"; fi
  fi
done
exit "$failed"
