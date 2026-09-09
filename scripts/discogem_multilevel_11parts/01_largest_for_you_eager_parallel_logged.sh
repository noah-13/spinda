#!/usr/bin/env bash
# Keep a durable transcript for the cross-dataset eager scheduler.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
log_dir="$repo_root/outputs/discogem/multilevel/logs"
log_file="$log_dir/01_largest_for_you_eager_parallel.log"
mkdir -p "$log_dir"

echo "=== started $(date --iso-8601=seconds) ===" | tee -a "$log_file"
set -o pipefail
bash "$script_dir/01_largest_for_you_eager_parallel.sh" 2>&1 | tee -a "$log_file"
