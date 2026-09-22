#!/usr/bin/env bash
# Complete the missing InfoXLM-base MD-Agreement paper runs on one GPU.
# The underlying scheduler resumes valid checkpoints and skips finalised seeds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/interrupt_cleanup.sh"
cd "$SCRIPT_DIR/.."

MODEL_SPECS="${MODEL_SPECS:-microsoft/infoxlm-base}"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-scripts/run_single_text_sweep_parallel.sh}"

# AUTO_PARALLEL launches more jobs only while the current GPU has enough free
# memory and is below the utilization target. MAX_PARALLEL is a ceiling, not a
# forced concurrency level, so this safely fills GPUs with different capacity.
MAX_PARALLEL="${MAX_PARALLEL:-20}"
AUTO_PARALLEL="${AUTO_PARALLEL:-1}"
AUTO_MIN_FREE_MB="${AUTO_MIN_FREE_MB:-6144}"
AUTO_PEAK_SAFETY_PCT="${AUTO_PEAK_SAFETY_PCT:-125}"

MODEL_SPECS="$MODEL_SPECS" SWEEP_SCRIPT="$SWEEP_SCRIPT" \
  MAX_PARALLEL="$MAX_PARALLEL" AUTO_PARALLEL="$AUTO_PARALLEL" \
  AUTO_MIN_FREE_MB="$AUTO_MIN_FREE_MB" AUTO_PEAK_SAFETY_PCT="$AUTO_PEAK_SAFETY_PCT" \
  GPU="${GPU:-0}" FORCE="${FORCE:-0}" SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-}" \
  RUN_SPECS="${RUN_SPECS:-}" TRAINING_CONFIG="${TRAINING_CONFIG:-}" \
  bash scripts/md_agreement.sh
