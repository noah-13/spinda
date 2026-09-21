#!/usr/bin/env bash
# Prepare and train MD-Agreement with the generic single-text sweep.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/interrupt_cleanup.sh"
cd "$SCRIPT_DIR/.."

INPUT_DIR="${MD_AGREEMENT_DIR:-data/raw/md_agreement}"
DATA_DIR="${MD_AGREEMENT_OUTPUT_DIR:-data/datasets/single_text/md_agreement}"
OUT_ROOT="${OUT_ROOT:-outputs/md_agreement}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.json}"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-scripts/run_single_text_sweep.sh}"
RUN_NAME="${RUN_NAME:-default}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"

# Regenerate manifests written by older releases: they lacked the direct-path
# fields required by the generic training configuration contract.
if [[ "$FORCE_PREPARE" == "1" || ! -s "$DATA_DIR/dataset.json" || ! -s "$DATA_DIR/train.json" || ! -s "$DATA_DIR/dev.json" || ! -s "$DATA_DIR/test.json" ]] \
  || ! uv run python -c 'import json, sys; manifest = json.load(open(sys.argv[1])); sys.exit(not all(manifest.get(key) for key in ("train_path", "dev_path")))' "$DATA_DIR/dataset.json"; then
  uv run python -m hlv_toolkits.scripts.prepare_md_agreement_annotation_labels \
    --input-dir "$INPUT_DIR" --output-dir "$DATA_DIR"
fi

DATASET_CONFIG="$DATA_DIR/dataset.json" RUN_NAME="$RUN_NAME" OUT_ROOT="$OUT_ROOT" \
  TRAINING_CONFIG="$TRAINING_CONFIG" GPU="${GPU:-0}" FORCE="${FORCE:-0}" \
  MODEL_SPECS="${MODEL_SPECS:-}" RUN_SPECS="${RUN_SPECS:-}" SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-}" \
  bash "$SWEEP_SCRIPT"
