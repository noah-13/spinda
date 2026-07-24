#!/usr/bin/env bash
set -euo pipefail

# DisCoGeM level2, RoBERTa-base, softmax + MSE.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

MODEL="${MODEL:-roberta-base}" \
SOFT_LABEL_LOSS=mse \
DISCOGEM_LABEL_LEVEL=level2 \
HEAD_TYPE=classification \
  "$SCRIPT_DIR/train_discogem_soft.sh"
