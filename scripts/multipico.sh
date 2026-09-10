#!/usr/bin/env bash
# Backward-compatible entry point for the all-language MultiPICo experiment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/multipico/multilingual.sh"
