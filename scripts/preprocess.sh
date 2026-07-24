#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."


SOURCE="${1:-${SOURCE:-}}"
if [[ -z "$SOURCE" ]]; then
  echo "Usage: $0 <snli|chaosnli|discogem>" >&2
  echo "   or: SOURCE=snli $0" >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-data/processed}"

case "$SOURCE" in
  snli)
    uv run python -m hlv_toolkits.scripts.preprocess \
      --source snli \
      --output_dir "$OUTPUT_DIR"
    ;;
  chaosnli)
    uv run python -m hlv_toolkits.scripts.download_data chaosnli \
      --chaosnli-dir "${CHAOSNLI_DIR:-data/external/chaosnli}"
    uv run python -m hlv_toolkits.scripts.preprocess \
      --source chaosnli \
      --output_dir "$OUTPUT_DIR" \
    ;;
  discogem)
    uv run python -m hlv_toolkits.scripts.download_data discogem \
      --discogem-path "${DISCOGEM_PATH:-data/external/DiscoGeM/DiscoGeM 2.0/DiscoGeM2.0_annotation.tgz}"
    uv run python -m hlv_toolkits.scripts.preprocess \
      --source discogem \
      --output_dir "$OUTPUT_DIR" \
      --discogem_language "${DISCOGEM_LANGUAGE:-en}"
    ;;
  *)
    echo "Unknown source: $SOURCE" >&2
    echo "Expected one of: snli, chaosnli, discogem" >&2
    exit 1
    ;;
esac
