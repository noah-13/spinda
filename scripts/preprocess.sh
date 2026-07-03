#!/usr/bin/env bash
set -euo pipefail

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
    uv run python -m hlv_toolkits.scripts.preprocess \
      --source chaosnli \
      --output_dir "$OUTPUT_DIR" \
      --chaosnli_train_path "${CHAOSNLI_TRAIN_PATH:-data/external/chaosnli/chaosNLI_snli_train.jsonl}" \
      --chaosnli_dev_path "${CHAOSNLI_DEV_PATH:-data/external/chaosnli/chaosNLI_snli_dev.jsonl}" \
      --chaosnli_test_path "${CHAOSNLI_TEST_PATH:-}"
    ;;
  discogem)
    uv run python -m hlv_toolkits.scripts.preprocess \
      --source discogem \
      --output_dir "$OUTPUT_DIR" \
      --discogem_path "${DISCOGEM_PATH:-data/external/DiscoGeM/DiscoGeM 2.0/DiscoGeM2.0_annotation.tgz}" \
      --discogem_version "${DISCOGEM_VERSION:-2.0}" \
      --discogem_language "${DISCOGEM_LANGUAGE:-en}"
    ;;
  *)
    echo "Unknown source: $SOURCE" >&2
    echo "Expected one of: snli, chaosnli, discogem" >&2
    exit 1
    ;;
esac
