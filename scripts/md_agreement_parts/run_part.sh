#!/usr/bin/env bash
# Workload labels: heavy parts contain XLM-R; standard parts contain base BERT-family models.
set -euo pipefail
part="$(basename "$0" .sh)"
case "$part" in
  01_heavy_xlmr_part1) tasks=('xlm-roberta-base|soft ce|42' 'xlm-roberta-base|soft ce|43' 'xlm-roberta-base|soft ce|44' 'xlm-roberta-base|soft mse|42') ;;
  02_heavy_xlmr_part2) tasks=('xlm-roberta-base|soft mse|43' 'xlm-roberta-base|soft mse|44' 'xlm-roberta-base|soft jsd|42' 'xlm-roberta-base|soft jsd|43') ;;
  03_heavy_xlmr_part3) tasks=('xlm-roberta-base|soft jsd|44' 'xlm-roberta-base|soft rel|42' 'xlm-roberta-base|soft rel|43' 'xlm-roberta-base|soft_to_hard ce|42') ;;
  04_heavy_xlmr_twhin) tasks=('xlm-roberta-base|soft rel|44' 'xlm-roberta-base|soft_to_hard ce|43' 'xlm-roberta-base|soft_to_hard ce|44' 'Twitter/twhin-bert-base|soft ce|42' 'Twitter/twhin-bert-base|soft ce|43') ;;
  05_standard_twhin_part1) tasks=('Twitter/twhin-bert-base|soft ce|44' 'Twitter/twhin-bert-base|soft mse|42' 'Twitter/twhin-bert-base|soft mse|43' 'Twitter/twhin-bert-base|soft mse|44' 'Twitter/twhin-bert-base|soft jsd|42' 'Twitter/twhin-bert-base|soft jsd|43') ;;
  06_standard_twhin_part2) tasks=('Twitter/twhin-bert-base|soft jsd|44' 'Twitter/twhin-bert-base|soft rel|42' 'Twitter/twhin-bert-base|soft rel|43' 'Twitter/twhin-bert-base|soft rel|44' 'Twitter/twhin-bert-base|soft_to_hard ce|42' 'Twitter/twhin-bert-base|soft_to_hard ce|43' 'Twitter/twhin-bert-base|soft_to_hard ce|44') ;;
  07_standard_mbert_bert_part1) tasks=('bert-base-multilingual-cased|soft ce|42' 'bert-base-multilingual-cased|soft ce|43' 'bert-base-multilingual-cased|soft ce|44' 'bert-base-multilingual-cased|soft mse|42' 'bert-base-multilingual-cased|soft mse|43') ;;
  08_standard_mbert_bert_part2) tasks=('bert-base-multilingual-cased|soft mse|44' 'bert-base-multilingual-cased|soft jsd|42' 'bert-base-multilingual-cased|soft jsd|43' 'bert-base-multilingual-cased|soft jsd|44' 'bert-base-multilingual-cased|soft rel|42' 'bert-base-uncased|soft mse|43' 'bert-base-uncased|soft mse|44') ;;
  09_standard_mbert_bert_part3) tasks=('bert-base-multilingual-cased|soft rel|43' 'bert-base-multilingual-cased|soft rel|44' 'bert-base-multilingual-cased|soft_to_hard ce|42' 'bert-base-uncased|soft jsd|42' 'bert-base-uncased|soft jsd|43' 'bert-base-uncased|soft jsd|44' 'bert-base-uncased|soft rel|42') ;;
  10_standard_mbert_bert_part4) tasks=('bert-base-multilingual-cased|soft_to_hard ce|43' 'bert-base-multilingual-cased|soft_to_hard ce|44' 'bert-base-uncased|soft rel|43' 'bert-base-uncased|soft rel|44' 'bert-base-uncased|soft_to_hard ce|42' 'bert-base-uncased|soft_to_hard ce|43' 'bert-base-uncased|soft_to_hard ce|44') ;;
  local_resume_started) tasks=('bert-base-uncased|soft ce|44' 'bert-base-uncased|soft mse|42') ;;
  *) echo "Run via a numbered MD-Agreement part script." >&2; exit 2 ;;
esac
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
for task in "${tasks[@]}"; do
  IFS='|' read -r model run_spec seed <<< "$task"
  echo "[$part] $model / $run_spec / seed_$seed"
  MODEL_SPECS="$model" RUN_SPECS="$run_spec" SEEDS_OVERRIDE="$seed" GPU="${GPU:-0}" FORCE="${FORCE:-0}" RUN_NAME="${RUN_NAME:-default}" OUT_ROOT="${OUT_ROOT:-outputs/md_agreement}" bash "$repo_root/scripts/md_agreement.sh"
done
