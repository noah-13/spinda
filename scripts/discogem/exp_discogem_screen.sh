#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Default DiscoGeM file expected by the training CLI.
DISCOGEM_VERSION="${DISCOGEM_VERSION:-2.0}"
DISCOGEM_LABEL_MODE="${DISCOGEM_LABEL_MODE:-soft}"
DISCOGEM_LABEL_LEVELS="${DISCOGEM_LABEL_LEVELS:-level1 level2 level3}"
DISCOGEM_LANGUAGE="${DISCOGEM_LANGUAGE:-en}"
DISCOGEM_DATA_FILE="${DISCOGEM_DATA_FILE:-data/processed/discogem.jsonl}"
WANDB_PROJECT="${WANDB_PROJECT:-discogem}"
WANDB_ENTITY="${WANDB_ENTITY:-noah103374955-ludwig-maximilianuniversity-of-munich}"
WANDB_GROUP="${WANDB_GROUP:-discogem}"
WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-screen}"
OUT_ROOT="${OUT_ROOT:-outputs/discogem/screen}"
SEED="${SEED:-42}"
NUM_EPOCHS="${NUM_EPOCHS:-30}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
MAX_LENGTH="${MAX_LENGTH:-0}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-64}"
TEST_DEVICE="${TEST_DEVICE:-cuda}"
DEVICE="${DEVICE:-auto}"
FORCE_RERUN="${FORCE_RERUN:-0}"

# Single A100 80GB: keep the defaults and run serially.
# MIG: start with TRAIN_BATCH_SIZE=8 or 16, EVAL_BATCH_SIZE=16 or 32, and increase GRAD_ACCUM if needed.
MODELS=(
  "bert-base-uncased"
  "roberta-base"
  "microsoft/deberta-v3-base"
  "xlm-roberta-base"
  "studio-ousia/luke-base"
  "answerdotai/ModernBERT-base"
  "microsoft/infoxlm-base"
  "Twitter/twhin-bert-base"
)

RUN_SPECS=(
  "classification|soft_label_loss|cross_entropy"
  "classification|soft_label_loss|kl_div"
  "classification|soft_label_loss|mse"
)

MULTILEVEL_RUN_SPECS=(
  "multilevel_classification|soft_label_loss|cross_entropy"
  "multilevel_classification|soft_label_loss|kl_div"
  "multilevel_classification|soft_label_loss|mse"
)

format_duration() {
  local total_seconds="$1"
  local hours=$(( total_seconds / 3600 ))
  local minutes=$(( (total_seconds % 3600) / 60 ))
  local seconds=$(( total_seconds % 60 ))
  printf "%02dh:%02dm:%02ds" "$hours" "$minutes" "$seconds"
}

render_progress_bar() {
  local completed="$1"
  local total="$2"
  local width=30
  local filled=0
  local empty=$width
  local bar=""
  local i

  if (( total > 0 )); then
    filled=$(( completed * width / total ))
    empty=$(( width - filled ))
  fi

  for ((i = 0; i < filled; i++)); do
    bar+="#"
  done
  for ((i = 0; i < empty; i++)); do
    bar+="-"
  done

  printf "[%s]" "$bar"
}

reset_run_dir() {
  local run_dir="$1"
  if [[ -d "$run_dir" ]]; then
    rm -rf "$run_dir"
  fi
}

ensure_discogem_processed() {
  local missing=0
  local level
  if [[ ! -f "$DISCOGEM_DATA_FILE" ]]; then
    echo "DiscoGeM processed data missing; running preprocess..."
    bash scripts/preprocess.sh discogem
  fi
}

run_train() {
  local model="$1"
  local head="$2"
  local loss_kind="$3"
  local loss_value="$4"
  local safe_model="$5"
  local label_level="$6"

  local out_dir="$OUT_ROOT/${label_level}/${safe_model}__${head}__${loss_kind}_${loss_value}"

  reset_run_dir "$out_dir/seed_${SEED}"

  cmd=(
    uv run python -m hlv_toolkits.scripts.train
    --device "$DEVICE"
    --data_source processed
    --processed_task discogem
    --discogem_label_mode "$DISCOGEM_LABEL_MODE"
    --discogem_label_level "$label_level"
    --discogem_language "$DISCOGEM_LANGUAGE"
    --head_type "$head"
    --soft_label_metric_for_best_model tvd
    --model "$model"
    --output_dir "$out_dir"
    --num_epochs "$NUM_EPOCHS"
    --learning_rate "$LEARNING_RATE"
    --train_batch_size "$TRAIN_BATCH_SIZE"
    --eval_batch_size "$EVAL_BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --max_length "$MAX_LENGTH"
    --seeds "$SEED"
    --fp16
    --use_wandb
    --wandb_project "$WANDB_PROJECT"
    --wandb_entity "$WANDB_ENTITY"
    --wandb_group "$WANDB_GROUP"
    --wandb_job_type "$WANDB_JOB_TYPE"
    --wandb_run_name "${WANDB_GROUP}__${label_level}__${safe_model}__${head}__${loss_kind}_${loss_value}__seed_${SEED}"
  )

  case "$head" in
    classification)
      cmd+=( --soft_label_loss "$loss_value" )
      ;;
    *)
      echo "Unsupported head: $head" >&2
      exit 1
      ;;
  esac

  "${cmd[@]}"
}

run_train_multilevel() {
  local model="$1"
  local head="$2"
  local loss_kind="$3"
  local loss_value="$4"
  local safe_model="$5"

  local out_dir="$OUT_ROOT/multilevel/${safe_model}__${head}__${loss_kind}_${loss_value}"

  reset_run_dir "$out_dir/seed_${SEED}"

  cmd=(
    uv run python -m hlv_toolkits.scripts.train
    --device "$DEVICE"
    --data_source processed
    --processed_task discogem
    --discogem_label_mode "$DISCOGEM_LABEL_MODE"
    --discogem_label_level all
    --discogem_language "$DISCOGEM_LANGUAGE"
    --head_type "$head"
    --soft_label_metric_for_best_model tvd
    --model "$model"
    --output_dir "$out_dir"
    --num_epochs "$NUM_EPOCHS"
    --learning_rate "$LEARNING_RATE"
    --train_batch_size "$TRAIN_BATCH_SIZE"
    --eval_batch_size "$EVAL_BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --max_length "$MAX_LENGTH"
    --seeds "$SEED"
    --fp16
    --use_wandb
    --wandb_project "$WANDB_PROJECT"
    --wandb_entity "$WANDB_ENTITY"
    --wandb_group "$WANDB_GROUP"
    --wandb_job_type "$WANDB_JOB_TYPE"
    --wandb_run_name "${WANDB_GROUP}__multilevel__${safe_model}__${head}__${loss_kind}_${loss_value}__seed_${SEED}"
  )

  case "$head" in
    multilevel_classification)
      cmd+=( --soft_label_loss "$loss_value" )
      ;;
    multilevel_regression)
      ;;
    *)
      echo "Unsupported multilevel head: $head" >&2
      exit 1
      ;;
  esac

  "${cmd[@]}"
}

run_test() {
  local model="$1"
  local head="$2"
  local loss_kind="$3"
  local loss_value="$4"
  local safe_model="$5"
  local label_level="$6"

  local out_dir="$OUT_ROOT/${label_level}/${safe_model}__${head}__${loss_kind}_${loss_value}"
  local model_dir="$out_dir/seed_${SEED}/final_model"
  local run_tag="${label_level}__${safe_model}__${head}__${loss_kind}_${loss_value}__seed_${SEED}"
  local test_dir="$out_dir/seed_${SEED}/test"
  local pred_file="$test_dir/predictions.jsonl"
  local eval_file="$test_dir/evaluation.json"

  if [[ ! -d "$model_dir" ]]; then
    echo "Missing final_model for test: $model_dir" >&2
    exit 1
  fi

  echo "Starting test: $model | level=$label_level | head=$head | $loss_kind=$loss_value"
  echo "Test predictions: $pred_file"
  echo "Test evaluation: $eval_file"

  uv run python -m hlv_toolkits.scripts.predict     --model_path "$model_dir"     --data_source processed     --processed_task discogem     --split test     --discogem_label_level "$label_level"     --discogem_label_mode "$DISCOGEM_LABEL_MODE"     --output_file "$pred_file"     --batch_size "$TEST_BATCH_SIZE"     --max_length "$MAX_LENGTH"     --device "$TEST_DEVICE"

  uv run python -m hlv_toolkits.scripts.evaluate \
    --predictions "$pred_file" \
    --ground_truth_source processed \
    --processed_task discogem \
    --ground_truth_split test \
    --discogem_label_level "$label_level" \
    --discogem_label_mode "$DISCOGEM_LABEL_MODE" \
    --output_file "$eval_file" \
    --no-plot \
    --no-ternary_browser

  echo "Test completed: $eval_file"
}

run_test_multilevel() {
  local model="$1"
  local head="$2"
  local loss_kind="$3"
  local loss_value="$4"
  local safe_model="$5"

  local out_dir="$OUT_ROOT/multilevel/${safe_model}__${head}__${loss_kind}_${loss_value}"
  local model_dir="$out_dir/seed_${SEED}/final_model"
  local run_tag="multilevel__${safe_model}__${head}__${loss_kind}_${loss_value}__seed_${SEED}"
  local test_dir="$out_dir/seed_${SEED}/test"
  local pred_file="$test_dir/predictions.jsonl"
  local eval_file="$test_dir/evaluation.json"

  if [[ ! -d "$model_dir" ]]; then
    echo "Missing final_model for test: $model_dir" >&2
    exit 1
  fi

  echo "Starting multilevel test: $model | head=$head | $loss_kind=$loss_value"
  echo "Test predictions: $pred_file"
  echo "Test evaluation: $eval_file"

  uv run python -m hlv_toolkits.scripts.predict     --model_path "$model_dir"     --data_source processed     --processed_task discogem     --split test     --discogem_label_level all     --discogem_label_mode "$DISCOGEM_LABEL_MODE"     --output_file "$pred_file"     --batch_size "$TEST_BATCH_SIZE"     --max_length "$MAX_LENGTH"     --device "$TEST_DEVICE"

  uv run python -m hlv_toolkits.scripts.evaluate \
    --predictions "$pred_file" \
    --ground_truth_source processed \
    --processed_task discogem \
    --ground_truth_split test \
    --discogem_label_level all \
    --discogem_label_mode "$DISCOGEM_LABEL_MODE" \
    --output_file "$eval_file" \
    --no-plot \
    --no-ternary_browser

  echo "Multilevel test completed: $eval_file"
}

get_run_state() {
  local safe_model="$1"
  local head="$2"
  local loss_kind="$3"
  local loss_value="$4"
  local label_level="$5"

  local out_dir="$OUT_ROOT/${label_level}/${safe_model}__${head}__${loss_kind}_${loss_value}"
  local model_dir="$out_dir/seed_${SEED}/final_model"
  local run_tag="${label_level}__${safe_model}__${head}__${loss_kind}_${loss_value}__seed_${SEED}"
  local eval_file="$out_dir/seed_${SEED}/test/evaluation.json"

  if [[ "$FORCE_RERUN" == "1" ]]; then
    echo pending
  elif [[ -f "$eval_file" ]]; then
    echo done
  elif [[ -d "$model_dir" ]]; then
    echo train_only
  else
    echo pending
  fi
}

get_multilevel_run_state() {
  local safe_model="$1"
  local head="$2"
  local loss_kind="$3"
  local loss_value="$4"

  local out_dir="$OUT_ROOT/multilevel/${safe_model}__${head}__${loss_kind}_${loss_value}"
  local model_dir="$out_dir/seed_${SEED}/final_model"
  local run_tag="multilevel__${safe_model}__${head}__${loss_kind}_${loss_value}__seed_${SEED}"
  local eval_file="$out_dir/seed_${SEED}/test/evaluation.json"

  if [[ "$FORCE_RERUN" == "1" ]]; then
    echo pending
  elif [[ -f "$eval_file" ]]; then
    echo done
  elif [[ -d "$model_dir" ]]; then
    echo train_only
  else
    echo pending
  fi
}

read -r estimated_completed estimated_avg_seconds <<< "$(OUT_ROOT="$OUT_ROOT" SEED="$SEED" python3 - <<'PY2'
from pathlib import Path
from statistics import mean
import os
out_root = Path(os.environ['OUT_ROOT'])
seed = os.environ['SEED']
runtimes = []
for final_model in out_root.glob(f'**/seed_{seed}/final_model'):
    seed_dir = final_model.parent
    run_name = final_model.parent.parent.name
    level_name = final_model.parent.parent.parent.name
    eval_file = seed_dir / 'test' / 'evaluation.json'
    checkpoints = list(seed_dir.glob('checkpoint-*/trainer_state.json'))
    if not checkpoints or not eval_file.exists():
        continue
    start = min(p.stat().st_mtime for p in checkpoints)
    end = eval_file.stat().st_mtime
    runtimes.append(max(0.0, end - start))
if not runtimes:
    print('0 0')
else:
    print(f'{len(runtimes)} {mean(runtimes):.1f}')
PY2
)"

read -r -a LABEL_LEVELS <<< "$DISCOGEM_LABEL_LEVELS"
TOTAL_MODELS="${#MODELS[@]}"
RUNS_PER_MODEL=$(( ${#RUN_SPECS[@]} * ${#LABEL_LEVELS[@]} ))
TOTAL_RUNS=$(( TOTAL_MODELS * RUNS_PER_MODEL ))
script_start_ts=$(date +%s)
declare -a PENDING_RUNS=()
completed_runs=0
train_only_runs=0

ensure_discogem_processed

for model in "${MODELS[@]}"; do
  safe_model="${model//\//_}"
  for label_level in "${LABEL_LEVELS[@]}"; do
    for spec in "${RUN_SPECS[@]}"; do
      IFS='|' read -r head loss_kind loss_value <<< "$spec"
      state="$(get_run_state "$safe_model" "$head" "$loss_kind" "$loss_value" "$label_level")"
      case "$state" in
        done)
          completed_runs=$(( completed_runs + 1 ))
          ;;
        train_only)
          train_only_runs=$(( train_only_runs + 1 ))
          PENDING_RUNS+=( "$model|$head|$loss_kind|$loss_value|$label_level|test_only" )
          ;;
        pending)
          PENDING_RUNS+=( "$model|$head|$loss_kind|$loss_value|$label_level|train_and_test" )
          ;;
      esac
    done
  done
done

PENDING_COUNT="${#PENDING_RUNS[@]}"
estimated_remaining_seconds=0
estimated_total_seconds=0
if [[ "$estimated_avg_seconds" != "0" ]]; then
  estimated_remaining_seconds="$(python3 - <<PY2
avg = float('${estimated_avg_seconds}')
pending = int('${PENDING_COUNT}')
print(int(round(avg * pending)))
PY2
)"
  estimated_total_seconds="$(python3 - <<PY2
avg = float('${estimated_avg_seconds}')
total = int('${TOTAL_RUNS}')
print(int(round(avg * total)))
PY2
)"
fi

echo "Starting DiscoGeM screening"
echo "Dataset: DiscoGeM ${DISCOGEM_VERSION} | language=${DISCOGEM_LANGUAGE} | levels=${DISCOGEM_LABEL_LEVELS}"
echo "Models: $TOTAL_MODELS | Runs per model: $RUNS_PER_MODEL | Total runs: $TOTAL_RUNS"
echo "Already complete: $completed_runs | Train done, test missing: $train_only_runs | Remaining to process now: $PENDING_COUNT"
if [[ "$estimated_avg_seconds" != "0" ]]; then
  echo "Estimated average completed run time: $(format_duration "${estimated_avg_seconds%.*}")"
  echo "Estimated remaining time: $(format_duration "$estimated_remaining_seconds")"
  echo "Estimated total time for all $TOTAL_RUNS runs: $(format_duration "$estimated_total_seconds")"
fi
echo

if [[ "$PENDING_COUNT" -eq 0 ]]; then
  echo "Nothing to do. All single-level runs already trained and tested."
fi

processed_now=0
for entry in "${PENDING_RUNS[@]}"; do
  IFS='|' read -r model head loss_kind loss_value label_level action <<< "$entry"
  safe_model="${model//\//_}"
  current_run=$(( processed_now + 1 ))
  remaining_runs=$(( PENDING_COUNT - current_run ))
  progress_bar="$(render_progress_bar "$processed_now" "$PENDING_COUNT")"

  echo "============================================================"
  echo "$progress_bar Pending run $current_run/$PENDING_COUNT | Remaining after this: $remaining_runs"
  echo "Model: $model"
  echo "Config: level=$label_level | head=$head | $loss_kind=$loss_value"
  echo "Action: $action"
  echo "============================================================"

  run_start_ts=$(date +%s)
  if [[ "$action" == "train_and_test" ]]; then
    run_train "$model" "$head" "$loss_kind" "$loss_value" "$safe_model" "$label_level"
  fi
  run_test "$model" "$head" "$loss_kind" "$loss_value" "$safe_model" "$label_level"
  run_end_ts=$(date +%s)

  processed_now=$current_run
  run_elapsed=$(( run_end_ts - run_start_ts ))
  echo "Processed pending run $processed_now/$PENDING_COUNT in $(format_duration "$run_elapsed")"
  echo
done

script_end_ts=$(date +%s)
script_elapsed=$(( script_end_ts - script_start_ts ))

echo "============================================================"
echo "DiscoGeM screening sync completed"
echo "Work done this invocation: $(format_duration "$script_elapsed")"
echo "Previously complete at start: $completed_runs/$TOTAL_RUNS"
echo "Processed now: $processed_now"
echo "Expected complete after this run: $(( completed_runs + processed_now ))/$TOTAL_RUNS"
echo "============================================================"

echo
echo "============================================================"
echo "Starting DiscoGeM multilevel screening"
echo "Dataset: DiscoGeM ${DISCOGEM_VERSION} | language=${DISCOGEM_LANGUAGE} | levels=all"
echo "Models: $TOTAL_MODELS | Runs per model: ${#MULTILEVEL_RUN_SPECS[@]} | Total runs: $(( TOTAL_MODELS * ${#MULTILEVEL_RUN_SPECS[@]} ))"
echo

declare -a MULTILEVEL_PENDING_RUNS=()
multi_completed_runs=0
multi_train_only_runs=0
for model in "${MODELS[@]}"; do
  safe_model="${model//\//_}"
  for spec in "${MULTILEVEL_RUN_SPECS[@]}"; do
    IFS='|' read -r head loss_kind loss_value <<< "$spec"
    state="$(get_multilevel_run_state "$safe_model" "$head" "$loss_kind" "$loss_value")"
    case "$state" in
      done)
        multi_completed_runs=$(( multi_completed_runs + 1 ))
        ;;
      train_only)
        multi_train_only_runs=$(( multi_train_only_runs + 1 ))
        MULTILEVEL_PENDING_RUNS+=( "$model|$head|$loss_kind|$loss_value|test_only" )
        ;;
      pending)
        MULTILEVEL_PENDING_RUNS+=( "$model|$head|$loss_kind|$loss_value|train_and_test" )
        ;;
    esac
  done
done

MULTILEVEL_PENDING_COUNT="${#MULTILEVEL_PENDING_RUNS[@]}"
echo "Already complete: $multi_completed_runs | Train done, test missing: $multi_train_only_runs | Remaining to process now: $MULTILEVEL_PENDING_COUNT"
if [[ "$MULTILEVEL_PENDING_COUNT" -eq 0 ]]; then
  echo "Nothing to do. All multilevel runs already trained and tested."
else
  multi_processed_now=0
  for entry in "${MULTILEVEL_PENDING_RUNS[@]}"; do
    IFS='|' read -r model head loss_kind loss_value action <<< "$entry"
    safe_model="${model//\//_}"
    current_run=$(( multi_processed_now + 1 ))
    remaining_runs=$(( MULTILEVEL_PENDING_COUNT - current_run ))
    progress_bar="$(render_progress_bar "$multi_processed_now" "$MULTILEVEL_PENDING_COUNT")"

    echo "============================================================"
    echo "$progress_bar Pending multilevel run $current_run/$MULTILEVEL_PENDING_COUNT | Remaining after this: $remaining_runs"
    echo "Model: $model"
    echo "Config: head=$head | $loss_kind=$loss_value"
    echo "Action: $action"
    echo "============================================================"

    run_start_ts=$(date +%s)
    if [[ "$action" == "train_and_test" ]]; then
      run_train_multilevel "$model" "$head" "$loss_kind" "$loss_value" "$safe_model"
    fi
    run_test_multilevel "$model" "$head" "$loss_kind" "$loss_value" "$safe_model"
    run_end_ts=$(date +%s)

    multi_processed_now=$current_run
    run_elapsed=$(( run_end_ts - run_start_ts ))
    echo "Processed pending multilevel run $multi_processed_now/$MULTILEVEL_PENDING_COUNT in $(format_duration "$run_elapsed")"
    echo
  done
fi

echo "============================================================"
echo "DiscoGeM multilevel screening completed"
echo "Processed now: ${multi_processed_now:-0}"
echo "Expected complete after this run: $(( multi_completed_runs + ${multi_processed_now:-0} ))/$(( TOTAL_MODELS * ${#MULTILEVEL_RUN_SPECS[@]} ))"
echo "============================================================"
