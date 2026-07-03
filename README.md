# HLV Toolkits

A toolkit for NLI (Natural Language Inference) training, prediction, and evaluation, with support for:

- Hard-label training on SNLI
- Soft-label (human distribution) training on ChaosNLI
- Unified JSONL prediction export
- Evaluation on both single-label and distributional targets
- TVD and ternary visualization (static PNG + interactive HTML)

## Feature Overview

- Data loading
  - `SNLIReader`: loads `train/dev/test` from HuggingFace `snli`
  - `ChaosNLIReader`: loads ChaosNLI JSONL with supported schemas (`label_count` / `label_dist` / `label_counter`)
  - `ProcessedJSONLReader`: loads canonical JSONL from `data/processed/<source>/` or a single file such as `data/processed/discogem.jsonl`
- Preprocessing
  - `preprocess.py`: normalizes raw datasets into canonical JSONL files
- Training
  - Backbones: `roberta-base`, `bert-base-uncased`, `microsoft/deberta-v3-base`
  - Multi-seed training
  - Soft-label training (`cross_entropy` or `kl_div`)
  - Optional Weights & Biases logging
- Prediction
  - Standard JSONL output: `id/task/split/source/outputs`
  - `outputs` includes at least `probs` (3-way probabilities) and `pred` (label id)
- Evaluation
  - Single-label (SNLI): `accuracy`
  - Distributional (ChaosNLI): `accuracy`, `tvd_mean`, `jsd_mean`, `kl_mean`, `soft_micro_f1`, `soft_macro_f1`, `distance_correlation`
- Visualization
  - TVD histogram plot (PNG)
  - Ternary distribution plot (PNG)
  - Interactive ternary plot (HTML with hover metadata)

## Installation

Requires Python `>=3.12`.

```bash
# Recommended: uv
uv sync

# Optional: pip
pip install -e .
```

Optional extras:

```bash
uv sync --extra dev
uv sync --extra plot
```

## Quick Start (SNLI, Hard Labels)

### 1) Train

```bash
uv run python -m hlv_toolkits.scripts.train \
  --data_source snli \
  --model roberta-base \
  --output_dir ./outputs/snli/hard \
  --num_epochs 3 \
  --train_batch_size 32 \
  --eval_batch_size 64 \
  --seeds 42
```

Default model output:

- `./outputs/snli/hard/seed_42/final_model`

DeBERTa baseline example:

```bash
uv run python -m hlv_toolkits.scripts.train \
  --data_source snli \
  --model microsoft/deberta-v3-base \
  --output_dir ./outputs/snli/hard_deberta \
  --num_epochs 3 \
  --train_batch_size 32 \
  --eval_batch_size 64 \
  --seeds 42
```

### 2) Predict

```bash
uv run python -m hlv_toolkits.scripts.predict \
  --model_path ./outputs/snli/hard/seed_42/final_model \
  --data_source snli \
  --split test \
  --output_file outputs/snli/predictions/snli_preds.jsonl \
  --batch_size 32
```

### 3) Evaluate

```bash
uv run python -m hlv_toolkits.scripts.evaluate \
  --predictions outputs/snli/predictions/snli_preds.jsonl \
  --ground_truth_source snli \
  --ground_truth_split test \
  --output_file outputs/snli/results/snli_eval.json
```

## ChaosNLI Workflow (Soft Labels)

### 1) Split train/dev (optional)

```bash
uv run python -m hlv_toolkits.scripts.split_chaosnli \
  --input data/external/chaosnli/chaosNLI_snli.jsonl \
  --train-size 1400 \
  --seed 42 \
  --train-output data/external/chaosnli/chaosNLI_snli_train.jsonl \
  --dev-output data/external/chaosnli/chaosNLI_snli_dev.jsonl
```

### 2) Soft-label training

```bash
uv run python -m hlv_toolkits.scripts.train \
  --data_source chaosnli \
  --chaosnli_train_path data/external/chaosnli/chaosNLI_snli_train.jsonl \
  --chaosnli_dev_path data/external/chaosnli/chaosNLI_snli_dev.jsonl \
  --use_soft_labels \
  --soft_label_loss cross_entropy \
  --soft_label_metric_for_best_model tvd \
  --model roberta-base \
  --output_dir ./outputs/chaosnli/soft \
  --num_epochs 20 \
  --seeds 42
```

`--soft_label_metric_for_best_model` choices:

- `kl_divergence`
- `tvd`
- `accuracy`

### 3) Predict

```bash
uv run python -m hlv_toolkits.scripts.predict \
  --model_path ./outputs/chaosnli/soft/seed_42/final_model \
  --data_source chaosnli \
  --chaosnli_path data/external/chaosnli/chaosNLI_snli.jsonl \
  --split test \
  --output_file outputs/chaosnli/predictions/chaos_preds.jsonl
```

### 4) Evaluate + visualize

```bash
uv run python -m hlv_toolkits.scripts.evaluate \
  --predictions outputs/chaosnli/predictions/chaos_preds.jsonl \
  --ground_truth_source chaosnli \
  --ground_truth_split test \
  --chaosnli_path data/external/chaosnli/chaosNLI_snli.jsonl \
  --output_file outputs/chaosnli/results/chaos_eval.json \
  --plots tvd ternary \
  --plot_dir outputs/chaosnli/figures \
  --ternary_source both
```

Useful flags:

- `--no-plot`: disable all plots
- `--no-ternary_browser`: disable interactive HTML ternary plot
- `--predictions_format machamp`: parse machamp/ChaosNLI-style TSV predictions
- `--ternary_source {model,human,both}`: choose ternary visualization source

### 5) Experiment scripts

The current experiment orchestrator lives at:

```bash
bash scripts/discogem/exp_discogem_screen.sh
```

It runs the HLV package CLI under `python -m hlv_toolkits.scripts.*` and writes screening / test artifacts under `outputs/discogem/`.
The DiscoGeM canonical preprocessed file is `data/processed/discogem.jsonl`.

Top-level analysis helpers also live under `scripts/`:

- `scripts/generate_chaosnli_dashboard.py`
- `scripts/generate_discogem_dashboard.py`

Hardware notes:

- Single `A100 80GB`: keep the default `TRAIN_BATCH_SIZE=32` and `EVAL_BATCH_SIZE=64`.
- MIG: start with `TRAIN_BATCH_SIZE=8` or `16`, `EVAL_BATCH_SIZE=16` or `32`, and raise gradient accumulation if needed.

## Standardized Preprocessing

The repo now supports a two-step flow:

1. Normalize raw data into canonical JSONL files under `data/processed/<source>/`.
2. Train, predict, and evaluate from those processed files with `--data_source processed`.

Examples:

Shell wrappers:

```bash
bash scripts/preprocess.sh snli
bash scripts/preprocess.sh chaosnli
bash scripts/preprocess.sh discogem
```

`scripts/preprocess.sh` accepts the source name as the first positional argument, or via `SOURCE=...`.

```bash
bash scripts/preprocess.sh discogem
SOURCE=discogem bash scripts/preprocess.sh
```

DiscoGeM is special-cased: preprocessing merges the soft and hard annotations into one canonical file,
`data/processed/discogem.jsonl`.

SNLI:

```bash
uv run python -m hlv_toolkits.scripts.preprocess \
  --source snli \
  --output_dir data/processed

uv run python -m hlv_toolkits.scripts.train \
  --data_source processed \
  --processed_task nli \
  --processed_data_dir data/processed/snli \
  --model roberta-base \
  --output_dir ./outputs/snli/hard
```

ChaosNLI:

```bash
uv run python -m hlv_toolkits.scripts.preprocess \
  --source chaosnli \
  --chaosnli_train_path data/external/chaosnli/chaosNLI_snli_train.jsonl \
  --chaosnli_dev_path data/external/chaosnli/chaosNLI_snli_dev.jsonl \
  --output_dir data/processed

uv run python -m hlv_toolkits.scripts.train \
  --data_source processed \
  --processed_task nli \
  --processed_data_dir data/processed/chaosnli \
  --use_soft_labels \
  --soft_label_loss cross_entropy \
  --soft_label_metric_for_best_model tvd \
  --model roberta-base \
  --output_dir ./outputs/chaosnli/soft
```

DiscoGeM 2.0:

```bash
uv run python -m hlv_toolkits.scripts.preprocess \
  --source discogem \
  --discogem_path "data/external/DiscoGeM/DiscoGeM 2.0/DiscoGeM2.0_annotation.tgz" \
  --discogem_version 2.0 \
  --output_dir data/processed

uv run python -m hlv_toolkits.scripts.train \
  --data_source processed \
  --processed_task discogem \
  --processed_data_dir data/processed/discogem.jsonl \
  --use_soft_labels \
  --soft_label_loss cross_entropy \
  --soft_label_metric_for_best_model tvd \
  --model roberta-base \
  --output_dir ./outputs/discogem/soft
```

## Unified CLI Entry

You can also use `main.py`:

```bash
uv run python main.py train ...
uv run python main.py predict ...
uv run python main.py evaluate ...
uv run python main.py preprocess ...
```

## Data Inspection

SNLI:

```bash
uv run python -m hlv_toolkits.scripts.inspect_data \
  --source snli \
  --split test \
  --num_examples 5
```

ChaosNLI:

```bash
uv run python -m hlv_toolkits.scripts.inspect_data \
  --source chaosnli \
  --split test \
  --chaosnli_path data/external/chaosnli/chaosNLI_snli.jsonl \
  --num_examples 10
```

## Prediction File Format (JSONL)

One-line example:

```json
{"id":"341#1","task":"nli","split":"test","source":"snli","outputs":{"probs":[0.7,0.2,0.1],"pred":0}}
```

Notes:

- `id` must align with ground-truth sample IDs
- `outputs.probs` must be a 3-way probability vector
- Label mapping: `0=entailment`, `1=neutral`, `2=contradiction`

## Project Structure

```text
hlv_toolkits/
├── data/
│   ├── schemas.py
│   └── readers/
│       ├── snli_reader.py
│       └── chaosnli_reader.py
├── models/
│   └── trainer.py
├── eval/
│   ├── evaluator.py
│   └── metrics.py
├── visualization/
│   └── plots.py
└── scripts/
    ├── train.py
    ├── predict.py
    ├── evaluate.py
    ├── preprocess.py
    ├── inspect_data.py
    └── split_chaosnli.py

scripts/
├── preprocess.sh
├── generate_chaosnli_dashboard.py
├── generate_discogem_dashboard.py
├── discogem/
│   ├── exp_discogem_screen.sh
│   ├── train_discogem_hard.sh
│   └── train_discogem_soft.sh
└── snli/
    ├── train_hard.sh
    └── train_soft.sh
```

## Tutorials and Docs

- HLV metrics tutorial: `docs/hlv_metrics_tutorial.md`
- Notebook tutorial: `notebooks/metrics_tutorial.ipynb`

## References

## DiscoGeM Workflow (2.0, Hard + Soft)

The training CLI supports DiscoGeM 2.0 as the DiscoGeM experiment track in this repo:

- `--data_source discogem`
- `--discogem_path <path>`
- `--discogem_version 2.0`
- `--discogem_label_mode {soft,hard}`

### Soft-label training (DiscoGeM 2.0)

```bash
uv run python -m hlv_toolkits.scripts.train \
  --data_source discogem \
  --discogem_path "data/external/DiscoGeM/DiscoGeM 2.0/DiscoGeM2.0_annotation.tgz" \
  --discogem_version 2.0 \
  --discogem_label_mode soft \
  --model roberta-base \
  --output_dir ./outputs/discogem/soft
```

### Hard-label training (DiscoGeM 2.0)

```bash
uv run python -m hlv_toolkits.scripts.train \
  --data_source discogem \
  --discogem_path "data/external/DiscoGeM/DiscoGeM 2.0/DiscoGeM2.0_annotation.tgz" \
  --discogem_version 2.0 \
  --discogem_label_mode hard \
  --model roberta-base \
  --output_dir ./outputs/discogem/hard
```

Notes:

- The DiscoGeM 2.0 archive is the default experiment source in this repo.
- The canonical processed dataset is the single file `data/processed/discogem.jsonl`.
- Use `--discogem_label_mode` to switch between soft and hard labels.
