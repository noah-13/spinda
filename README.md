# NLI Toolkits

A toolkit for Natural Language Inference (NLI) training and evaluation, with support for:
- Hard-label training on SNLI
- Soft-label (human distribution) training on ChaosNLI
- Unified prediction export (JSONL)
- Evaluation on both single-label and distributional ground truth
- TVD and ternary visualizations (including interactive HTML)

## 1. Feature Overview

- Data loading
  - `SNLIReader`: loads `train/dev/test` from HuggingFace `snli`
  - `ChaosNLIReader`: loads ChaosNLI JSONL with multiple supported schemas (`label_count` / `label_dist` / `label_counter`)
- Model training
  - Backbones: `roberta-base` / `bert-base-uncased`
  - Multi-seed training support
  - Soft-label training (`cross_entropy` or `kl_div`)
  - Optional Weights & Biases logging
- Prediction
  - Standard JSONL output with `id/task/split/source/outputs`
  - `outputs` includes at least `probs` (3-way probabilities) and `pred` (label id)
- Evaluation
  - Single-label (SNLI): `accuracy`
  - Distributional (ChaosNLI):
    - `accuracy`
    - `tvd_mean`
    - `jsd_mean`
    - `kl_mean` (KL(human || pred))
    - `soft_micro_f1`
    - `soft_macro_f1`
    - `distance_correlation`
- Visualization
  - TVD distribution plot (PNG)
  - Ternary distribution plot (PNG)
  - Interactive ternary plot (HTML with hover metadata)

## 2. Installation

Python 3.12+ is recommended.

```bash
# With uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## 3. Quick Start

### 3.1 Train (SNLI, hard labels)

```bash
python -m nli_toolkits.scripts.train \
  --data_source snli \
  --model roberta-base \
  --output_dir ./outputs/hard_snli \
  --num_epochs 3 \
  --train_batch_size 32 \
  --eval_batch_size 64 \
  --seeds 42
```

Default model output location:
- `./outputs/hard_snli/seed_42/final_model`

### 3.2 Predict

```bash
python -m nli_toolkits.scripts.predict \
  --model_path ./outputs/hard_snli/seed_42/final_model \
  --data_source snli \
  --split test \
  --output_file outputs/predictions/snli_preds.jsonl \
  --batch_size 32
```

### 3.3 Evaluate (SNLI)

```bash
python -m nli_toolkits.scripts.evaluate \
  --predictions outputs/predictions/snli_preds.jsonl \
  --ground_truth_source snli \
  --ground_truth_split test \
  --output_file outputs/results/snli_eval.json
```

## 4. ChaosNLI Workflow (Soft Labels)

### 4.1 Split train/dev (optional)

```bash
python -m nli_toolkits.scripts.split_chaosnli \
  --input chaosNLI_v1.0/chaosNLI_snli.jsonl \
  --train-size 1400 \
  --seed 42 \
  --train-output chaosNLI_v1.0/chaosNLI_snli_train.jsonl \
  --dev-output chaosNLI_v1.0/chaosNLI_snli_dev.jsonl
```

### 4.2 Soft-label training (ChaosNLI)

```bash
python -m nli_toolkits.scripts.train \
  --data_source chaosnli \
  --chaosnli_train_path chaosNLI_v1.0/chaosNLI_snli_train.jsonl \
  --chaosnli_dev_path chaosNLI_v1.0/chaosNLI_snli_dev.jsonl \
  --use_soft_labels \
  --soft_label_loss cross_entropy \
  --soft_label_metric_for_best_model tvd \
  --model roberta-base \
  --output_dir ./outputs/soft_chaosnli \
  --num_epochs 20 \
  --seeds 42
```

### 4.3 Predict on ChaosNLI

```bash
python -m nli_toolkits.scripts.predict \
  --model_path ./outputs/soft_chaosnli/seed_42/final_model \
  --data_source chaosnli \
  --chaosnli_path chaosNLI_v1.0/chaosNLI_snli.jsonl \
  --split test \
  --output_file outputs/predictions/chaos_preds.jsonl
```

### 4.4 Evaluate + visualize on ChaosNLI

```bash
python -m nli_toolkits.scripts.evaluate \
  --predictions outputs/predictions/chaos_preds.jsonl \
  --ground_truth_source chaosnli \
  --chaosnli_path chaosNLI_v1.0/chaosNLI_snli.jsonl \
  --output_file outputs/results/chaos_eval.json \
  --plots tvd ternary \
  --plot_dir outputs/figures \
  --ternary_source both
```

Useful optional flags:
- `--no-plot`: disable plotting
- `--no-ternary_browser`: disable interactive HTML ternary output
- `--predictions_format machamp`: parse predictions in machamp/ChaosNLI-style TSV format

## 5. Unified CLI Entry (main.py)

You can also use the unified entrypoint:

```bash
python main.py train ...
python main.py predict ...
python main.py evaluate ...
```

## 6. Data Inspection

```bash
python -m nli_toolkits.scripts.inspect_data \
  --source snli \
  --split test \
  --num_examples 5
```

Or for ChaosNLI:

```bash
python -m nli_toolkits.scripts.inspect_data \
  --source chaosnli \
  --split test \
  --chaosnli_path chaosNLI_v1.0/chaosNLI_snli.jsonl \
  --num_examples 10
```

## 7. Prediction File Format (JSONL)

One-line example:

```json
{"id":"341#1","task":"nli","split":"test","source":"snli","outputs":{"probs":[0.7,0.2,0.1],"pred":0}}
```

Notes:
- `id` must align with ground-truth sample IDs for evaluation
- `outputs.probs` must be a 3-way probability vector
- `outputs.pred` is label id (`0`: entailment, `1`: neutral, `2`: contradiction)

## 8. Project Structure

```text
nli_toolkits/
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
    ├── inspect_data.py
    └── split_chaosnli.py
```

## 9. References

- Baan et al., 2022, *Stop Measuring Calibration When Humans Disagree* (EMNLP)
- Pang et al., 2024, *Seeing the Small Through the Big* (global-structure metrics/visualization inspiration)
