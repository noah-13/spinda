# HLV Toolkits

HLV Toolkits is a toolkit for natural language inference (NLI) and distributional-label experiments. It provides data download and preprocessing, model training, prediction, evaluation, visualization, and experiment dashboards.

## Datasets

- **SNLI**: hard-label NLI data loaded through Hugging Face `datasets`.
- **ChaosNLI**: human label distributions from 100 annotators.
- **DiscoGeM 2.0**: English, German, French, and Czech discourse-relation data with hierarchical labels and human distributions.
- **Processed JSONL**: the normalized format used by the training and evaluation scripts.

The default data locations are:

```text
data/external/chaosnli/
data/external/DiscoGeM/DiscoGeM 2.0/
data/processed/
```

## Models and training heads

Models are loaded with Hugging Face `AutoTokenizer`, `AutoModel`, or `AutoModelForSequenceClassification`. Standard encoder-only models such as BERT, RoBERTa, DeBERTa, XLM-R, ModernBERT, and InfoXLM are supported. Decoder-only and encoder-decoder models are not guaranteed to work with the current NLI interface.

The training entry point is `hlv_toolkits.scripts.train`. Supported heads are:

- `classification`
- `joint_classification`
- `multilevel_classification` for DiscoGeM
- `multilevel_regression` for DiscoGeM

For the current DiscoGeM experiments, `classification` and `multilevel_classification` use a linear output layer followed by softmax for human distributions. The available soft-label losses are:

- `cross_entropy`
- `kl_div`
- `mse` — MSE between the softmax-normalized prediction and the human distribution

The current experiment screen does not include BCE or `per_label_regression`.

Model selection can use `accuracy`, `tvd`, or `kl_divergence`. DiscoGeM options include:

```text
--discogem_label_mode soft|hard
--discogem_label_level level1|level2|level3|all
--discogem_language en|de|fr|cs
```

## Environment and versions

Minimum requirements are defined in `pyproject.toml`:

- Python `>=3.10`
- `uv`
- PyTorch `>=2.2,<2.4`
- Transformers `>=4.57.6`
- Accelerate `>=0.26.0`
- Datasets `>=4.5.0`
- Weights & Biases `>=0.24.0` for experiment tracking

The current `uv.lock` resolves the main development environment with Python 3.12, PyTorch 2.3.1, Transformers 4.57.6, Accelerate 1.13.0, Datasets 4.5.0, and W&B 0.24.0. The lockfile is the source of truth for reproducible installation.

CUDA is optional for CPU execution. For GPU training, use a CUDA-enabled PyTorch installation and verify the available devices with:

```bash
nvidia-smi
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

The CLI accepts `auto`, `cpu`, `cuda`, and `cuda:<index>`:

```text
--device auto
--device cpu
--device cuda
--device cuda:0
```

`cuda` is normalized to `cuda:0`; use `cuda:<index>` to select a specific visible GPU.

## Installation

```bash
uv sync
```

Optional development and plotting dependencies:

```bash
uv sync --extra dev
uv sync --extra plot
```

All commands below should be run from the repository root and through `uv run`, so that the locked environment is used.

## Workflow

```text
download data
    ↓
preprocess to normalized JSONL
    ↓
train
    ↓
generate predictions
    ↓
evaluate and visualize
```

## 1. Download and preprocess

The preprocessing wrapper downloads missing source archives and creates normalized JSONL files. Existing files are not downloaded again.

```bash
bash scripts/preprocess.sh discogem
```

The normalized DiscoGeM file is:

```text
data/processed/discogem.jsonl
```

## 2. Training

### Direct DiscoGeM soft-label training

```bash
uv run python -m hlv_toolkits.scripts.train \
  --data_source discogem \
  --model roberta-base \
  --device cuda:0 \
  --discogem_label_mode soft \
  --discogem_label_level level2 \
  --discogem_language en \
  --soft_label_loss mse \
  --output_dir outputs/discogem/runs/single/level2/roberta-base__classification__soft_label_loss_mse \
  --num_epochs 20 \
  --seeds 42
```

### Shell wrappers

```bash
DEVICE=cuda:0 bash scripts/discogem/train_discogem_soft.sh
DEVICE=cpu bash scripts/discogem/train_discogem_hard.sh
```

The focused RoBERTa-base level2 softmax+MSE train/test wrapper is:

```bash
MODEL=roberta-base DISCOGEM_LABEL_LEVEL=level2 \
  DEVICE=cuda:0 bash scripts/discogem/train_test_discogem_soft_mse.sh
```

The wrapper writes predictions and evaluation to:

```text
outputs/discogem/runs/single/level2/roberta-base__classification__soft_label_loss_mse/seed_42/test/predictions.jsonl
outputs/discogem/runs/single/level2/roberta-base__classification__soft_label_loss_mse/seed_42/test/evaluation.json
```

### Full DiscoGeM screening

The screening script runs the configured encoder models across level1, level2, and level3. It evaluates classification and multilevel classification with cross-entropy, KL divergence, and MSE. It uses W&B project `discogem` by default.

```bash
DEVICE=cuda:0 bash scripts/discogem/exp_discogem_screen.sh
```

The default screening output layout is:

```text
outputs/discogem/screen/<level>/<model>__<head>__<objective>/seed_42/
```

Use `FORCE_RERUN=0` to skip completed runs and test completed runs whose evaluation file is missing. Set `WANDB_PROJECT`, `WANDB_ENTITY`, `SEED`, `DEVICE`, or `OUT_ROOT` to override defaults.

## 3. Prediction

Generate JSONL predictions from a trained model:

```bash
uv run python -m hlv_toolkits.scripts.predict \
  --model_path outputs/snli/hard/seed_42/final_model \
  --data_source snli \
  --split test \
  --device cuda:0 \
  --batch_size 32 \
  --output_file outputs/snli/predictions/test.jsonl
```

For DiscoGeM, pass the matching `--discogem_label_level` and `--discogem_label_mode` used during training.

## 4. Evaluation and visualization

Evaluate predictions against the matching ground truth split:

```bash
uv run python -m hlv_toolkits.scripts.evaluate \
  --predictions outputs/snli/predictions/test.jsonl \
  --ground_truth_source snli \
  --ground_truth_split test \
  --output_file outputs/snli/results/test.json
```

For DiscoGeM, keep predictions and evaluation next to the run:

```bash
uv run python -m hlv_toolkits.scripts.evaluate \
  --predictions outputs/discogem/runs/single/level2/roberta-base__classification__soft_label_loss_mse/seed_42/test/predictions.jsonl \
  --ground_truth_source discogem \
  --ground_truth_split test \
  --discogem_label_level level2 \
  --discogem_label_mode soft \
  --output_file outputs/discogem/runs/single/level2/roberta-base__classification__soft_label_loss_mse/seed_42/test/evaluation.json
```

Distributional evaluation reports accuracy, TVD, JSD, KL divergence, soft micro-F1, soft macro-F1, and distance correlation where applicable.

## 5. DiscoGeM dashboard

Generate or refresh the dashboard notebook:

```bash
uv run python scripts/generate_discogem_dashboard.py
```

Open [`notebooks/discogem_experiment_dashboard.ipynb`](notebooks/discogem_experiment_dashboard.ipynb) and run `Restart Kernel → Run All`. The final cell automatically exports [`notebooks/discogem_results.html`](notebooks/discogem_results.html).

The dashboard reads all `test/evaluation.json` files from the current `outputs/discogem` layout. It aligns single-level and multilevel results by concrete level, showing level1, level2, level3, and overall multilevel performance separately.

## Project structure

```text
hlv_toolkits/
├── data/
├── models/
├── eval/
├── visualization/
└── scripts/
    ├── download_data.py
    ├── train.py
    ├── predict.py
    ├── evaluate.py
    ├── preprocess.py
    └── inspect_data.py

scripts/
├── preprocess.sh
├── generate_discogem_dashboard.py
├── discogem/
│   ├── exp_discogem_screen.sh
│   ├── train_discogem_hard.sh
│   ├── train_discogem_soft.sh
│   ├── train_discogem_soft_mse.sh
│   ├── train_test_discogem_soft_mse.sh
│   └── test_discogem_soft_mse.sh
└── snli/
    ├── train_hard.sh
    └── train_soft.sh
```

## Notes

- The repository uses fixed dataset locations by default.
- Model checkpoints are written below the selected run directory, usually under `seed_<seed>/`.
- Keep the training label level, language, head, and loss consistent between training, prediction, and evaluation.
- W&B logging is opt-in for the generic CLI and enabled by the DiscoGeM screening wrappers.
