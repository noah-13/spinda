# HLV Toolkits

HLV Toolkits is a toolkit for natural language inference (NLI) and distributional-label experiments. It provides data download and preprocessing, model training, prediction, evaluation, visualization, and experiment dashboards.

## Datasets

- **SNLI**: hard-label NLI data loaded through Hugging Face `datasets`.
- **ChaosNLI**: converted from its official SNLI JSONL into the shared text-pair `annotation_labels` format.
- **DiscoGeM 2.0**: English, German, French, and Czech discourse-relation data with hierarchical labels and human distributions.
- **Processed JSONL**: the normalized format used by the training and evaluation scripts.
- **Text-pair classification JSONL**: a fixed public format for training on user-provided hard-label datasets; see [data/README.md](data/README.md#public-text-pair-classification-format).

The default data locations are:

```text
data/external/DiscoGeM/DiscoGeM 2.0/
data/processed/
```

## Quick start: ChaosNLI

ChaosNLI downloads and prepares the `snli` and `mnli_m` subsets if needed, then runs the three-seed training sweep and test evaluation for fold 0.

```bash
bash scripts/chaosnli.sh
```

GPU 0 is used by default; for another visible GPU, run `GPU=1 bash scripts/chaosnli.sh`. Select another split with `FOLD=1 bash scripts/chaosnli.sh`.

The default configuration trains six models with seeds 42, 43, and 44. Final models are written under `outputs/chaosnli/<subset>/fold_<fold>/<model>__<label_mode>__<strategy>/seed_<seed>/final_model`; test artifacts are written beside them under `test/`. Existing final models and `test/evaluation.json` files are skipped. Set `EVALUATE=0` to train only, or `FORCE_EVAL=1` to rerun metrics while reusing existing predictions.

Future datasets should follow the same pattern: one dataset launcher and a short README subsection.

## Prediction and evaluation

Predictions from this toolkit and external models use the same JSONL contract.
`predict` is for checkpoints trained by this repository; `evaluate` is
model-agnostic and accepts any prediction file that follows the contract.
Evaluation accepts either the processed dataset directory (including
`dataset.json`) or a lightweight external ground-truth JSONL file. See
[docs/prediction_contract.md](docs/prediction_contract.md) for the required
fields, optional ignored metadata, and external-model examples.

## Models and training heads

Models are loaded with Hugging Face `AutoTokenizer`, `AutoModel`, or `AutoModelForSequenceClassification`. Standard encoder-only models such as BERT, RoBERTa, DeBERTa, XLM-R, ModernBERT, and InfoXLM are supported. Decoder-only and encoder-decoder models are not guaranteed to work with the current HLV interface.

The training entry point is `hlv_toolkits.scripts.train`. Supported heads are:

- `classification`
- `multilevel_classification` for DiscoGeM

For the current DiscoGeM experiments, `classification` and `multilevel_classification` use a linear output layer followed by softmax for human distributions. The available label-training strategies are:

- `ce`
- `mse` — MSE between the softmax-normalized prediction and the human distribution
- `jsd` — Jensen–Shannon divergence between the softmax-normalized prediction and the human distribution
- `rel` — repeated-label CE for text-pair data with raw `annotation_labels`

Model selection can use `accuracy`, `tvd`, or `kl_divergence`. Label space and label mode come from each direct dataset manifest.

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

## Shared training settings

[`configs/training.json`](configs/training.json) holds dataset-agnostic training defaults: epochs, learning rate, batch sizes, maximum sequence length, seed, and model-selection metric. Dataset manifests only define their data paths and labels; the scripts layer the shared config before the manifest.

- ChaosNLI: `bash scripts/chaosnli.sh`
├── md_agreement.sh
├── run_single_text_sweep.sh
- English DiscoGeM, all levels by default: `GPU=0 bash scripts/discogem/english.sh`
- Use `LEVEL=level1`, `level2`, or `level3` with either DiscoGeM entry to run only that level.
- ChaosNLI, default three seeds: `GPU=0 bash scripts/chaosnli.sh`
├── md_agreement.sh
├── run_single_text_sweep.sh
- ChaosNLI, only new seeds: `TRAINING_CONFIG=configs/training_new_two_seeds.json GPU=0 bash scripts/chaosnli.sh`
├── md_agreement.sh
├── run_single_text_sweep.sh
- MD-Agreement: `GPU=0 bash scripts/md_agreement.sh`
- Multilingual DiscoGeM (multilingual encoders only, all levels by default): `GPU=0 bash scripts/discogem/multilingual.sh`
- DiscoGeM multilevel (English and multilingual by default): `GPU=0 bash scripts/discogem/multilevel.sh`; use `VARIANTS=english` or `VARIANTS=multilingual` to run one variant.

Set `LEVEL=level1`, `level2`, or `level3` with either entry to restrict the run to one level.


## Dataset preparation

Prepare direct JSONL datasets with their format-specific exporters:

```bash
# DiscoGeM: text-pair and multilevel data
uv run python -m hlv_toolkits.scripts.prepare_discogem_annotation_labels

# ChaosNLI: text-pair data
bash scripts/chaosnli.sh
├── md_agreement.sh
├── run_single_text_sweep.sh

# MD-Agreement: single-text data
bash scripts/md_agreement.sh
```

Train using each generated `dataset.json`; direct data uses either
`text_pair_label_distribution`, `text_pair_multilevel_label_distribution`, or
`single_text_label_distribution`. The runnable dataset entry scripts are under
`scripts/` and `scripts/discogem/`.

## Project structure

```text
hlv_toolkits/scripts/
├── download_data.py
├── train.py
├── predict.py
├── evaluate.py
├── prepare_chaosnli_annotation_labels.py
├── prepare_discogem_annotation_labels.py
└── prepare_md_agreement_annotation_labels.py

scripts/
├── chaosnli.sh
├── md_agreement.sh
├── run_single_text_sweep.sh
├── discogem/
└── run_text_pair_sweep.sh
```

## Notes

- The repository uses fixed dataset locations by default.
- Model checkpoints are written below the selected run directory, usually under `seed_<seed>/`.
- Keep the training label level, language, head, and loss consistent between training, prediction, and evaluation.
- W&B logging is opt-in for the generic CLI and enabled by the DiscoGeM screening wrappers.
