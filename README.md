# SPINDA

> Simple Prediction and Interpretation of Data with Human Label Variation

SPINDA is a toolkit for training, predicting, evaluating, and interpreting NLP models with Human Label Variation (HLV). It supports single-label, multi-label, and multi-dimensional annotations; distribution-aware metrics; and disagreement-stratified, instance-level analysis.

Start with [the paper reproduction guide](docs/reproducing_paper.md), [the configuration reference](docs/configuration_reference.md), and [the prediction contract](docs/prediction_contract.md). See [the paper-to-code consistency record](docs/paper_consistency.md) for release scope and metadata still needed.

## Datasets

- **SNLI**: hard-label NLI data loaded through Hugging Face `datasets`.
- **ChaosNLI**: converted from its official SNLI JSONL into the shared text-pair `annotation_labels` format.
- **DiscoGeM 2.0**: English, German, French, and Czech discourse-relation data with hierarchical labels and human distributions.
- **Processed JSON**: the normalized format used by the training and evaluation scripts.
- **MFRC**: Reddit moral-foundation annotations in a dedicated multi-label, per-annotator single-text format.
- **MultiPICo**: multilingual post/reply irony annotations aggregated into per-conversation label distributions.
- **Text-pair classification JSON**: a fixed public format for training on user-provided hard-label datasets; see [data/README.md](data/README.md#public-text-pair-classification-format).

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

Predictions from this toolkit and external models use the same JSON contract.
`predict` is for checkpoints trained by this repository; `evaluate` is
model-agnostic and accepts any prediction file that follows the contract.
Evaluation accepts either the processed dataset directory (including
`dataset.json`) or a lightweight external ground-truth JSON file. See
[docs/prediction_contract.md](docs/prediction_contract.md) for the required
fields, optional ignored metadata, and external-model examples.

See [docs/configuration_reference.md](docs/configuration_reference.md) for all training, prediction, evaluation, and analysis options.


For categorical soft-label data, add `--analysis` to write an entropy-stratified
report and a CSV with TVD, JSD, KL, cross-entropy, and L2 error for every
instance. The default low/medium/high groups use empirical entropy tertiles.
Use `--disagreement-groups N` for N quantile groups, or provide explicit
normalized-entropy cutoffs with `--disagreement-boundaries`:

```bash
uv run python -m hlv_toolkits.scripts.evaluate \
  --predictions outputs/predictions.json --data_dir data/processed/text_pair/chaosnli \
  --analysis --disagreement-boundaries 0.33 0.67
```

The analysis JSON contains group-level accuracy and mean distribution errors;
the adjacent `__instance_errors.csv` supports sorting and filtering the
individual model--human mismatches.

## Models and training heads

Models are loaded with Hugging Face `AutoTokenizer`, `AutoModel`, or `AutoModelForSequenceClassification`. Standard encoder-only models such as BERT, RoBERTa, DeBERTa, XLM-R, ModernBERT, and InfoXLM are supported. Decoder-only and encoder-decoder models are not guaranteed to work with the current HLV interface.

The training entry point is `hlv_toolkits.scripts.train`. Supported heads are:

- `classification`
- `multidimensional_classification` for DiscoGeM

For the current DiscoGeM experiments, `classification` and `multidimensional_classification` use a linear output layer followed by softmax for human distributions. The available label-training strategies are:

- `ce`
- `mse` — MSE between the softmax-normalized prediction and the human distribution
- `jsd` — Jensen–Shannon divergence between the softmax-normalized prediction and the human distribution
- `rel` — repeated-label CE for text-pair data with raw `annotation_labels`

Model selection can use `accuracy`, `tvd`, or `kl_divergence`. Label space and label mode come from each direct dataset manifest.

For normalized categorical human-label distributions, evaluation reports these
metrics by default: `accuracy`, `tvd`, `jsd`, `kl`, `ce`, `l2`,
`entropy_correlation`, and `distance_correlation`. Related compatibility
metrics such as `l1`, `pojsd`, and `soft_accuracy` remain available by passing
`distribution_metrics` to `Evaluator`, but are not included in the default
result table.

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
- MFRC: `GPU=0 bash scripts/mfrc.sh`
- Humans-and-Domains/TGeGUM: `GPU=0 bash scripts/humans_and_domains.sh`
- MultiPICo multilingual: `GPU=0 bash scripts/multipico/multilingual.sh`
- MultiPICo English-only: `GPU=0 bash scripts/multipico/english.sh`
- Multilingual DiscoGeM (multilingual encoders only, all levels by default): `GPU=0 bash scripts/discogem/multilingual.sh`
- DiscoGeM multidimensional (English and multilingual by default): `GPU=0 bash scripts/discogem/multilevel.sh`; use `VARIANTS=english` or `VARIANTS=multilingual` to run one variant.

Set `LEVEL=level1`, `level2`, or `level3` with either entry to restrict the run to one level.


## Dataset preparation

Prepare direct JSON datasets with their format-specific exporters:

```bash
# DiscoGeM: text-pair and multidimensional data
uv run python -m hlv_toolkits.scripts.prepare_discogem_annotation_labels

# ChaosNLI: text-pair data
bash scripts/chaosnli.sh
├── md_agreement.sh
├── run_single_text_sweep.sh

# MD-Agreement: single-text data
bash scripts/md_agreement.sh
```

Train using each generated `dataset.json`; direct data uses either
`text_pair_label_distribution`, `text_pair_multidimensional_label_distribution`, or
`single_text_label_distribution`. The runnable dataset entry scripts are under
`scripts/` and `scripts/discogem/`.

## MultiPICo

MultiPICo is downloaded from the official LeWiDi MP release, which provides
the benchmark train/dev/test split (12,017 / 3,005 / 3,756 conversations).
The exporter preserves each conversation's original annotator votes and writes
a soft `text_pair_label_distribution` dataset to
`data/processed/text_pair/multipico/`.

```bash
GPU=0 bash scripts/multipico/multilingual.sh
# English-only counterpart
GPU=0 bash scripts/multipico/english.sh
```

The launcher sweeps `soft ce`, `soft mse`, `soft jsd`, `soft rel`, and
`soft_to_hard ce`. Its generated manifest fixes best-model selection to
development-set TVD.

## Humans-and-Domains / TGeGUM

The official sentence-level TGeGUM splits are exported as three independent soft-label tasks (`genre`, `topic1`, and `topic2`) under `data/processed/single_text/humans_and_domains/`, plus one shared three-head experiment under `data/processed/single_text/humans_and_domains/multidimensional/`. The latter maps `level1=genre`, `level2=topic1`, and `level3=topic2`; only the two topic heads form a true hierarchy. All runs select checkpoints with development-set TVD.

```bash
GPU=0 bash scripts/humans_and_domains.sh
# Restrict work, for example:
TASKS="genre topic1" GPU=0 bash scripts/humans_and_domains.sh
```

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
