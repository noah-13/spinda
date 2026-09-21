# SPINDA

> Simple Prediction and Interpretation of Data with Human Label Variation

SPINDA is a toolkit for training, evaluating, and interpreting NLP models when
multiple humans label the same example. It works with hard labels, empirical
label distributions, multi-label tasks, and multi-dimensional annotations.

## Install

```bash
uv sync
```

Use `uv run spinda <command>` from the repository root. The main commands are
`train`, `predict`, `evaluate`, and `analyze`.

## Start with ChaosNLI

ChaosNLI is a three-way NLI task with many annotations per example. SPINDA
keeps the original votes rather than reducing them to one label. Its prepared
layout is:

```text
data/datasets/text_pair/chaosnli/mnli_m/0/
├── dataset.json
├── train.json
├── dev.json
└── test.json
```

The manifest declares the task format and the class order:

```json
{
  "format": "text_pair_label_distribution",
  "labels": ["entailment", "neutral", "contradiction"]
}
```

A row contains the two texts and every annotator's label index. For example,
the following row represents two entailment votes, one neutral vote, and one
contradiction vote; the reader derives the human distribution `[0.50, 0.25,
0.25]` automatically.

```json
{
  "id": "example-1",
  "text_a": "The oil filter is within reach.",
  "text_b": "It is worth trying to remove the oil filter.",
  "annotation_labels": [0, 0, 1, 2]
}
```

See [data/README.md](data/README.md) for the complete data layout and the
public dataset format.

### Run one ChaosNLI configuration

The launcher prepares missing ChaosNLI data and uses the project defaults. This
command restricts the sweep to one model, one loss, and one seed:

```bash
SUBSETS=mnli_m MODEL_SPECS=roberta-base RUN_SPECS='soft rel' SEEDS_OVERRIDE=42 GPU=0 \
  bash scripts/chaosnli.sh
```

For direct training on an already prepared split, the essential configuration
is the data paths, class order, `label_mode`, learning objective, seed, and
output directory:

```bash
uv run spinda train \
  --config configs/training.json \
  --model roberta-base \
  --format text_pair_label_distribution \
  --train_path data/datasets/text_pair/chaosnli/mnli_m/0/train.json \
  --dev_path data/datasets/text_pair/chaosnli/mnli_m/0/dev.json \
  --labels entailment neutral contradiction \
  --label_mode soft \
  --label_training_strategy rel \
  --seeds 42 \
  --output_dir outputs/chaosnli_example
```

`soft` trains against the empirical label distribution. Choose `ce`, `mse`,
`jsd`, or `rel` as the training objective; `rel` uses the individual annotation
votes as repeated hard-label observations. The full option reference and
configuration precedence are in
[docs/configuration_reference.md](docs/configuration_reference.md).

## Evaluate and analyze a run

First predict with a saved checkpoint, then evaluate the prediction file. The
evaluation contract is model-independent, so external models can use the same
analysis after exporting the required JSON/JSONL format.

```bash
uv run spinda predict \
  --model_path outputs/chaosnli_example/seed_42/final_model \
  --data_dir data/datasets/text_pair/chaosnli/mnli_m/0 \
  --split test \
  --output_file outputs/chaosnli_example/seed_42/test/predictions.jsonl

uv run spinda evaluate \
  --predictions outputs/chaosnli_example/seed_42/test/predictions.jsonl \
  --data_dir data/datasets/text_pair/chaosnli/mnli_m/0 \
  --analysis --no-plot \
  --output_file outputs/chaosnli_example/seed_42/test/evaluation.json
```

`evaluate --analysis` writes aggregate metrics, a disagreement-stratified
report, and a per-instance error table. To compare strategies across several
seeds, pass one run directory per strategy. SPINDA discovers the matching
analysis artifacts beneath each `seed_*/test/` directory.

```bash
uv run spinda analyze --run-dirs \
  outputs/chaosnli/mnli_m/fold_0/roberta-base__soft_to_hard__ce \
  outputs/chaosnli/mnli_m/fold_0/roberta-base__soft__rel \
  --labels Hard_CE ReL \
  --output-dir outputs/disagreement_analysis
```

This produces group-level TVD comparisons with cross-seed error bars and
instance-level TVD violin plots.

| Disagreement-stratified TVD | Instance-level TVD |
| --- | --- |
| ![TVD by disagreement level](docs/assets/disagreement_tvd_example.png) | ![Instance-level TVD violin plot](docs/assets/instance_tvd_violin_example.png) |

For three-class soft-label data, `evaluate` can also write a static ternary PNG
and an interactive HTML diagnostic with per-instance hover details. It is
optional: use `--no-plot` to skip it, or `--no-ternary-browser` to write only
the PNG.

## Learn more

- [Data layout and public dataset format](data/README.md)
- [Configuration reference](docs/configuration_reference.md)
- [Prediction JSON/JSONL contract](docs/prediction_contract.md)
- [HLV metric tutorial](docs/hlv_metrics_tutorial.md)

SPINDA supports text-pair and single-text classification, multi-label data, and
multi-dimensional annotations. The repository's canonical launchers for
ChaosNLI, DiscoGeM, MD-Agreement, MultiPICo, TGeGUM/Humans-and-Domains, and
MFRC are listed in the paper guide below.

## Reproduce the paper

[docs/reproducing_paper.md](docs/reproducing_paper.md) documents the supported
paper launchers, default multi-seed settings, and dataset-specific experiment
scope. It is intentionally separate from the general workflow above.

## License

SPINDA is released under the [MIT License](LICENSE).
