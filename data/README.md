# Bring your own data

This guide describes how to use SPInDa with a dataset of your own. Your data
may live anywhere; `data/datasets/my_dataset/` is a convenient local default.
Do not commit large or licensed data files.

The repository's paper launchers manage their own source downloads and
preprocessing automatically. Run the relevant script in `scripts/` when
reproducing an experiment; it will use `data/raw/` and `data/datasets/` as
working locations. Dataset-specific details belong in
[the paper reproduction guide](../docs/reproducing_paper.md), not here.

## Choose a format

All public formats use `dataset.json` plus `train.json`, `dev.json`, and
optionally `test.json`; every split is a top-level JSON array. Choose the
format from the input shape and annotation structure.

| Format | Inputs | Annotation field | Use when |
| --- | --- | --- | --- |
| `text_pair_label_distribution` | `text_a`, `text_b` | `annotation_labels: [int, ...]` | One categorical label for a text pair. |
| `single_text_label_distribution` | `text` | `annotation_labels: [int, ...]` | One categorical label for one text. |
| `single_text_multilabel_annotation_distribution` | `text` | `annotation_label_sets: [[int, ...], ...]` | Each annotator may choose zero or more labels. |
| `text_pair_multidimensional_label_distribution` | `text_a`, `text_b` | `annotation_labels: {dimension: [int, ...]}` | A text pair has several categorical annotation dimensions. |
| `single_text_multidimensional_label_distribution` | `text` | `annotation_labels: {dimension: [int, ...]}` | One text has several categorical annotation dimensions. |

`id` is required and must be unique within a split. Label indices are always
zero-based and refer to the order declared in `labels` or `level_labels`.

## A categorical text-pair dataset

For NLI, paraphrase detection, or any task with two input texts, create one
directory containing a manifest and JSON-array splits:

```text
data/datasets/my_dataset/
├── dataset.json
├── train.json
├── dev.json
└── test.json       # needed for prediction or test evaluation
```

`dataset.json` fixes the class-index order and declares how labels should be
interpreted. A hard-label binary task, for example, can use:

```json
{
  "format": "text_pair_label_distribution",
  "label_mode": "hard",
  "labels": ["not_duplicate", "duplicate"],
  "train_path": "data/datasets/my_dataset/train.json",
  "dev_path": "data/datasets/my_dataset/dev.json"
}
```

Every split is a top-level JSON array. Text-pair records require `id`,
`text_a`, `text_b`, and `annotation_labels`:

```json
[
  {
    "id": "train-0001",
    "text_a": "How do I reset my password?",
    "text_b": "What is the password reset procedure?",
    "annotation_labels": [1]
  }
]
```

The integer values in `annotation_labels` refer to the order in `labels`.
Extra fields are retained as metadata but do not affect label parsing.

## Human-label variation: choose a label mode

Use `annotation_labels` to retain every individual vote. SPInDa derives the
empirical distribution from the vote counts.

| `label_mode` | Row example | Training target |
| --- | --- | --- |
| `hard` | `[1]` | One hard class; only `ce` is valid. |
| `soft` | `[0, 1, 1, 1]` | Distribution `[0.25, 0.75]`; use `ce`, `mse`, `jsd`, or `rel`. |
| `soft_to_hard` | `[0, 1, 1, 1]` | Majority-vote hard class; only `ce` is valid, while soft labels remain available for evaluation. |

Declare the mode explicitly. If it is omitted, SPInDa infers `hard` for a
single vote and `soft` for multiple votes and emits a warning. `rel`
(repeated-label learning) is useful when the individual annotations—not only
their normalized aggregate—should affect the objective.

## Train, predict, and evaluate

Layer the shared optimization defaults before the dataset manifest. Later
configuration files override earlier ones, and explicit CLI options override
both.

```bash
uv run spinda train \
  --config configs/training.json data/datasets/my_dataset/dataset.json \
  --model roberta-base \
  --label_training_strategy ce \
  --seeds 42 43 44 \
  --output_dir outputs/my_dataset
```

Use the saved model for prediction, then evaluate it against the directory that
contains `dataset.json` and `test.json`:

```bash
uv run spinda predict \
  --model_path outputs/my_dataset/seed_42/final_model \
  --data_dir data/datasets/my_dataset \
  --split test \
  --output_file outputs/my_dataset/seed_42/test/predictions.json

uv run spinda evaluate \
  --predictions outputs/my_dataset/seed_42/test/predictions.json \
  --data_dir data/datasets/my_dataset \
  --analysis \
  --output_file outputs/my_dataset/seed_42/test/evaluation.json
```

For an external model, write its predictions in the same contract and run only
the evaluation command. See the
[prediction contract](../docs/prediction_contract.md) for the required output
schema.

## A categorical single-text dataset

Use `single_text_label_distribution` when each row has one input text rather
than a pair. Its `label_mode` and `annotation_labels` rules are exactly the
same as the text-pair format.

```json
{
  "format": "single_text_label_distribution",
  "label_mode": "soft",
  "labels": ["not_offensive", "offensive"],
  "train_path": "data/datasets/my_single_text/train.json",
  "dev_path": "data/datasets/my_single_text/dev.json"
}
```

```json
[
  {
    "id": "train-0001",
    "text": "Example text.",
    "annotation_labels": [0, 0, 0, 1, 1]
  }
]
```

## A multi-label single-text dataset

Use `single_text_multilabel_annotation_distribution` when each annotator can
select more than one label. `annotation_label_sets` has one list per
annotator; an empty inner list is valid and means that annotator selected no
labels. SPInDa derives one independent Bernoulli probability per label, so the
probabilities do not need to sum to one. Its rows do not use a categorical
label mode. For training, set `label_mode` to `soft` for probability targets or
`soft_to_hard` for per-label majority-vote targets; the latter permits only
`ce`.

```json
{
  "format": "single_text_multilabel_annotation_distribution",
  "label_mode": "soft",
  "labels": ["care", "fairness", "loyalty"],
  "train_path": "data/datasets/my_multilabel/train.json",
  "dev_path": "data/datasets/my_multilabel/dev.json"
}
```

```json
[
  {
    "id": "train-0001",
    "text": "Example comment.",
    "annotation_label_sets": [[0, 1], [0], []]
  }
]
```

Here the resulting human probabilities are `[2/3, 1/3, 0]`.

## A multi-dimensional dataset

Use a multi-dimensional format when every row has several categorical
annotation dimensions. Declare a separate ordered label vocabulary for each
dimension in `level_labels`. Each row must provide every declared dimension,
and the vote lists must have the same length: position *i* across dimensions
comes from the same annotator.

For text pairs, use `text_pair_multidimensional_label_distribution`:

```json
{
  "format": "text_pair_multidimensional_label_distribution",
  "label_mode": "soft",
  "level_labels": {
    "sentiment": ["negative", "positive"],
    "topic": ["billing", "technical", "other"]
  },
  "train_path": "data/datasets/my_dimensions/train.json",
  "dev_path": "data/datasets/my_dimensions/dev.json"
}
```

```json
[
  {
    "id": "train-0001",
    "text_a": "The service disconnected repeatedly.",
    "text_b": "The connection was unreliable.",
    "annotation_labels": {
      "sentiment": [0, 0, 1],
      "topic": [1, 1, 1]
    }
  }
]
```

For one-text inputs, change only the format and input fields:

```json
{
  "format": "single_text_multidimensional_label_distribution",
  "label_mode": "soft",
  "level_labels": {
    "sentiment": ["negative", "positive"],
    "topic": ["billing", "technical", "other"]
  },
  "train_path": "data/datasets/my_dimensions_single/train.json",
  "dev_path": "data/datasets/my_dimensions_single/dev.json"
}
```

```json
[
  {
    "id": "train-0001",
    "text": "The service disconnected repeatedly.",
    "annotation_labels": {
      "sentiment": [0, 0, 1],
      "topic": [1, 1, 1]
    }
  }
]
```

## Validation and extensions

Use `spinda train` with the manifest as a configuration file; the readers
validate the format-specific required fields, index ranges, label vocabularies,
and annotation alignment before training. See the
[configuration reference](../docs/configuration_reference.md) for objectives
and all training options.

For custom data that does not fit a public format yet, add a reader that emits
the normalized sample objects in `hlv_toolkits.data.schemas`, then document the
new format before relying on it in a release.
