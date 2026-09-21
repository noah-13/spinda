# Bring your own data

This guide describes how to use SPINDA with a dataset of your own. Your data
may live anywhere; `data/datasets/my_dataset/` is a convenient local default.
Do not commit large or licensed data files.

The repository's paper launchers manage their own source downloads and
preprocessing automatically. Run the relevant script in `scripts/` when
reproducing an experiment; it will use `data/raw/` and `data/datasets/` as
working locations. Dataset-specific details belong in
[the paper reproduction guide](../docs/reproducing_paper.md), not here.

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

Use `annotation_labels` to retain every individual vote. SPINDA derives the
empirical distribution from the vote counts.

| `label_mode` | Row example | Training target |
| --- | --- | --- |
| `hard` | `[1]` | One hard class; only `ce` is valid. |
| `soft` | `[0, 1, 1, 1]` | Distribution `[0.25, 0.75]`; use `ce`, `mse`, `jsd`, or `rel`. |
| `soft_to_hard` | `[0, 1, 1, 1]` | Majority-vote hard class; only `ce` is valid, while soft labels remain available for evaluation. |

Declare the mode explicitly. If it is omitted, SPINDA infers `hard` for a
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

## Other input types

SPINDA also supports single-text, multi-label, and multi-dimensional datasets.
The categorical single-text format replaces `text_a` and `text_b` with `text`
and uses `format: "single_text_label_distribution"`; its label-mode semantics
are identical. The available formats and all training options are listed in the
[configuration reference](../docs/configuration_reference.md).

For custom data that does not fit a public format yet, add a reader that emits
the normalized sample objects in `hlv_toolkits.data.schemas`, then document the
new format before relying on it in a release.
