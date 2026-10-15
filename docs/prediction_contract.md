# Prediction contract

`spinda predict` generates predictions from a checkpoint trained by this
repository. `spinda evaluate` is model-agnostic: it evaluates prediction JSON
from SPInDa or any external system, and never loads a model checkpoint.

## Generate predictions with SPInDa

`predict` needs a checkpoint and an input JSON array. Each row must have a
unique string `id` and exactly one input shape:

- Text pair: `text_a` and `text_b`, both strings.
- Single text: `text`, a string.

Extra fields, including annotations, are ignored during prediction.

```json
[
  {
    "id": "example-0001",
    "text_a": "A dog is running.",
    "text_b": "An animal is moving."
  }
]
```

```bash
uv run spinda predict \
  --model_path outputs/model/seed_42/final_model \
  --input_file input.json \
  --output_file predictions.json
```

The command writes an object with a `predictions` array. It may also include
checkpoint label names in a top-level `labels` field.

## Evaluate predictions

`evaluate` requires a prediction file and exactly one ground-truth source:

- `--predictions`: a prediction JSON file following one of the output contracts
  below.
- `--data_dir`: a processed dataset directory containing `dataset.json` and
  the requested split file (`test.json` by default). It supplies the task,
  class order, and human annotations.
- `--ground_truth`: an external categorical JSON array, used instead of
  `--data_dir`. Each row requires `id` and a zero-based integer `label`; add
  a numeric `human_dist` to every row for distributional metrics.

For example, evaluate a ChaosNLI test split from its processed dataset
directory:

```bash
uv run spinda evaluate \
  --predictions predictions.json \
  --data_dir data/datasets/chaosnli \
  --metrics tvd kl entropy_correlation \
  --output_file evaluation.json
```

The evaluator loads categorical, multilabel, or multidimensional labels from
the manifest's declared format. Pass `--ground_truth_split train` or
`--ground_truth_split dev` to select another split.

## Prediction JSON

The evaluator accepts either a top-level array or an object with a
`predictions` array. Every prediction record needs an `id` and `outputs`.
Top-level record fields such as `source`, `task`, and `split` are optional
metadata and do not affect metrics.

### Categorical output

```json
[
  {
    "id": "example-0001",
    "outputs": {"probs": [0.10, 0.75, 0.15], "pred": 1}
  }
]
```

`outputs.probs` is a probability vector; `outputs.pred` is its zero-based
predicted class, normally `argmax(probs)`. The probability and class indices
must use the same order as the integer votes in `annotation_labels`.

The corresponding human-label row has a non-empty integer `annotation_labels`
array:

```json
{
  "id": "example-0001",
  "text_a": "A dog is running.",
  "text_b": "An animal is moving.",
  "annotation_labels": [1, 1, 0, 2]
}
```

### Multilabel output

Multilabel predictions use the same record structure. `outputs.probs` and
`outputs.pred` are same-length vectors in label order; `pred` has a zero-or-one
decision for each label.

```json
[
  {
    "id": "example-0001",
    "outputs": {"probs": [0.90, 0.20, 0.55], "pred": [1, 0, 1]}
  }
]
```

The matching human-label row uses `annotation_label_sets`: a non-empty array
of annotation votes, with each vote expressed as the list of active zero-based
labels.

```json
{
  "id": "example-0001",
  "text": "An example document.",
  "annotation_label_sets": [[0, 2], [0], [0, 2]]
}
```

### Multidimensional output

Multidimensional predictions have one categorical output per named dimension.
Each dimension must appear in every record.

```json
[
  {
    "id": "example-0001",
    "outputs": {
      "dimensions": {
        "level1": {"probs": [0.10, 0.90], "pred": 1},
        "level2": {"probs": [0.75, 0.25], "pred": 0}
      }
    }
  }
]
```

The corresponding `annotation_labels` value is an object with the same
ordered dimensions, each containing a non-empty array of integer votes:

```json
{
  "id": "example-0001",
  "text_a": "A dog is running.",
  "text_b": "An animal is moving.",
  "annotation_labels": {
    "level1": [1, 1, 0],
    "level2": [0, 0, 1]
  }
}
```

Evaluation reports metrics for every dimension and their unweighted mean under
`overall`.

## Optional artifacts

Add `--analysis` to write disagreement-stratified metrics and per-instance
errors. It requires categorical distribution labels, including each dimension
of a multidimensional task. Add `--ternary-plot` to write ternary diagnostics
for three-class categorical soft-label predictions; it is disabled by default.

## External models

Use the external model native inference code to produce one of the prediction
JSON forms above, then pass it to `spinda evaluate` with the intended
`--data_dir` or `--ground_truth`. External predictions are not loaded through
`spinda predict`.
