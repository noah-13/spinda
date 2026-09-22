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

`evaluate` requires all three of these files:

- `--predictions`: a prediction JSON file following one of the output contracts
  below.
- `--input_file`: the input JSON used for inference. Its IDs must exactly match
  the prediction IDs. It can be the same file as `--human_labels`.
- `--human_labels`: an annotated JSON array with the matching IDs and text
  fields. Its annotation field depends on the prediction kind.

For example, an annotated ChaosNLI test file can serve as both input and
human-label source:

```bash
uv run spinda evaluate \
  --predictions predictions.json \
  --input_file data/datasets/chaosnli/test.json \
  --human_labels data/datasets/chaosnli/test.json \
  --output_file evaluation.json
```

The evaluator infers whether the predictions are categorical, multilabel, or
multidimensional from `outputs`; no task format or dataset directory is
required.

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
JSON forms above, then pass it to `spinda evaluate` with the matching input and
human-label files. External predictions are not loaded through `spinda predict`.
