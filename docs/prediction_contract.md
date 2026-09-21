# Prediction contract

`hlv_toolkits.scripts.evaluate` is model-agnostic: it evaluates a JSON
prediction file produced by this toolkit or by any external system. It does
not load external model checkpoints.

## Required inputs

Evaluation requires both of the following:

1. A prediction JSON file following the record format below.
2. Either a processed dataset directory containing `dataset.json` and the
   requested split (normally `test.json`), or a lightweight external
   ground-truth JSON file. The dataset manifest is the source of truth for
   the task, class order, and human labels when using `--data_dir`.

For example, to evaluate an external model on ChaosNLI:

```bash
uv run python -m hlv_toolkits.scripts.evaluate \
  --predictions external_predictions.json \
  --data_dir data/datasets/text_pair/chaosnli/snli/0 \
  --ground_truth_split test \
  --output_file results/external_model.json
```

## External ground-truth JSON

To evaluate without a repository dataset, pass `--ground_truth` instead of
`--data_dir`. Write a top-level JSON array of example objects:

```json
[
  {"id":"example-0001","label":1,"human_dist":[0.10,0.75,0.15]}
]
```

- `id` and zero-based integer `label` are required.
- `human_dist` is optional, but if it is present it must be present for every
  record. It enables distributional metrics; without it, evaluation reports
  hard-label accuracy only.
- The class-index order of `label` and `human_dist` must match the prediction
  file's `outputs.pred` and `outputs.probs`.
- Unknown fields are optional metadata and are ignored.

```bash
uv run python -m hlv_toolkits.scripts.evaluate \
  --predictions external_predictions.json \
  --ground_truth external_ground_truth.json \
  --no-plot
```

## Prediction JSON

Write a top-level JSON array. Each object requires `id`,
`outputs.probs`, and `outputs.pred`:

```json
[
  {"id":"example-0001","outputs":{"probs":[0.10,0.75,0.15],"pred":1}}
]
```

- `id` must exactly match the ID in the evaluated dataset split.
- `outputs.probs` must be a probability vector in the class order specified
  by `dataset.json`'s `labels` field. It is required for distributional
  metrics such as TVD, JSD, KL, and soft F1.
- `outputs.pred` is the zero-based predicted class index. It should normally
  be `argmax(outputs.probs)`.

This toolkit's `predict` command emits the same format, with additional
`task`, `split`, and `source` fields. Thus a toolkit prediction can be passed
to `evaluate` unchanged.

## Multidimensional prediction JSON

For manifests with format `text_pair_multidimensional_label_distribution` or
`single_text_multidimensional_label_distribution`, `predict` emits one categorical
output per level. Pass that JSON to `evaluate` unchanged:

```json
[
  {"id":"example-0001","outputs":{"dimensions":{"level1":{"probs":[0.10,0.90],"pred":1},"level2":{"probs":[0.75,0.25],"pred":0},"level3":{"probs":[0.20,0.80],"pred":1}}}}
]
```

- Each manifest-defined dimension requires `probs` and `pred`.
- Each probability-vector order is the matching `level_labels.<level>` order
  in `dataset.json`.
- Evaluation reports metrics for every level and an unweighted mean under
  `overall`.

## Optional metadata

Any top-level fields besides the required fields are optional metadata and are
ignored by metric computation. This includes `task`, `split`, `source`,
`extras`, and producer-specific fields such as `model_name`, `checkpoint`,
`seed`, `timestamp`, or `git_commit`.

Metadata is useful for provenance, but it must not be relied on to choose the
ground truth or label order; pass the intended dataset directory explicitly
with `--data_dir` instead.

## Scope

The toolkit provides prediction for checkpoints trained by this repository.
For models trained elsewhere, use their native inference code to produce the
JSON file above, then use this toolkit only for evaluation.
