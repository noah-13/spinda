# Prediction contract

`hlv_toolkits.scripts.evaluate` is model-agnostic: it evaluates a JSONL
prediction file produced by this toolkit or by any external system. It does
not load external model checkpoints.

## Required inputs

Evaluation requires both of the following:

1. A prediction JSONL file following the record format below.
2. Either a processed dataset directory containing `dataset.json` and the
   requested split (normally `test.jsonl`), or a lightweight external
   ground-truth JSONL file. The dataset manifest is the source of truth for
   the task, class order, and human labels when using `--data_dir`.

For example, to evaluate an external model on ChaosNLI:

```bash
uv run python -m hlv_toolkits.scripts.evaluate \
  --predictions external_predictions.jsonl \
  --data_dir data/processed/text_pair/chaosnli/snli/0 \
  --ground_truth_split test \
  --output_file results/external_model.json
```

## External ground-truth JSONL

To evaluate without a repository dataset, pass `--ground_truth` instead of
`--data_dir`. Write one JSON object per example:

```json
{"id":"example-0001","label":1,"human_dist":[0.10,0.75,0.15]}
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
  --predictions external_predictions.jsonl \
  --ground_truth external_ground_truth.jsonl \
  --no-plot
```

## Prediction JSONL

Write one JSON object per example. The required fields are `id`,
`outputs.probs`, and `outputs.pred`:

```json
{"id":"example-0001","outputs":{"probs":[0.10,0.75,0.15],"pred":1}}
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
JSONL file above, then use this toolkit only for evaluation.
