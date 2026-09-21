# Configuration reference

This page lists every user-facing option exposed by the training, prediction,
and evaluation commands. Training settings may be supplied as command-line
arguments or in JSON files passed through `--config`. The training target type
is determined only by `label_mode`; there is no separate soft-label switch.

## Configuration precedence

Pass one or more files with `--config FILE [FILE ...]`. Files are merged
**left to right**, so a key in a later file replaces the same key in an earlier
file. Explicit CLI arguments override values from every config file.

In JSON, use the public keys `format` and `label_training_strategy`. They map
to the command-line destinations `data_format` and `soft_label_loss`;
do not supply both names for either setting.

## Dataset and input options

| Key / CLI option | Accepted values | Default | Description |
|---|---|---|---|
| `config` / `--config` | One or more JSON paths | `[]` | Training configuration layers. |
| `format` / `--format` | `text_pair_label_distribution`, `single_text_label_distribution`, `single_text_multilabel_annotation_distribution`, `text_pair_multidimensional_label_distribution`, `single_text_multidimensional_label_distribution` | Required | Dataset contract; selects the input reader and model-head family. |
| `train_path` / `--train_path` | JSON path | Required | Training data path. |
| `dev_path` / `--dev_path` | JSON path or `null` | `null` | Development data path; omit to train without development evaluation. |
| `labels` / `--labels` | Ordered list of strings | `null` | Label names for categorical and multi-label data. |
| `level_labels` / `--level_labels` | JSON object `{dimension: [label, ...]}` | `null` | Optional multidimensional label mapping; must match the manifest exactly. |
| `label_mode` / `--label_mode` | `hard`, `soft`, `soft_to_hard` | `null` | Hard targets, aggregated soft targets, or majority-vote targets with retained soft evaluation. |
| `data_source` / `--data_source` | `text_pair`, `single_text`, `single_text_multilabel`, `multilevel` | `text_pair` | Internal route; `format` determines the effective value, so normally leave this unset. |

For dataset directories, `dataset.json` supplies `format`, plus `labels` or
`level_labels` and optionally `label_mode`. Multidimensional rows use raw,
aligned `annotation_labels` per dimension; distributions are derived by the
reader.

## Model and training objectives

| Key / CLI option | Accepted values | Default | Description |
|---|---|---|---|
| `model` / `--model` | Hugging Face model ID or local checkpoint path | `roberta-base` | Pretrained encoder to fine-tune. |
| `label_training_strategy` / `--label_training_strategy` | `ce`, `mse`, `jsd`, `rel` | `ce` | Loss: cross-entropy, MSE, Jensen--Shannon divergence, or repeated-label learning. |
| `soft_label_metric_for_best_model` / `--soft_label_metric_for_best_model` | `accuracy`, `tvd`, `kl_divergence`, `soft_micro_f1`, `soft_macro_f1` | `tvd` | Development selection metric for categorical soft-label tasks. |
| `multilabel_metric_for_best_model` / `--multilabel_metric_for_best_model` | `accuracy`, `soft_micro_f1`, `soft_macro_f1`, `multilabel_pojsd`, `multilabel_entropy_correlation` | `soft_micro_f1` | Development selection metric for multi-label tasks. |

`hard` and `soft_to_hard` training support only `ce`. `rel` requires soft data
with retained individual annotations. For multi-label data, `soft_to_hard`
also supports only `ce`.

## Optimization

| Key / CLI option | Accepted values | Default | Description |
|---|---|---|---|
| `learning_rate` / `--learning_rate` | Float | `2e-5` | Optimizer learning rate. |
| `train_batch_size` / `--train_batch_size` | Integer | `32` | Per-device training batch size. |
| `eval_batch_size` / `--eval_batch_size` | Integer | `64` | Per-device evaluation batch size. |
| `num_epochs` / `--num_epochs` | Integer | `3` | Number of fine-tuning epochs. |
| `warmup_ratio` / `--warmup_ratio` | Float | `0.1` | Fraction of training used for learning-rate warmup. |
| `gradient_accumulation_steps` / `--gradient_accumulation_steps` | Integer | `1` | Batches accumulated before an optimizer update. |
| `max_length` / `--max_length` | Integer | `0` | Tokenized maximum length; `0` resolves to the model/tokenizer limit. |
| `seeds` / `--seeds` | One or more integers | `[42]` | Independent random seeds to train. |

## Prediction

Prediction is CLI-only; it does not read a JSON configuration file.

| CLI option | Accepted values | Default | Description |
|---|---|---|---|
| `--model_path` | Model/checkpoint directory | Required | Fine-tuned model to load. |
| `--data_dir` | Processed dataset directory | Required | Directory containing `dataset.json` and the requested split. |
| `--split` | `train`, `dev`, `test` | `test` | Split to predict. |
| `--output_file` | JSON path | `predictions.json` | Prediction-output path. |
| `--batch_size` | Integer | `32` | Inference batch size. |
| `--device` | `auto`, `cpu`, `cuda`, `cuda:<index>` | CUDA if available, otherwise CPU | Inference device. |
| `--max_length` | Integer | `0` | Tokenized maximum length; `0` uses the model/tokenizer limit. |

See [prediction_contract.md](prediction_contract.md) for output JSON schemas.

## Evaluation and analysis

| CLI option | Accepted values | Default | Description |
|---|---|---|---|
| `--predictions` | Prediction JSON path | Required | Predictions to evaluate. |
| `--predictions_format` | `json` | `json` | Prediction format; JSON is the only implemented format. |
| `--data_dir` | Processed dataset directory | One of this or `--ground_truth` is required | Loads ground truth and manifest-defined label order. |
| `--ground_truth` | External JSON path | One of this or `--data_dir` is required | Lightweight external ground-truth source. |
| `--ground_truth_split` | `train`, `dev`, `test` | `test` | Split to load from `--data_dir`. |
| `--output_file` | JSON path or `null` | Auto-generated | Evaluation-results path. |
| `--plot` / `--no-plot` | Boolean | `true` | Enable or disable optional ternary diagnostics for three-class soft-label data. |
| `--plot_dir` | Path or `null` | Auto-generated | Directory for ternary outputs. |
| `--plot_title` | String or `null` | Prediction filename | Ternary plot title. |
| `--ternary_source` | `model`, `human`, `both` | `both` | Probability source displayed in ternary plots. |
| `--ternary_browser` / `--no-ternary_browser` | Boolean | `true` | Also write an interactive HTML ternary plot with per-instance hover details. |
| `--analysis` | Flag | `false` | Write disagreement-stratified metrics and instance-level errors. |
| `--disagreement_groups` | Integer | `3` | Number of entropy/disagreement strata. |
| `--disagreement_boundaries` | One or more floats | `null` | Explicit normalized-entropy cutoffs; provide `groups - 1` values. |
| `--analysis-output-file` | JSON path or `null` | Next to evaluation output | Aggregate disagreement-analysis path. |
| `--instance-errors-file` | CSV path or `null` | Next to analysis output | Per-instance error-table path. |

## Seed-aggregated disagreement plots

After producing categorical evaluation artifacts with `evaluate --analysis`, run
`spinda analyze --run-dirs` with one strategy root per label. It discovers every
`seed_*/test` analysis JSON and instance-error CSV pair and creates two PNG/PDF
diagnostics: group-level TVD with standard-deviation error bars, and an
instance-level TVD violin plot with an embedded box plot. All inputs must use
the same grouping; pass `--level` for one multi-dimensional annotation level.

## Advanced and runtime options

| Key / CLI option | Accepted values | Default | Description |
|---|---|---|---|
| `device` / `--device` (training) | `auto`, `cpu`, `cuda`, `cuda:<index>` | `auto` | Training device. |
| `fp16` / `--fp16`, `--no-fp16` | Boolean | `false` | Mixed-precision training. |
| `dataloader_num_workers` / `--dataloader_num_workers` | Integer | `2` | DataLoader worker count; use `0` to disable multiprocessing. |
| `output_dir` / `--output_dir` | Path | `./outputs` | Root directory for seed runs, checkpoints, and final models. |
| `resume_from_checkpoint` / `--resume_from_checkpoint` | Checkpoint path or `null` | `null` | Resume a seed run from a Hugging Face checkpoint. |
| `use_wandb` / `--use_wandb`, `--no-use_wandb` | Boolean | `false` | Enable Weights & Biases logging. |
| `wandb_project` / `--wandb_project` | String | `hlv` | Weights & Biases project. |
| `wandb_entity` / `--wandb_entity` | String or `null` | `null` | Weights & Biases entity/team. |
| `wandb_group` / `--wandb_group` | String or `null` | `null` | Weights & Biases run group. |
| `wandb_job_type` / `--wandb_job_type` | String or `null` | `null` | Weights & Biases job-type label. |
| `wandb_run_name` / `--wandb_run_name` | String or `null` | `null` | Explicit run name; otherwise generated automatically. |
