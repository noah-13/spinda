# Configuration reference

This page lists every user-facing option exposed by the training, prediction,
evaluation, and analysis commands. Training settings may be supplied as command-line
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

For dataset directories, `dataset.json` supplies `format`, plus `labels` or
`level_labels` and optionally `label_mode`. Multidimensional rows use raw,
aligned `annotation_labels` per dimension; distributions are derived by the
reader.

## Separate versus joint (multilevel) training

`format` is the sole selector for the input reader and model-head family.
Separate training uses `text_pair_label_distribution` (or
`single_text_label_distribution`): create one dataset manifest and independent
run per level. DiscoGeM's `english.sh` and `multilingual.sh` loop over all
levels; `LEVEL=level2` selects one.

Joint training uses `text_pair_multidimensional_label_distribution` (or its
single-text equivalent): one shared encoder feeds an independent softmax head
per dimension, and the per-dimension losses are summed. It does not create a
flat combined label space or enforce hierarchy constraints at decoding time.

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

`predict` loads a repository-trained checkpoint and writes predictions for an
arbitrary JSON input file. The input contract and output JSON forms are in the
[prediction contract](prediction_contract.md).

| Option | Accepted values | Default | Description |
|---|---|---|---|
| `--model_path` | Model/checkpoint directory | Required | Checkpoint to load. |
| `--input_file` | JSON array path | Required | Text rows to predict. |
| `--output_file` | JSON path | `predictions.json` | Prediction-output path. |
| `--batch_size` | Integer | `32` | Inference batch size. |
| `--device` | `auto`, `cpu`, `cuda`, `cuda:<index>` | CUDA if available, otherwise CPU | Inference device. |
| `--max_length` | Integer | `0` | Tokenized maximum length; `0` uses the model limit. |
| `--config` | One or more JSON paths | Optional | Merged prediction-run settings; CLI values override them. |

For repeated inference, a config is useful for stable settings such as
`model_path`, `device`, and `batch_size`. Keep the per-run input and output
paths on the command line.

## Evaluation

`evaluate` compares predictions with annotated human-label rows. It requires
matching `--predictions`, `--input_file`, and `--human_labels` files; the input
and human-label file may be the same. The evaluator infers categorical,
multilabel, or multidimensional output from `outputs`. See the
[prediction contract](prediction_contract.md#evaluate-predictions) for file
shapes and the [metric guide](evaluation_metrics.md) for metric selection.

| Option | Accepted values | Default | Description |
|---|---|---|---|
| `--predictions` | JSON path | Required | Prediction file to evaluate. |
| `--input_file` | JSON array path | Required | Prediction input; IDs must exactly match the prediction IDs. |
| `--human_labels` | Annotated JSON array path | Required | Human annotation votes or distributions. |
| `--output_file` | JSON path | `outputs/evaluation/results/<stem>__eval.json` | Aggregate-metrics output path. |
| `--metrics` | One or more metric names | All supported metrics | Retain only the named metrics. |
| `--config` | One or more JSON paths | Optional | Merged evaluation settings; CLI values override them. |
| `--analysis` | Flag | Disabled | Write disagreement-stratified metrics and per-instance errors for categorical distribution labels. |
| `--disagreement-groups` | Integer | `3` | Number of human-disagreement strata. |
| `--disagreement-boundaries` | One or more floats | Automatic | Explicit normalized-entropy cutoffs. |
| `--analysis-output-file` | JSON path | Next to evaluation output | Analysis-report path. |
| `--instance-errors-file` | CSV path | Next to analysis output | Per-instance error-table path. |
| `--ternary-plot` / `--no-ternary-plot` | Boolean | Disabled | Write ternary diagnostics for three-class categorical soft labels. |
| `--ternary-plot-dir` | Directory path | `outputs/evaluation/figures` | Ternary-artifact directory. |
| `--ternary-plot-title` | String | Prediction filename | Ternary-plot title. |
| `--ternary-source` | `model`, `human`, `both` | `both` | Distribution source rendered in ternary plots. |
| `--ternary-browser` / `--no-ternary-browser` | Boolean | Enabled with ternary plot | Also write interactive HTML. |

## Analysis

`analyze` pools artifacts created by `evaluate --analysis` across seed runs and writes one or both selected-metric plots: disagreement-stratified and instance-level.

| Option | Accepted values | Default | Description |
|---|---|---|---|
| `--run-dirs` | One or more directories | One of this or `--analysis-files` is required | Strategy roots; discovers `seed_*/test/evaluation__analysis.json` and matching CSV files. |
| `--analysis-files` | One or more JSON paths | One of this or `--run-dirs` is required | Explicit analysis reports. |
| `--instance-errors-files` | One or more CSV paths | Required for explicit artifacts only when `--plots` includes `instance` | CSV paired with each explicit analysis report. Not needed for `stratified` only. |
| `--labels` | One or more strings | Run-directory names | Display label per strategy or explicit artifact pair. |
| `--metric` | `tvd`, `jsd`, `kl`, `ce`, `l2` | `tvd` | Metric plotted in either output type. |
| `--plots` | One or both of `stratified`, `instance` | Both | Output figure types; `stratified` needs only analysis JSON, while `instance` also needs CSV errors. |
| `--level` | Dimension name | `null` | Dimension to plot for multidimensional analyses. |
| `--output-dir` | Directory path | `outputs/evaluation/disagreement_analysis` | Plot-output directory; writes `disagreement_<metric>` for stratified and `instance_<metric>_violin` for instance plots (PNG and PDF). |
| `--title` | String | `null` | Optional plot title. |

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
