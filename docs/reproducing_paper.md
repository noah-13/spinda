# Reproducing the SPINDA experiments

This page defines the supported public experiment interface for the SPINDA paper.
Each launcher prepares data as needed, trains the documented sweep, and writes
runs below `outputs/`, which is ignored by Git. Run commands from the repository root.

## Install and smoke test

```bash
uv sync
MODEL_SPECS=roberta-base RUN_SPECS='soft ce' SEEDS_OVERRIDE=42 GPU=0 \
  bash scripts/chaosnli.sh
```

The paper configuration defaults, including seeds 42, 43, and 44, are in
[`configs/training.json`](../configs/training.json). Launchers accept `GPU`,
`FORCE=1`, and `SEEDS_OVERRIDE="42 43 44"`. Upstream dataset download may
require accepting its terms; raw and processed datasets are Git-ignored.

## Canonical launchers

| Paper setting | Launcher |
| --- | --- |
| ChaosNLI (SNLI and MNLI-M) | `bash scripts/chaosnli.sh` |
| MD-Agreement | `bash scripts/md_agreement.sh` |
| MultiPICo (English / multilingual) | `bash scripts/multipico/english.sh` / `bash scripts/multipico/multilingual.sh` |
| DiscoGeM 2.0 (English / multilingual / joint) | `bash scripts/discogem/english.sh` / `multilingual.sh` / `multilevel.sh` |
| TGeGUM / Humans-and-Domains | `bash scripts/humans_and_domains.sh` |
| MFRC multi-label | `bash scripts/mfrc.sh` |

Use `LEVEL=level1` (or `level2`, `level3`) to restrict a DiscoGeM run.
The launcher defaults retain the paper's English-only versus multilingual model pools.

## Independent prediction, evaluation, and analysis

Prediction and evaluation are decoupled. Any model can be assessed if its output
follows [the prediction JSON contract](prediction_contract.md).

```bash
uv run python -m hlv_toolkits.scripts.evaluate \
  --predictions outputs/example/predictions.json \
  --data_dir data/datasets/text_pair/chaosnli/snli/0 --analysis --no-plot
```

This writes distribution-aware metrics, entropy-stratified disagreement metrics,
and instance-level errors beside the output. Do not commit these generated files.
Other shell scripts are development, parallel, or recovery helpers—not the
supported paper-reproduction interface.

## Aggregate seed runs and visualize disagreement

Run `evaluate --analysis` once for every seed and strategy. Then give each
strategy root to `spinda analyze`; repeated seed artifacts are pooled under one
label. The command writes PNG and PDF versions of group-level TVD with
standard-deviation error bars and instance-level TVD violin plots with embedded
IQR boxes, medians, and means.

    uv run spinda analyze --run-dirs outputs/chaosnli/mnli_m/fold_0/roberta-base__soft_to_hard__ce outputs/chaosnli/mnli_m/fold_0/roberta-base__soft__rel --labels Hard_CE ReL --output-dir outputs/disagreement_analysis

For a multi-dimensional dataset, add `--level level1`. The input analyses must
use the same entropy grouping. The default three tertiles reproduce the paper's
reported grouping, while other groupings are supported for new analyses.
