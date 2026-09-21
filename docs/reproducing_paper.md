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
  --data_dir data/processed/text_pair/chaosnli/snli/0 --analysis --no-plot
```

This writes distribution-aware metrics, entropy-stratified disagreement metrics,
and instance-level errors beside the output. Do not commit these generated files.
Other shell scripts are development, parallel, or recovery helpers—not the
supported paper-reproduction interface.
