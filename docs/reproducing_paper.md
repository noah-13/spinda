# Reproducing the SPInDa experiments

This page defines the supported public experiment interface for the SPInDa paper.
Each launcher prepares data as needed, trains the documented sweep, and writes
runs below `outputs/`, which is ignored by Git. Run commands from the repository root.

## Install and smoke test

```bash
uv sync
MODEL_SPECS=roberta-base RUN_SPECS='soft ce' SEEDS_OVERRIDE=42 GPU=0 \
  bash scripts/chaosnli.sh
```

Every canonical launcher uses the shared
[`configs/training.json`](../configs/training.json) by default, including its
seeds 42, 43, and 44. They also use one consistent model pool per language
scope: English uses the four English models (DeBERTa-v3-large, RoBERTa-base,
BERT-base-uncased, and Twhin-BERT-base) plus the three multilingual models
(XLM-RoBERTa-base, mBERT, and InfoXLM-base); multilingual uses those three
multilingual models only.
Launchers accept `GPU`, `FORCE=1`, `SEEDS_OVERRIDE="42 43 44"`,
`TRAINING_CONFIG`, and `MODEL_SPECS` for explicit ablations. Upstream dataset
download may require accepting its terms; raw and processed datasets are
Git-ignored.

## Canonical launchers

| Paper setting | Launcher | Dataset source |
| --- | --- | --- |
| ChaosNLI (SNLI and MNLI-M) | `bash scripts/chaosnli.sh` | [Nie, Zhou, and Bansal (2020)](https://aclanthology.org/2020.emnlp-main.734/) |
| MD-Agreement | `bash scripts/md_agreement.sh` | [Leonardelli et al. (2021)](https://aclanthology.org/2021.emnlp-main.822/) |
| MultiPICo (English / multilingual) | `bash scripts/multipico.sh` | [Casola et al. (2024)](https://aclanthology.org/2024.acl-long.849/) |
| DiscoGeM 2.0 (English / multilingual / separate / joint) | `bash scripts/discogem.sh` | [Yung et al. (2024)](https://aclanthology.org/2024.lrec-main.443/) |
| TGeGUM / Humans-and-Domains | `bash scripts/humans_and_domains.sh` | [Barrett et al. (2024)](https://aclanthology.org/2024.lrec-main.245/) |
| MFRC multi-label | `bash scripts/mfrc.sh` | [Trager et al. (2026)](https://aclanthology.org/2026.lrec-1.507/) |

### MFRC version and annotation counts

The MFRC launcher uses the public Hugging Face
[`USC-MOLA-Lab/MFRC`](https://huggingface.co/datasets/USC-MOLA-Lab/MFRC)
`train_dedup` release, then groups its one-row-per-annotation records by
`(text, subreddit, bucket)`. This processed source contains 53,827 annotation
rows and yields **17,886** grouped comments. Its observed annotation-count
distribution is 135 comments with 2 rows, 17,451 with 3, 296 with 4, and 4
with 5. Thus, `N = 17,886` and `Ann./item = 2–5` refer specifically to this
processed Hugging Face version, not to the original MFRC-paper corpus.

The original MFRC paper reports 16,123 English Reddit comments, each annotated
by at least three annotators. Treat those as the source-paper statistics; do
not substitute the processed-release counts when describing the original corpus.

`scripts/multipico.sh` defaults to both the English and multilingual settings.
Set `VARIANTS=english` or `VARIANTS=multilingual` to run one setting only.

`scripts/discogem.sh` defaults to both language settings, all three separate
levels, and joint multilevel training. Set `VARIANTS=english` (or
`multilingual`) to limit language scope; set `MODES=separate` or `MODES=joint`
to choose the implementation. `LEVEL=level1` (or `level2`, `level3`) further
limits a separate-only run, for example:

```bash
VARIANTS=english MODES=separate LEVEL=level1 bash scripts/discogem.sh
```

The defaults cover every training setting reported in the paper: ChaosNLI-S/M,
MD-Agreement, MultiPICo English/multilingual, DiscoGeM English/multilingual
separate and joint, TGeGUM separate and joint, and MFRC. They prepare data and
train the paper model/strategy/seed matrix. They do not automatically recreate
the paper's aggregated tables and figures; run prediction, evaluation, and
analysis over the completed checkpoints to produce those artifacts.

## Independent prediction, evaluation, and analysis

Prediction and evaluation are decoupled. Any model can be assessed if its output
follows [the prediction JSON contract](prediction_contract.md).

```bash
uv run python -m spinda.scripts.evaluate \
  --predictions outputs/example/predictions.json \
  --input_file data/datasets/text_pair/chaosnli/snli/0/test.json \
  --human_labels data/datasets/text_pair/chaosnli/snli/0/test.json --analysis
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
