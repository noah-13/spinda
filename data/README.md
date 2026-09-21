# Data Layout

Keep large, cloned, or generated datasets under this directory.

Recommended layout:

Only raw source data and SPINDA-ready datasets live below data/. Runtime cache files belong in .cache/spinda/.

```text
data/
  raw/
    chaosnli/
      chaosNLI_snli.jsonl
    snli/
  datasets/
    snli/
      train.json
      dev.json
      test.json
    text_pair/
      chaosnli/
        0/
          dataset.json
          train.json
          dev.json
          test.json
        ...
        9/
          dataset.json
          train.json
          dev.json
          test.json
    discogem.jsonl
    single_text/
      md_agreement/
        dataset.json
        train.json
        dev.json
        test.json
```

Guidelines:

- Put downloaded or cloned source datasets in `data/raw/`.
- Put derived splits and intermediate artifacts in `data/datasets/`.
- Prepare paper-compatible DiscoGeM datasets with `uv run python -m hlv_toolkits.scripts.prepare_discogem_annotation_labels`. It downloads the 2.0 archive if absent, uses `MV_dist` to retain original annotation vote counts, excludes `norel`, and writes separate English and multilingual level1, level2, and level3 directories under `data/datasets/text_pair/discogem/`. It also writes the named `text_pair_multidimensional_label_distribution` format under `data/datasets/text_pair/discogem/{english,multilingual}/multidimensional/`, with all hierarchy levels in each record.
- The multilingual directories merge `en`, `de`, `fr`, and `cs`; IDs are language-prefixed to remain unique.
- Put caches, temporary files, and local scratch data in `.cache/spinda/`.
- Do not commit large dataset files to git.

## Public text-pair classification format

For reusable hard-label text-pair training, preprocess any dataset into one directory:

```text
my_dataset/
  dataset.json
  train.json
  dev.json
  test.json  # optional unless predicting/evaluating test
```

`dataset.json` fixes the class order and can specify the label mode. Every split file is a top-level JSON array of records. `label_mode: "hard"` uses one integer label per record:

```json
{"format": "text_pair_label_distribution", "label_mode": "hard", "labels": ["not_duplicate", "duplicate"]}
```

```json
{"id": "train-0001", "text_a": "How do I reset my password?", "text_b": "What is the password reset procedure?", "annotation_labels": [1]}
```

`label_mode: "soft"` uses raw annotation indices and the reader derives the normalized distribution from their counts. A one-vote row is allowed in an explicitly soft dataset and becomes a one-hot distribution; without an explicit soft manifest, one annotation is inferred as a hard-label row.

```json
{"format": "text_pair_label_distribution", "label_mode": "soft", "labels": ["not_duplicate", "duplicate"]}
```

```json
{"id": "train-0001", "text_a": "How do I reset my password?", "text_b": "What is the password reset procedure?", "annotation_labels": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```


`label_mode: "soft_to_hard"` requires two or more `annotation_labels` and converts their vote counts with argmax. It rejects hard examples; it is the only permitted soft-to-hard conversion.

Only `annotation_labels` affects label parsing. Every other record field, regardless of name (including `meta`, `label`, or `label_distribution`), is ignored. If `label_mode` is omitted, one annotation means hard and two or more means soft, and the reader emits a warning. `labels` may be omitted, in which case the reader infers the class range from the training annotations, uses `label0`, `label1`, ... and emits a warning. Every annotation index must be in `[0, len(labels) - 1]` when labels are provided.

Training uses `label_training_strategy`. The CLI can override the manifest `label_mode`:

- `hard` and `soft_to_hard` permit only `ce`.
- `soft` permits `ce`, `mse`, `jsd`, or `rel`.
- `rel` expands every value in `annotation_labels` into a repeated hard-label CE instance.

The model always emits one softmax distribution at inference. Incompatible mode/strategy combinations are errors.

```json
{"label_mode": "soft", "label_training_strategy": "jsd"}
```

```bash
--label_mode soft_to_hard
```

For single-text tasks, use the parallel `single_text_label_distribution` format. Rows have `id`, `text`, and `annotation_labels`; its label-mode semantics are the same as the text-pair format. MD-Agreement uses it because each example is one tweet rather than a pair.

```json
{"format": "single_text_label_distribution", "label_mode": "soft", "labels": ["not_offensive", "offensive"]}
```

```json
{"id": "md_agreement:train:1", "text": "Example tweet", "annotation_labels": [0, 0, 0, 1, 1]}
```

Train from direct paths. `format` and `train_path` are required; `dev_path` is optional.

```json
{
  "format": "text_pair_label_distribution",
  "train_path": "data/my_dataset/train.json",
  "dev_path": "data/my_dataset/dev.json",
  "labels": ["not_duplicate", "duplicate"],
  "label_mode": "soft",
  "model": "roberta-base",
  "output_dir": "outputs/my_dataset"
}
```

Save this as `dataset.json` and run. Shared, dataset-agnostic training defaults may be layered before it; this repository provides `configs/training.json`:

```bash
uv run python -m hlv_toolkits.scripts.train --config configs/training.json dataset.json
```

Or use the manifest alone:

```bash
uv run python -m hlv_toolkits.scripts.train --config dataset.json
```

`dataset.json` may contain both the dataset fields above and any training
fields, such as `model`, `num_epochs`, or `learning_rate`. You can also combine
it with shared training presets in one `--config` invocation:

```bash
uv run python -m hlv_toolkits.scripts.train \
  --config data/my_dataset/dataset.json configs/roberta.json configs/long_run.json
```

JSON files are merged left to right, so each later file overrides matching
keys from earlier files. Explicit argparse values override every JSON file:

```bash
uv run python -m hlv_toolkits.scripts.train \
  --config dataset.json configs/roberta.json --num_epochs 10 --no-fp16
```

The equivalent direct CLI requires the same two data fields:

```bash
uv run python -m hlv_toolkits.scripts.train \
  --format text_pair_label_distribution \
  --train_path data/my_dataset/train.json \
  --dev_path data/my_dataset/dev.json \
  --labels not_duplicate duplicate \
  --label_mode soft \
  --model roberta-base --output_dir outputs/my_dataset
```

If `dev_path` is omitted, training emits a warning, does not evaluate or select a best checkpoint, and writes the last epoch model to `final_model`.

Use `hlv_toolkits.scripts.prepare_discogem_annotation_labels` to download and export the current DiscoGeM datasets. It writes independent `discogem/english/level{1,2,3}` and `discogem/multilingual/level{1,2,3}` soft-label datasets, each with `dataset.json`, `train.json`, `dev.json`, and `test.json`.


## MD-Agreement

MD-Agreement is an English single-text offensiveness dataset from LeWiDi 2023. Each tweet has five individual binary judgments. Download and preserve those raw votes, then export the experiment-ready soft-label data with:

```bash
./scripts/md_agreement.sh
```

This invokes the versioned Python download step (`hlv_toolkits.scripts.download_data md_agreement`) followed by `hlv_toolkits.scripts.prepare_md_agreement_annotation_labels`. Raw source files remain in `data/raw/md_agreement/`; the output is `data/datasets/single_text/md_agreement/`.

## MFRC

MFRC (Moral Foundations Reddit Corpus) is exported as `single_text_multilabel_annotation_distribution`, because a Reddit comment can receive several moral-foundation labels from one annotator. Its JSON rows retain those per-annotator sets in `annotation_label_sets`; the reader derives independent per-label vote probabilities (they intentionally do not sum to one).

```bash
bash scripts/mfrc.sh
```

The launcher prepares the data if needed, then runs the shared model/seeds sweep over `soft ce`, `soft mse`, `soft jsd`, `soft rel`, and `soft_to_hard ce`. Set `MODEL_SPECS`, `RUN_SPECS`, `SEEDS_OVERRIDE`, `GPU`, or `FORCE` as for the other launchers. To prepare only, run `uv run python -m hlv_toolkits.scripts.prepare_mfrc_annotation_labels`.

The exporter downloads `USC-MOLA-Lab/MFRC` through `datasets`, groups the source's one-row-per-annotator records into comments, makes a deterministic 80/10/10 train/dev/test split, and writes `data/datasets/single_text/mfrc/`. Metadata retains subreddit, topical bucket, annotator IDs, and confidence.

### MFRC training

MFRC uses eight independent sigmoid outputs. Use `label_mode: "soft_to_hard"` with `ce` for the per-label majority-vote baseline, or `label_mode: "soft"` with `ce`, `mse`, `jsd`, or `rel`. In MFRC, `ce` is binary cross-entropy; `rel` expands one full multi-hot target per annotator. `soft_to_hard` trains those thresholded hard targets but retains original vote probabilities for dev evaluation. All multilabel-classification runs select the checkpoint with the highest `eval_soft_micro_f1` by default. Use `--multilabel_metric_for_best_model` for direct training or `MULTILABEL_METRIC_FOR_BEST_MODEL` in the shared sweep launcher to override this (for example, `multilabel_pojsd`).

### MFRC prediction and evaluation

Run predictions and standard test evaluation for every completed MFRC seed with:

```bash
GPU=0 MAX_PARALLEL=8 bash scripts/mfrc_predict_eval_parallel.sh
```

The launcher shares the one selected physical GPU across prediction jobs, using
available GPU memory to gate parallel launches. Evaluation begins as soon as its
prediction completes and runs without plots. It reuses complete artifacts by
default; set `FORCE=1` to regenerate predictions and evaluations, or
`FORCE_EVAL=1` to rerun evaluation only.

```bash
uv run python -m hlv_toolkits.scripts.train \
  --config data/datasets/single_text/mfrc/dataset.json configs/training.json \
  --head_type multilabel_classification --label_mode soft \
  --label_training_strategy jsd --output_dir outputs/mfrc/jsd
```
