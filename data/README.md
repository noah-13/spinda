# Data Layout

Keep large, cloned, or generated datasets under this directory.

Recommended layout:

```text
data/
  external/
    chaosnli/
      chaosNLI_snli.jsonl
    snli/
  processed/
    snli/
      train.jsonl
      dev.jsonl
      test.jsonl
    text_pair/
      chaosnli/
        0/
          dataset.json
          train.jsonl
          dev.jsonl
          test.jsonl
        ...
        9/
          dataset.json
          train.jsonl
          dev.jsonl
          test.jsonl
    discogem.jsonl
    single_text/
      md_agreement/
        dataset.json
        train.jsonl
        dev.jsonl
        test.jsonl
  cache/
```

Guidelines:

- Put downloaded or cloned source datasets in `data/external/`.
- Put derived splits and intermediate artifacts in `data/processed/`.
- Prepare paper-compatible DiscoGeM datasets with `uv run python -m hlv_toolkits.scripts.prepare_discogem_annotation_labels`. It downloads the 2.0 archive if absent, uses `MV_dist` to retain original annotation vote counts, excludes `norel`, and writes separate English and multilingual level1, level2, and level3 directories under `data/processed/text_pair/discogem/`. It also writes the named `text_pair_multilevel_label_distribution` format under `data/processed/text_pair/discogem/{english,multilingual}/multilevel/`, with all hierarchy levels in each record.
- The multilingual directories merge `en`, `de`, `fr`, and `cs`; IDs are language-prefixed to remain unique.
- Put caches, temporary files, and local scratch data in `data/cache/`.
- Do not commit large dataset files to git.

## Public text-pair classification format

For reusable hard-label text-pair training, preprocess any dataset into one directory:

```text
my_dataset/
  dataset.json
  train.jsonl
  dev.jsonl
  test.jsonl  # optional unless predicting/evaluating test
```

`dataset.json` fixes the class order and can specify the label mode. `label_mode: "hard"` uses one integer label per row:

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

Only `annotation_labels` affects label parsing. Every other JSONL field, regardless of name (including `meta`, `label`, or `label_distribution`), is ignored. If `label_mode` is omitted, one annotation means hard and two or more means soft, and the reader emits a warning. `labels` may be omitted, in which case the reader infers the class range from the training annotations, uses `label0`, `label1`, ... and emits a warning. Every annotation index must be in `[0, len(labels) - 1]` when labels are provided.

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
  "train_path": "data/my_dataset/train.jsonl",
  "dev_path": "data/my_dataset/dev.jsonl",
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
  --train_path data/my_dataset/train.jsonl \
  --dev_path data/my_dataset/dev.jsonl \
  --labels not_duplicate duplicate \
  --label_mode soft \
  --model roberta-base --output_dir outputs/my_dataset
```

If `dev_path` is omitted, training emits a warning, does not evaluate or select a best checkpoint, and writes the last epoch model to `final_model`.

Use `hlv_toolkits.scripts.prepare_discogem_annotation_labels` to download and export the current DiscoGeM datasets. It writes independent `discogem/english/level{1,2,3}` and `discogem/multilingual/level{1,2,3}` soft-label datasets, each with `dataset.json`, `train.jsonl`, `dev.jsonl`, and `test.jsonl`.


## MD-Agreement

MD-Agreement is an English single-text offensiveness dataset from LeWiDi 2023. Each tweet has five individual binary judgments. Download and preserve those raw votes, then export the experiment-ready soft-label data with:

```bash
./scripts/md_agreement.sh
```

This invokes the versioned Python download step (`hlv_toolkits.scripts.download_data md_agreement`) followed by `hlv_toolkits.scripts.prepare_md_agreement_annotation_labels`. Raw source files remain in `data/external/md_agreement/`; the output is `data/processed/single_text/md_agreement/`.
