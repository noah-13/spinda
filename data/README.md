# Data Layout

Keep large, cloned, or generated datasets under this directory.

Recommended layout:

```text
data/
  external/
    chaosnli/
      chaosNLI_snli.jsonl
      chaosNLI_snli_train.jsonl
      chaosNLI_snli_dev.jsonl
    snli/
  processed/
    snli/
      train.jsonl
      dev.jsonl
      test.jsonl
    chaosnli/
      train.jsonl
      dev.jsonl
      test.jsonl
    discogem.jsonl
  cache/
```

Guidelines:

- Put downloaded or cloned source datasets in `data/external/`.
- Put derived splits and intermediate artifacts in `data/processed/`.
- For DiscoGeM-specific label tables and the processed-file format, see `data/discogem/README.md`.
- For DiscoGeM, run `bash scripts/preprocess.sh discogem` to build the canonical merged file at `data/processed/discogem.jsonl`.
- Put caches, temporary files, and local scratch data in `data/cache/`.
- Do not commit large dataset files to git.
