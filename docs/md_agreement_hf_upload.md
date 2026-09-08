# Share one MD-Agreement part through Hugging Face

The sender uploads the completed part to a public Hugging Face Dataset repo.
The receiver downloads it from the repo name or link.

## Sender: upload

Run from the repository root after the assigned part is complete:

```bash
uv run hf auth login

uv run hf repo create <SENDER_USERNAME>/md-agreement-part-01 \
  --repo-type dataset \
  --exist-ok

HF_XET_HIGH_PERFORMANCE=1 uv run hf upload \
  <SENDER_USERNAME>/md-agreement-part-01 \
  outputs/md_agreement/default \
  --repo-type dataset \
  --include '**/final_model/**' \
  --commit-message 'Upload MD-Agreement part 01'
```

Send this repository name or URL to the receiver:

```text
<SENDER_USERNAME>/md-agreement-part-01
```

Do not upload `checkpoint-*` directories.

## Receiver: download

```bash
uv run hf download <SENDER_USERNAME>/md-agreement-part-01 \
  --repo-type dataset \
  --local-dir outputs/md_agreement/default
```

No Hugging Face login is needed to download a public repo.
