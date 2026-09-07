# Run MD-Agreement on another server

## Environment

Requirements:

- NVIDIA GPU with a working driver
- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- Internet access for the first run (dataset and model downloads)

```bash
# Install uv if it is not already available
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env" 2>/dev/null || true

git clone <REPOSITORY_URL>
cd nli-tookits
uv sync

nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available())"
```

The last command should print `True`.

## Run

Run the assigned script from the repository root:

```bash
GPU=0 bash scripts/md_agreement_parts/<assigned-part>.sh
```

For example:

```bash
GPU=0 bash scripts/md_agreement_parts/01_heavy_xlmr_part1.sh
```

The script downloads and prepares MD-Agreement automatically on its first run.
