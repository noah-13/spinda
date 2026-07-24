"""Download raw datasets used by the preprocessing scripts."""

from __future__ import annotations

import argparse
import io
import os
import random
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

CHAOSNLI_URL = os.environ.get("CHAOSNLI_URL", "https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1")
DISCOGEM_URL = os.environ.get("DISCOGEM_URL", "https://raw.githubusercontent.com/merelscholman/DiscoGeM/main/DiscoGeM%202.0/DiscoGeM2.0_annotation.tgz")

def _download(url: str) -> bytes:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:
        return response.read()

def download_chaosnli(output_dir: Path) -> None:
    required = [output_dir / name for name in ("chaosNLI_snli.jsonl", "chaosNLI_mnli_m.jsonl", "chaosNLI_alphanli.jsonl")]
    train_path = output_dir / "chaosNLI_snli_train.jsonl"
    dev_path = output_dir / "chaosNLI_snli_dev.jsonl"
    if not all(path.exists() for path in required):
        output_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(_download(CHAOSNLI_URL))) as archive:
            names = {Path(name).name: name for name in archive.namelist()}
            missing = [path.name for path in required if path.name not in names]
            if missing:
                raise RuntimeError(f"ChaosNLI archive is missing: {', '.join(missing)}")
            for target in required:
                with archive.open(names[target.name]) as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
        print(f"Downloaded ChaosNLI -> {output_dir}")
    else:
        print(f"ChaosNLI archive files already exist in {output_dir}; skipping download")
    if not train_path.exists() or not dev_path.exists():
        lines = [line for line in required[0].read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) <= 1400:
            raise RuntimeError(f"ChaosNLI SNLI file has too few samples to split: {len(lines)}")
        indices = set(random.Random(42).sample(range(len(lines)), 1400))
        train_path.write_text("\n".join(line for i, line in enumerate(lines) if i in indices) + "\n", encoding="utf-8")
        dev_path.write_text("\n".join(line for i, line in enumerate(lines) if i not in indices) + "\n", encoding="utf-8")
        print(f"Created ChaosNLI train/dev split -> {train_path}, {dev_path}")

def download_discogem(output_path: Path) -> None:
    if output_path.exists():
        print(f"DiscoGeM already exists at {output_path}; skipping download")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False) as tmp:
        tmp.write(_download(DISCOGEM_URL))
        temporary_path = Path(tmp.name)
    temporary_path.replace(output_path)
    print(f"Downloaded DiscoGeM -> {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw datasets")
    parser.add_argument("source", choices=["chaosnli", "discogem"])
    parser.add_argument("--chaosnli-dir", type=Path, default=Path("data/external/chaosnli"))
    parser.add_argument("--discogem-path", type=Path, default=Path("data/external/DiscoGeM/DiscoGeM 2.0/DiscoGeM2.0_annotation.tgz"))
    args = parser.parse_args()
    (download_chaosnli(args.chaosnli_dir) if args.source == "chaosnli" else download_discogem(args.discogem_path))

if __name__ == "__main__":
    main()
