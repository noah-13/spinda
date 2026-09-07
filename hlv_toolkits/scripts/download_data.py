"""Download raw datasets used by the preprocessing scripts."""

from __future__ import annotations

import argparse
import io
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

CHAOSNLI_URL = os.environ.get("CHAOSNLI_URL", "https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1")
DISCOGEM_URL = os.environ.get("DISCOGEM_URL", "https://raw.githubusercontent.com/merelscholman/DiscoGeM/main/DiscoGeM%202.0/DiscoGeM2.0_annotation.tgz")
MD_AGREEMENT_BASE_URL = os.environ.get("MD_AGREEMENT_BASE_URL", "https://raw.githubusercontent.com/Le-Wi-Di/le-wi-di.github.io/main/LeWiDi_2-2023/MD-Agreement_dataset")

def _download(url: str) -> bytes:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:
        return response.read()

def download_chaosnli(output_dir: Path) -> None:
    required = [output_dir / name for name in ("chaosNLI_snli.jsonl", "chaosNLI_mnli_m.jsonl")]
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

def download_md_agreement(output_dir: Path) -> None:
    """Download the official LeWiDi 2023 MD-Agreement split files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        output_path = output_dir / f"MD-Agreement_{split}.json"
        if output_path.exists():
            print(f"MD-Agreement {split} already exists at {output_path}; skipping download")
            continue
        with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as tmp:
            tmp.write(_download(f"{MD_AGREEMENT_BASE_URL}/MD-Agreement_{split}.json"))
            temporary_path = Path(tmp.name)
        temporary_path.replace(output_path)
        print(f"Downloaded MD-Agreement {split} -> {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw datasets")
    parser.add_argument("source", choices=["chaosnli", "discogem", "md_agreement"])
    parser.add_argument("--chaosnli-dir", type=Path, default=Path("data/external/chaosnli"))
    parser.add_argument("--discogem-path", type=Path, default=Path("data/external/DiscoGeM/DiscoGeM 2.0/DiscoGeM2.0_annotation.tgz"))
    parser.add_argument("--md-agreement-dir", type=Path, default=Path("data/external/md_agreement"))
    args = parser.parse_args()
    if args.source == "chaosnli":
        download_chaosnli(args.chaosnli_dir)
    elif args.source == "discogem":
        download_discogem(args.discogem_path)
    else:
        download_md_agreement(args.md_agreement_dir)

if __name__ == "__main__":
    main()
