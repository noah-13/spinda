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
MULTIPICO_BASE_URL = os.environ.get("MULTIPICO_BASE_URL", "https://raw.githubusercontent.com/Le-Wi-Di/le-wi-di.github.io/main/LeWiDi_3-2025/MP")
HUMANS_AND_DOMAINS_URL = os.environ.get("HUMANS_AND_DOMAINS_URL", "https://bitbucket.org/robvanderg/humans-and-domains/get/master.zip")

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

def download_multipico(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        output_path = output_dir / f"MP_{split}.json"
        if output_path.exists():
            print(f"MultiPICo {split} already exists in {output_dir}; skipping download")
            continue
        with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as tmp:
            tmp.write(_download(f"{MULTIPICO_BASE_URL}/MP_{split}.json"))
            temporary_path = Path(tmp.name)
        temporary_path.replace(output_path)
        print(f"Downloaded MultiPICo {split} -> {output_path}")

def download_humans_and_domains(output_dir: Path) -> None:
    """Download official TGeGUM JSON splits from Humans-and-Domains."""
    required = [output_dir / f"{split}-sent.json" for split in ("train", "dev", "test")]
    if all(path.exists() for path in required):
        print(f"Humans-and-Domains files already exist in {output_dir}; skipping download")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(_download(HUMANS_AND_DOMAINS_URL))) as archive:
        members = {Path(name).name: name for name in archive.namelist() if "/final_annotations/" in name and name.endswith(".json")}
        missing = [path.name for path in required if path.name not in members]
        if missing:
            raise RuntimeError("Humans-and-Domains archive is missing: " + ", ".join(missing))
        for output_path in required:
            with archive.open(members[output_path.name]) as source, output_path.open("wb") as dest:
                shutil.copyfileobj(source, dest)
    print(f"Downloaded Humans-and-Domains -> {output_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw datasets")
    parser.add_argument("source", choices=["chaosnli", "discogem", "md_agreement", "multipico", "humans_and_domains"])
    parser.add_argument("--chaosnli-dir", type=Path, default=Path("data/external/chaosnli"))
    parser.add_argument("--discogem-path", type=Path, default=Path("data/external/DiscoGeM/DiscoGeM 2.0/DiscoGeM2.0_annotation.tgz"))
    parser.add_argument("--md-agreement-dir", type=Path, default=Path("data/external/md_agreement"))
    parser.add_argument("--multipico-dir", type=Path, default=Path("data/external/multipico"))
    parser.add_argument("--humans-and-domains-dir", type=Path, default=Path("data/external/humans_and_domains"))
    args = parser.parse_args()
    if args.source == "chaosnli":
        download_chaosnli(args.chaosnli_dir)
    elif args.source == "discogem":
        download_discogem(args.discogem_path)
    elif args.source == "md_agreement":
        download_md_agreement(args.md_agreement_dir)
    elif args.source == "multipico":
        download_multipico(args.multipico_dir)
    else:
        download_humans_and_domains(args.humans_and_domains_dir)

if __name__ == "__main__":
    main()
