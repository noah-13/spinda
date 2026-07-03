#!/usr/bin/env python3
"""Normalize raw dataset files into canonical JSONL samples."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from hlv_toolkits.data import ChaosNLIReader, DiscoGeMReader, SNLIReader
from hlv_toolkits.data.schemas import DiscoGeMMultiLevelSample
from hlv_toolkits.data.serialization import dump_samples


def _write_split(samples, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dump_samples(samples) + "\n", encoding="utf-8")


def _process_snli(output_dir: Path, splits: Iterable[str]) -> None:
    reader = SNLIReader()
    for split in splits:
        samples = reader.load_split(split)  # type: ignore[arg-type]
        _write_split(samples, output_dir / f"{split}.jsonl")
        print(f"Wrote {len(samples)} SNLI samples -> {output_dir / f'{split}.jsonl'}")


def _process_chaosnli(args: argparse.Namespace, output_dir: Path) -> None:
    split_paths = {
        "train": args.chaosnli_train_path,
        "dev": args.chaosnli_dev_path,
        "test": args.chaosnli_test_path,
    }
    for split, input_path in split_paths.items():
        if not input_path:
            continue
        reader = ChaosNLIReader(data_path=input_path)
        samples = reader.load_split(split)  # type: ignore[arg-type]
        _write_split(samples, output_dir / f"{split}.jsonl")
        print(f"Wrote {len(samples)} ChaosNLI samples -> {output_dir / f'{split}.jsonl'}")


def _merge_discogem_samples(soft_samples, hard_samples) -> List[DiscoGeMMultiLevelSample]:
    hard_by_id = {sample.id: sample for sample in hard_samples}
    merged: List[DiscoGeMMultiLevelSample] = []
    for sample in soft_samples:
        hard_sample = hard_by_id.get(sample.id)
        if hard_sample is None:
            continue
        merged.append(
            DiscoGeMMultiLevelSample(
                id=sample.id,
                task=sample.task,
                split=sample.split,
                source=sample.source,
                premise=sample.premise,
                hypothesis=sample.hypothesis,
                hard_labels=dict(getattr(hard_sample, "hard_labels", {})),
                human_dists=dict(getattr(sample, "human_dists", {})),
            )
        )
    return merged


def _process_discogem(args: argparse.Namespace, output_dir: Path) -> None:
    soft_reader = DiscoGeMReader(
        data_path=args.discogem_path,
        version=args.discogem_version,
        label_mode="soft",
        label_level="all",
        language=args.discogem_language,
    )
    hard_reader = DiscoGeMReader(
        data_path=args.discogem_path,
        version=args.discogem_version,
        label_mode="hard",
        label_level="all",
        language=args.discogem_language,
    )

    all_samples: List[DiscoGeMMultiLevelSample] = []
    for split in args.splits:
        soft_samples = soft_reader.load_split(split)  # type: ignore[arg-type]
        hard_samples = hard_reader.load_split(split)  # type: ignore[arg-type]
        merged = _merge_discogem_samples(soft_samples, hard_samples)
        all_samples.extend(merged)
        print(f"Wrote {len(merged)} DiscoGeM {split} samples")

    output_file = output_dir / "discogem.jsonl"
    _write_split(all_samples, output_file)
    print(f"Wrote {len(all_samples)} DiscoGeM samples -> {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess datasets into canonical JSONL samples")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["snli", "chaosnli", "discogem"],
        help="Raw data source to normalize",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where canonical JSONL files will be written",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        choices=["train", "dev", "test"],
        help="Splits to materialize (used for snli and discogem)",
    )
    parser.add_argument(
        "--chaosnli_train_path",
        type=str,
        default=None,
        help="Input ChaosNLI train JSONL path",
    )
    parser.add_argument(
        "--chaosnli_dev_path",
        type=str,
        default=None,
        help="Input ChaosNLI dev JSONL path",
    )
    parser.add_argument(
        "--chaosnli_test_path",
        type=str,
        default=None,
        help="Input ChaosNLI test JSONL path",
    )
    parser.add_argument(
        "--discogem_path",
        type=str,
        default="",
        help="Path to the DiscoGeM 2.0 annotation archive",
    )
    parser.add_argument(
        "--discogem_version",
        type=str,
        default="2.0",
        choices=["auto", "2.0"],
        help="DiscoGeM schema version",
    )
    parser.add_argument(
        "--discogem_language",
        type=str,
        default="en",
        choices=["en", "de", "fr", "cs"],
        help="DiscoGeM language slice to use for version 2.0",
    )

    args = parser.parse_args()
    if args.source == "snli":
        _process_snli(args.output_dir / args.source, args.splits)
    elif args.source == "chaosnli":
        if not any([args.chaosnli_train_path, args.chaosnli_dev_path, args.chaosnli_test_path]):
            raise ValueError(
                "At least one of --chaosnli_train_path/--chaosnli_dev_path/--chaosnli_test_path is required when --source chaosnli"
            )
        _process_chaosnli(args, args.output_dir / args.source)
    elif args.source == "discogem":
        _process_discogem(args, args.output_dir)
    else:
        raise ValueError(f"Unknown source: {args.source}")


if __name__ == "__main__":
    main()
