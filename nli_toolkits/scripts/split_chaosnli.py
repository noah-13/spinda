#!/usr/bin/env python3
"""Randomly split a JSONL file into train/dev sets."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample N lines as train and use the rest as dev."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("chaosNLI_v1.0/chaosNLI_snli.jsonl"),
        help="Input JSONL path",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=1400,
        help="Number of samples to place in train set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("chaosNLI_v1.0/chaosNLI_snli_train.jsonl"),
        help="Train output JSONL path",
    )
    parser.add_argument(
        "--dev-output",
        type=Path,
        default=Path("chaosNLI_v1.0/chaosNLI_snli_dev.jsonl"),
        help="Dev output JSONL path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    lines = [line for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(lines)

    if args.train_size <= 0:
        raise ValueError("--train-size must be > 0")
    if args.train_size >= total:
        raise ValueError(f"--train-size ({args.train_size}) must be smaller than total samples ({total})")

    rng = random.Random(args.seed)
    train_indices = set(rng.sample(range(total), args.train_size))

    train_lines = [line for i, line in enumerate(lines) if i in train_indices]
    dev_lines = [line for i, line in enumerate(lines) if i not in train_indices]

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.dev_output.parent.mkdir(parents=True, exist_ok=True)

    args.train_output.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    args.dev_output.write_text("\n".join(dev_lines) + "\n", encoding="utf-8")

    print(f"Total: {total}")
    print(f"Train: {len(train_lines)} -> {args.train_output}")
    print(f"Dev: {len(dev_lines)} -> {args.dev_output}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
