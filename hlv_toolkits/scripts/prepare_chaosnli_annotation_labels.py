#!/usr/bin/env python3
"""Convert official ChaosNLI NLI rows to the annotation_labels text-pair contract.

The split protocol reimplements the train/test procedure in
https://github.com/kmkurn/train-eval-hlv/blob/9b4529093bb0492372cbf701860454de64677fc5/create_kfold_splits.py
and the dev split in
https://github.com/kmkurn/train-eval-hlv/blob/9b4529093bb0492372cbf701860454de64677fc5/create_random_split.py.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from hlv_toolkits.scripts.download_data import download_chaosnli
from hlv_toolkits.data.json_io import write_records


LABELS = ["entailment", "neutral", "contradiction"]
SUBSET_FILENAMES = {
    "snli": "chaosNLI_snli.json",
    "mnli_m": "chaosNLI_mnli_m.json",
}
NUM_FOLDS = 10
DEV_PORTION = 0.1
SPLIT_SEED = 42


def _counts(row: dict[str, Any], source: Path, line_number: int) -> list[int]:
    values = row.get("label_count")
    if isinstance(values, list) and len(values) == len(LABELS):
        counts = values
    else:
        counter = row.get("label_counter")
        if not isinstance(counter, dict):
            raise ValueError(f"{source}:{line_number} has neither a 3-way label_count nor label_counter.")
        counts = [counter.get("e", 0), counter.get("n", 0), counter.get("c", 0)]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError(f"{source}:{line_number} has invalid annotation counts.")
    if sum(counts) < 2:
        raise ValueError(f"{source}:{line_number} needs at least two annotations for ChaosNLI soft data.")
    return list(counts)


def _convert_row(row: dict[str, Any], source: Path, line_number: int) -> dict[str, Any]:
    example = row.get("example")
    if not isinstance(example, dict):
        raise ValueError(f"{source}:{line_number} is missing its example object.")
    premise = example.get("premise")
    hypothesis = example.get("hypothesis")
    sample_id = row.get("uid") or example.get("uid")
    if not isinstance(sample_id, str) or not sample_id or not isinstance(premise, str) or not isinstance(hypothesis, str):
        raise ValueError(f"{source}:{line_number} has missing id, premise, or hypothesis.")
    counts = _counts(row, source, line_number)
    return {
        "id": sample_id,
        "text_a": premise,
        "text_b": hypothesis,
        "annotation_labels": [label for label, count in enumerate(counts) for _ in range(count)],
    }


def _load_rows(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.is_file():
        raise FileNotFoundError(f"ChaosNLI split not found: {input_path}")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {input_path}:{line_number}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{input_path}:{line_number} must be an object.")
        converted = _convert_row(raw, input_path, line_number)
        if converted["id"] in seen_ids:
            raise ValueError(f"Duplicate ChaosNLI id {converted['id']!r} in {input_path}.")
        seen_ids.add(converted["id"])
        rows.append(converted)
    return rows


def _write_split(rows: list[dict[str, Any]], output_path: Path) -> None:
    write_records(output_path, rows)


def _split_kfold_train_dev_test(
    rows: list[dict[str, Any]], folds: int, dev_portion: float, seed: int
) -> list[dict[str, list[dict[str, Any]]]]:
    """Create train/dev/test folds without leaking an identical text pair.

    This reimplements the linked train-eval-hlv scripts: make ten train/test
    folds, then hold out 10% of each fold's training portion for dev.
    """
    if folds < 2:
        raise ValueError(f"--folds must be at least 2, got {folds}.")
    if folds > len(rows):
        raise ValueError(f"--folds must not exceed the number of rows ({len(rows)}), got {folds}.")
    if not 0 < dev_portion < 1:
        raise ValueError(f"--dev-portion must be between 0 and 1, got {dev_portion}.")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["text_a"], row["text_b"]), []).append(row)
    group_rows = list(groups.values())
    if folds > len(group_rows):
        raise ValueError(
            f"--folds must not exceed the number of unique text pairs ({len(group_rows)}), got {folds}."
        )

    random.Random(seed).shuffle(group_rows)
    base_size, remainder = divmod(len(group_rows), folds)
    test_group_counts = [base_size + (fold < remainder) for fold in range(folds)]
    result: list[dict[str, list[dict[str, Any]]]] = []
    offset = 0
    for fold, test_group_count in enumerate(test_group_counts):
        test_groups = group_rows[offset : offset + test_group_count]
        train_groups = group_rows[:offset] + group_rows[offset + test_group_count :]
        offset += test_group_count

        # The original project shuffles before its random dev split.  Use a
        # fold-specific seed here so the same command is reproducible.
        random.Random(seed + fold + 1).shuffle(train_groups)
        dev_group_count = round(dev_portion * len(train_groups))
        if not 0 < dev_group_count < len(train_groups):
            raise ValueError("--dev-portion leaves an empty train or dev split.")
        result.append(
            {
                "train": [row for group in train_groups[dev_group_count:] for row in group],
                "dev": [row for group in train_groups[:dev_group_count] for row in group],
                "test": [row for group in test_groups for row in group],
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ChaosNLI SNLI to annotation_labels text-pair JSON.")
    parser.add_argument("--input_dir", type=Path, default=Path("data/raw/chaosnli"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/datasets/text_pair/chaosnli"))
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=list(SUBSET_FILENAMES),
        default=list(SUBSET_FILENAMES),
        help="ChaosNLI subsets to prepare. alphaNLI is excluded because it is not standard NLI.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=range(NUM_FOLDS),
        default=None,
        help="Only write this fold after constructing the deterministic 10-fold split.",
    )
    args = parser.parse_args()

    download_chaosnli(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for subset in args.subsets:
        input_path = args.input_dir / SUBSET_FILENAMES[subset]
        rows = _load_rows(input_path)
        for fold, splits in enumerate(_split_kfold_train_dev_test(rows, NUM_FOLDS, DEV_PORTION, SPLIT_SEED)):
            if args.fold is not None and fold != args.fold:
                continue
            fold_dir = args.output_dir / subset / str(fold)
            fold_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "format": "text_pair_label_distribution",
                "train_path": str(fold_dir / "train.json"),
                "dev_path": str(fold_dir / "dev.json"),
                "labels": LABELS,
            }
            (fold_dir / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            for split, split_rows in splits.items():
                output_path = fold_dir / f"{split}.json"
                _write_split(split_rows, output_path)
                print(f"Wrote ChaosNLI {subset} fold {fold} {split} ({len(split_rows)} rows) -> {output_path}")


if __name__ == "__main__":
    main()
