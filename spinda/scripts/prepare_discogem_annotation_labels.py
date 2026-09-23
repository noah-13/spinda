#!/usr/bin/env python3
"""Download and export paper-compatible DiscoGeM 2.0 datasets.

This follows Costa and Kosseim (SIGDIAL 2025): official train/dev/test splits,
original annotation votes, no ``norel`` class, and 4/17/28 labels at levels 1/2/3.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from collections import Counter, defaultdict

from spinda.data.tie_breaking import tied_argmax
from spinda.data.json_io import write_records
from pathlib import Path
from typing import Any

from spinda.scripts.download_data import download_discogem

LANGUAGES = ("en", "de", "fr", "cs")
LABELS = {
    "level1": ("temporal", "contingency", "comparison", "expansion"),
    "level2": (
        "synchronous", "asynchronous", "cause", "condition", "neg-condition", "purpose",
        "concession", "contrast", "similarity", "conjunction", "disjunction", "equivalence",
        "exception", "instantiation", "level-of-detail", "manner", "substitution",
    ),
    "level3": (
        "synchronous", "precedence", "succession", "reason", "result", "arg1-as-cond",
        "arg2-as-cond", "arg1-as-negcond", "arg2-as-negcond", "arg1-as-goal", "arg2-as-goal",
        "arg1-as-denier", "arg2-as-denier", "contrast", "similarity", "conjunction", "disjunction",
        "equivalence", "arg1-as-excpt", "arg2-as-excpt", "arg1-as-instance", "arg2-as-instance",
        "arg1-as-detail", "arg2-as-detail", "arg1-as-manner", "arg2-as-manner", "arg1-as-subst",
        "arg2-as-subst",
    ),
}
LEVEL3_TO_LEVEL2 = {
    "synchronous": "synchronous", "precedence": "asynchronous", "succession": "asynchronous",
    "reason": "cause", "result": "cause", "arg1-as-cond": "condition", "arg2-as-cond": "condition",
    "arg1-as-negcond": "neg-condition", "arg2-as-negcond": "neg-condition",
    "arg1-as-goal": "purpose", "arg2-as-goal": "purpose",
    "arg1-as-denier": "concession", "arg2-as-denier": "concession", "contrast": "contrast",
    "similarity": "similarity", "conjunction": "conjunction", "disjunction": "disjunction",
    "equivalence": "equivalence", "arg1-as-excpt": "exception", "arg2-as-excpt": "exception",
    "arg1-as-instance": "instantiation", "arg2-as-instance": "instantiation",
    "arg1-as-detail": "level-of-detail", "arg2-as-detail": "level-of-detail",
    "arg1-as-manner": "manner", "arg2-as-manner": "manner",
    "arg1-as-subst": "substitution", "arg2-as-subst": "substitution",
}
LEVEL2_TO_LEVEL1 = {
    "synchronous": "temporal", "asynchronous": "temporal", "cause": "contingency",
    "condition": "contingency", "neg-condition": "contingency", "purpose": "contingency",
    "concession": "comparison", "contrast": "comparison", "similarity": "comparison",
    "conjunction": "expansion", "disjunction": "expansion", "equivalence": "expansion",
    "exception": "expansion", "instantiation": "expansion", "level-of-detail": "expansion",
    "manner": "expansion", "substitution": "expansion",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    with tarfile.open(path, "r:gz") as archive:
        member = archive.extractfile("DiscoGeM2.0_annotation/DiscoGeM2.0_items.csv")
        if member is None:
            raise FileNotFoundError("DiscoGeM2.0_items.csv is missing from the archive.")
        rows = list(csv.DictReader(io.StringIO(member.read().decode("utf-8-sig")), delimiter="\t"))
        annotations = archive.extractfile("DiscoGeM2.0_annotation/DiscoGeM2.0_labels.csv")
        if annotations is None:
            raise FileNotFoundError("DiscoGeM2.0_labels.csv is missing from the archive.")
        by_task = defaultdict(list)
        for vote in csv.DictReader(io.StringIO(annotations.read().decode("utf-8-sig")), delimiter="\t"):
            if vote["label"] != "norel":
                if vote["label"] not in LABELS["level3"]:
                    raise ValueError(f"Unknown DiscoGeM label: {vote['label']}")
                by_task[vote["task"]].append(vote["label"])
        for row in rows:
            for language in LANGUAGES:
                row[f"annotation_votes_{language}"] = by_task.get(f"{row['itemid']}.{language}", [])
        return rows


def _votes(annotation_votes: list[str], level: str) -> list[int]:
    """Map every retained annotation once, preserving count and alignment."""
    index = {label: position for position, label in enumerate(LABELS[level])}
    mapped = []
    for label in annotation_votes:
        if level in {"level1", "level2"}:
            label = LEVEL3_TO_LEVEL2[label]
        if level == "level1":
            label = LEVEL2_TO_LEVEL1[label]
        mapped.append(index[label])
    return mapped


def _write_variant(rows: list[dict[str, Any]], output_root: Path, variant: str, languages: tuple[str, ...]) -> None:
    for level, labels in LABELS.items():
        output_dir = output_root / "discogem" / variant / level
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "original_votes_v1").unlink(missing_ok=True)
        manifest = {
            "format": "text_pair_label_distribution",
            "label_mode": "soft",
            "labels": labels,
            "train_path": str(output_dir / "train.json"),
            "dev_path": str(output_dir / "dev.json"),
        }
        (output_dir / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        split_counts: Counter[str] = Counter()
        records = {split: [] for split in ("train", "dev", "test")}
        try:
            for language in languages:
                for row in rows:
                    split = row.get("split", "")
                    text_a = (row.get(f"arg1_context_{language}") or "").strip()
                    text_b = (row.get(f"arg2_context_{language}") or "").strip()
                    distribution = (row.get(f"MV_dist_{language}") or "").strip()
                    if split not in records or not text_a or not text_b or not distribution:
                        continue
                    votes = _votes(row[f"annotation_votes_{language}"], level)
                    if not votes:
                        continue
                    sample_id = (row.get("itemid") or "").strip()
                    if not sample_id:
                        continue
                    record = {"id": sample_id if variant == "english" else f"{language}:{sample_id}", "text_a": text_a, "text_b": text_b, "annotation_labels": votes, "meta": {"label_identity": {"id": f"{language}:{sample_id}", "namespace": f"discogem:{level}"}}}
                    records[split].append(record)
                    split_counts[split] += 1
        finally:
            for split, split_records in records.items():
                write_records(output_dir / f"{split}.json", split_records)
        (output_dir / "original_votes_v1").write_text("complete\n")
        print(f"{variant}/{level}: train={split_counts['train']} dev={split_counts['dev']} test={split_counts['test']}")


def _write_multilevel_variant(rows: list[dict[str, Any]], output_root: Path, variant: str, languages: tuple[str, ...]) -> None:
    """Write one paper-compatible record per pair with all three label levels."""
    output_dir = output_root / "discogem" / variant / "multilevel"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "original_votes_v1").unlink(missing_ok=True)
    manifest = {
        "format": "text_pair_multidimensional_label_distribution",
        "label_mode": "soft",
        "level_labels": LABELS,
        "train_path": str(output_dir / "train.json"),
        "dev_path": str(output_dir / "dev.json"),
    }
    (output_dir / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    split_counts: Counter[str] = Counter()
    records = {split: [] for split in ("train", "dev", "test")}
    try:
        for language in languages:
            for row in rows:
                split = row.get("split", "")
                text_a = (row.get(f"arg1_context_{language}") or "").strip()
                text_b = (row.get(f"arg2_context_{language}") or "").strip()
                distribution = (row.get(f"MV_dist_{language}") or "").strip()
                sample_id = (row.get("itemid") or "").strip()
                if split not in records or not text_a or not text_b or not distribution or not sample_id:
                    continue
                votes = {level: _votes(row[f"annotation_votes_{language}"], level) for level in LABELS}
                if any(not level_votes for level_votes in votes.values()):
                    continue
                human_dists = {
                    level: [level_votes.count(index) / len(level_votes) for index in range(len(LABELS[level]))]
                    for level, level_votes in votes.items()
                }
                hard_labels = {level: tied_argmax(dist, sample_id, f"discogem:{level}") for level, dist in human_dists.items()}
                record = {
                    "_schema": "MultilevelSample",
                    "id": sample_id if variant == "english" else f"{language}:{sample_id}",
                    "task": "discogem",
                    "split": split,
                    "source": "discogem_paper",
                    "text_a": text_a,
                    "text_b": text_b,
                    "hard_labels": hard_labels,
                    "annotation_labels": votes,
                    "meta": {"language": language, "label_identity": {level: {"id": f"{language}:{sample_id}", "namespace": f"discogem:{level}"} for level in LABELS}},
                }
                records[split].append(record)
                split_counts[split] += 1
    finally:
        for split, split_records in records.items():
            write_records(output_dir / f"{split}.json", split_records)
    (output_dir / "original_votes_v1").write_text("complete\n")
    print(f"multilevel/{variant}: train={split_counts['train']} dev={split_counts['dev']} test={split_counts['test']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare paper-compatible DiscoGeM datasets.")
    parser.add_argument("--discogem-path", type=Path, default=Path("data/raw/DiscoGeM/DiscoGeM 2.0/DiscoGeM2.0_annotation.tgz"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/datasets/text_pair"))
    parser.add_argument("--multilevel-output-dir", type=Path, default=Path("data/datasets/text_pair"), help="Root for the multilevel_label_distribution export.")
    parser.add_argument("--skip-multilevel", action="store_true", help="Only write the single-level text-pair exports.")
    parser.add_argument("--variants", nargs="+", choices=("english", "multilingual"), default=("english", "multilingual"))
    args = parser.parse_args()
    download_discogem(args.discogem_path)
    rows = _rows(args.discogem_path)
    if "english" in args.variants:
        _write_variant(rows, args.output_dir, "english", ("en",))
        if not args.skip_multilevel:
            _write_multilevel_variant(rows, args.multilevel_output_dir, "english", ("en",))
    if "multilingual" in args.variants:
        _write_variant(rows, args.output_dir, "multilingual", LANGUAGES)
        if not args.skip_multilevel:
            _write_multilevel_variant(rows, args.multilevel_output_dir, "multilingual", LANGUAGES)


if __name__ == "__main__":
    main()
