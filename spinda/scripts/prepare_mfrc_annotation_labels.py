"""Download and normalize MFRC while preserving each annotator's label set."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

from spinda.data.json_io import write_records

MFRC_LABELS = ["Care", "Equality", "Proportionality", "Loyalty", "Authority", "Purity", "Thin Morality", "Non-Moral"]
MFRC_LABEL_TO_ID = {label: index for index, label in enumerate(MFRC_LABELS)}
MFRC_DATASET_ID = "USC-MOLA-Lab/MFRC"
MFRC_DATASET_SPLIT = "train_dedup"


def _label_set(value: Any) -> list[int]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("MFRC annotation must be a non-empty comma-separated string.")
    names = [name.strip() for name in value.split(",")]
    unknown = [name for name in names if name not in MFRC_LABEL_TO_ID]
    if unknown:
        raise ValueError(f"Unknown MFRC labels: {', '.join(unknown)}")
    result = sorted({MFRC_LABEL_TO_ID[name] for name in names})
    if len(result) != len(names):
        raise ValueError(f"Repeated labels in MFRC annotation: {value!r}")
    return result


def _split_for(key: str) -> str:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "test" if bucket == 0 else "dev" if bucket == 1 else "train"


def prepare_mfrc(rows: Iterable[dict[str, Any]], output_dir: Path) -> dict[str, int]:
    """Aggregate Hugging Face's one-row-per-annotator records by comment."""
    grouped: OrderedDict[tuple[str, str, str], list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        text, subreddit, bucket = row.get("text"), row.get("subreddit"), row.get("bucket")
        if not isinstance(text, str) or not text.strip() or not all(isinstance(value, str) and value for value in (subreddit, bucket)):
            raise ValueError("Every MFRC row requires non-empty text, subreddit, and bucket strings.")
        grouped.setdefault((text, subreddit, bucket), []).append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "single_text_multilabel_annotation_distribution",
        "labels": MFRC_LABELS,
        "label_mode": "soft",
        "train_path": str(output_dir / "train.json"),
        "dev_path": str(output_dir / "dev.json"),
    }
    (output_dir / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "source": {
            "dataset": MFRC_DATASET_ID,
            "split": MFRC_DATASET_SPLIT,
            "aggregation": "one grouped item per (text, subreddit, bucket); source rows retain individual annotations",
        },
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    records = {split: [] for split in ("train", "dev", "test")}
    for ordinal, ((text, subreddit, bucket), annotations) in enumerate(grouped.items()):
        split = _split_for("\u241f".join((text, subreddit, bucket)))
        records[split].append({"id": f"mfrc:{split}:{ordinal}", "text": text, "annotation_label_sets": [_label_set(row.get("annotation")) for row in annotations], "meta": {"subreddit": subreddit, "bucket": bucket, "annotators": [row.get("annotator") for row in annotations], "confidence": [row.get("confidence") for row in annotations]}})
    for split, split_records in records.items():
        write_records(output_dir / f"{split}.json", split_records)
    return {split: len(split_records) for split, split_records in records.items()}


def _download_rows() -> list[dict[str, Any]]:
    from datasets import load_dataset
    return [dict(row) for row in load_dataset(MFRC_DATASET_ID, split=MFRC_DATASET_SPLIT)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare MFRC multi-label annotation distributions")
    parser.add_argument("--output-dir", type=Path, default=Path("data/datasets/single_text/mfrc"))
    args = parser.parse_args()
    counts = prepare_mfrc(_download_rows(), args.output_dir)
    print(f"MFRC: train={counts['train']} dev={counts['dev']} test={counts['test']}")


if __name__ == "__main__":
    main()
