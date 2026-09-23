"""Export TGeGUM / Humans-and-Domains annotations into HLV formats."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from spinda.data.tie_breaking import tied_argmax
from spinda.data.json_io import write_records
from spinda.scripts.download_data import download_humans_and_domains

TASKS = ("genre", "topic1", "topic2")
UNITS = ("sent",)


def _normalise_label(task: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{task} annotation must be a non-empty string.")
    value = value.strip()
    return "No Topic" if task in {"topic1", "topic2"} and value.casefold() == "no topic" else value


def _load(input_dir: Path, unit: str) -> dict[str, dict[str, Mapping[str, Any]]]:
    result = {}
    for split in ("train", "dev", "test"):
        path = input_dir / f"{split}-{unit}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Raw TGeGUM file not found: {path}. Run download_data humans_and_domains first.") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must be an object keyed by item ID.")
        result[split] = payload
    return result


def _task_labels(splits: Mapping[str, Mapping[str, Mapping[str, Any]]], task: str) -> list[str]:
    labels = set()
    for rows in splits.values():
        for row in rows.values():
            annotations = row.get(f"annotations-{task}")
            if not isinstance(annotations, list) or not annotations:
                raise ValueError(f"Missing annotations-{task}.")
            labels.update(_normalise_label(task, value) for value in annotations)
    return sorted(labels)


def _record(item_id: str, row: Mapping[str, Any], split: str, task: str, label_to_id: Mapping[str, int]) -> dict[str, Any]:
    text, annotations = row.get("text"), row.get(f"annotations-{task}")
    if not isinstance(text, str) or not text.strip() or not isinstance(annotations, list) or not annotations:
        raise ValueError(f"{split} item {item_id} has invalid text or annotations-{task}.")
    votes = [label_to_id[_normalise_label(task, value)] for value in annotations]
    return {
        "id": f"humans_and_domains:{task}:{split}:{item_id}",
        "text": text,
        "annotation_labels": votes,
        "meta": {"label_identity": {"id": f"{split}:{item_id}", "namespace": f"tgegum:{task}"}, "source_id": item_id, "gold_genre": row.get("gold_genre"), "annotators": row.get("annotators")},
    }


def _write_single_task(splits: Mapping[str, Mapping[str, Mapping[str, Any]]], output_dir: Path, task: str) -> dict[str, int]:
    labels = _task_labels(splits, task)
    label_to_id = {label: index for index, label in enumerate(labels)}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "normalized_topics_v1").unlink(missing_ok=True)
    manifest = {"format": "single_text_label_distribution", "label_mode": "soft", "labels": labels, "train_path": str(output_dir / "train.json"), "dev_path": str(output_dir / "dev.json")}
    (output_dir / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    counts = {}
    for split, rows in splits.items():
        records = [_record(item_id, row, split, task, label_to_id) for item_id, row in rows.items()]
        write_records(output_dir / f"{split}.json", records)
        counts[split] = len(records)
    (output_dir / "normalized_topics_v1").write_text("complete\n")
    return counts


def _write_multilevel(splits: Mapping[str, Mapping[str, Mapping[str, Any]]], output_dir: Path) -> dict[str, int]:
    labels = {f"level{index + 1}": _task_labels(splits, task) for index, task in enumerate(TASKS)}
    ids = {level: {label: index for index, label in enumerate(values)} for level, values in labels.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "normalized_topics_v1").unlink(missing_ok=True)
    manifest = {"format": "single_text_multidimensional_label_distribution", "label_mode": "soft", "level_labels": labels, "train_path": str(output_dir / "train.json"), "dev_path": str(output_dir / "dev.json")}
    (output_dir / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    counts = {}
    for split, rows in splits.items():
        records = []
        for item_id, row in rows.items():
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{split} item {item_id} has invalid text.")
            votes, human_dists, hard_labels = {}, {}, {}
            for level, task in zip(("level1", "level2", "level3"), TASKS):
                raw = row.get(f"annotations-{task}")
                if not isinstance(raw, list) or not raw:
                    raise ValueError(f"{split} item {item_id} has invalid annotations-{task}.")
                level_votes = [ids[level][_normalise_label(task, value)] for value in raw]
                votes[level] = level_votes
                distribution = [level_votes.count(index) / len(level_votes) for index in range(len(labels[level]))]
                human_dists[level] = distribution
                hard_labels[level] = tied_argmax(distribution, str(item_id), f"humans_and_domains:{level}")
            records.append({"_schema": "SingleTextMultilevelSample", "id": f"humans_and_domains:multilevel:{split}:{item_id}", "task": "humans_and_domains", "split": split, "source": "tgegum", "text": text, "annotation_labels": votes, "meta": {"label_identity": {level: {"id": f"{split}:{item_id}", "namespace": f"tgegum:{task}"} for level, task in zip(labels, TASKS)}, "source_id": item_id, "dimension_tasks": {"level1": "genre", "level2": "topic1", "level3": "topic2"}}})
        write_records(output_dir / f"{split}.json", records)
        counts[split] = len(records)
    (output_dir / "normalized_topics_v1").write_text("complete\n")
    return counts


def prepare_humans_and_domains(input_dir: Path, single_text_root: Path, units: tuple[str, ...] = UNITS) -> None:
    for unit in units:
        if unit not in UNITS:
            raise ValueError(f"Unknown unit {unit!r}; expected one of {UNITS}.")
        splits = _load(input_dir, unit)
        for task in TASKS:
            counts = _write_single_task(splits, single_text_root / "humans_and_domains" / task, task)
            print(f"{unit}/{task}: train={counts['train']} dev={counts['dev']} test={counts['test']}")
        counts = _write_multilevel(splits, single_text_root / "humans_and_domains" / "multilevel")
        print(f"{unit}/multilevel: train={counts['train']} dev={counts['dev']} test={counts['test']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official Humans-and-Domains/TGeGUM annotations")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/humans_and_domains"))
    parser.add_argument("--single-text-output-root", type=Path, default=Path("data/datasets/single_text"))
    parser.add_argument("--units", nargs="+", choices=UNITS, default=list(UNITS))
    args = parser.parse_args()
    download_humans_and_domains(args.input_dir)
    prepare_humans_and_domains(args.input_dir, args.single_text_output_root, tuple(args.units))


if __name__ == "__main__":
    main()
