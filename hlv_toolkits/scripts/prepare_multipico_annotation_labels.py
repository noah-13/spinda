"""Export official LeWiDi MultiPICo splits as text-pair annotation votes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from hlv_toolkits.scripts.download_data import download_multipico

LABELS = ["not_ironic", "ironic"]


def _text(value: Any, name: str, source: Path, item_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source} item {item_id} has no non-empty {name!r}.")
    return value


def _value(row: Mapping[str, Any], name: str, source: Path, item_id: str) -> str:
    value = row.get(name)
    if value is None or not str(value).strip():
        raise ValueError(f"{source} item {item_id} has no {name!r}.")
    return str(value)


def _labels(row: Mapping[str, Any], source: Path, item_id: str) -> list[int]:
    annotations = row.get("annotations")
    if not isinstance(annotations, Mapping) or not annotations:
        raise ValueError(f"{source} item {item_id} has no annotations object.")
    labels = []
    for annotator, value in annotations.items():
        try:
            label = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source} item {item_id} annotator {annotator!r} has invalid label {value!r}.") from exc
        if label not in {0, 1} or str(label) != str(value):
            raise ValueError(f"{source} item {item_id} annotator {annotator!r} has invalid label {value!r}.")
        labels.append(label)
    return labels


def prepare_multipico(splits: Mapping[str, Mapping[str, Mapping[str, Any]]], output_dir: Path, language: str | None = None) -> dict[str, int]:
    """Preserve the official LeWiDi MultiPICo train/dev/test split and votes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "text_pair_label_distribution",
        "label_mode": "soft",
        "labels": LABELS,
        "train_path": str(output_dir / "train.jsonl"),
        "dev_path": str(output_dir / "dev.jsonl"),
    }
    (output_dir / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    counts = {}
    for split in ("train", "dev", "test"):
        rows = splits.get(split)
        if not isinstance(rows, Mapping):
            raise ValueError(f"MultiPICo {split!r} split must be an object keyed by item ID.")
        source = Path(f"MP_{split}.json")
        count = 0
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for item_id, row in rows.items():
                if not isinstance(row, Mapping):
                    raise ValueError(f"{source} item {item_id} must be an object.")
                text, info = row.get("text"), row.get("other_info")
                if not isinstance(text, Mapping) or not isinstance(info, Mapping):
                    raise ValueError(f"{source} item {item_id} must contain text and other_info objects.")
                if row.get("split") != split:
                    raise ValueError(f"{source} item {item_id} has split={row.get('split')!r}, expected {split!r}.")
                row_language = _value(row, "lang", source, str(item_id))
                if language is not None and row_language != language:
                    continue
                record = {
                    "id": f"multipico:{split}:{item_id}",
                    "text_a": _text(text.get("post"), "post", source, str(item_id)),
                    "text_b": _text(text.get("reply"), "reply", source, str(item_id)),
                    "annotation_labels": _labels(row, source, str(item_id)),
                    "meta": {
                        "source": _value(info, "source", source, str(item_id)),
                        "language": row_language,
                        "language_variety": _value(info, "language_variety", source, str(item_id)),
                        "source_id": str(item_id),
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        counts[split] = count
    return counts


def _load_splits(input_dir: Path) -> dict[str, Mapping[str, Mapping[str, Any]]]:
    splits = {}
    for split in ("train", "dev", "test"):
        path = input_dir / f"MP_{split}.json"
        try:
            splits[split] = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Raw MultiPICo split not found: {path}. Run download_data multipico first.") from exc
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official MultiPICo annotation distributions")
    parser.add_argument("--input-dir", type=Path, default=Path("data/external/multipico"))
    parser.add_argument("--language", default=None, help="Only export one ISO language code, e.g. en.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/text_pair/multipico"))
    args = parser.parse_args()
    download_multipico(args.input_dir)
    counts = prepare_multipico(_load_splits(args.input_dir), args.output_dir, args.language)
    print(f"MultiPICo: train={counts['train']} dev={counts['dev']} test={counts['test']}")


if __name__ == "__main__":
    main()
