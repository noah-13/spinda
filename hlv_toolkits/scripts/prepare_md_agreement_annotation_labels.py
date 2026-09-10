"""Export MD-Agreement into the public single-text annotation-vote format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hlv_toolkits.scripts.download_data import download_md_agreement


def _annotation_votes(row: dict[str, Any], source: Path, item_id: str) -> list[int]:
    raw = row.get("annotations")
    if not isinstance(raw, str):
        raise ValueError(f"{source} item {item_id} has no comma-separated annotations string.")
    try:
        votes = [int(value.strip()) for value in raw.split(",")]
    except ValueError as exc:
        raise ValueError(f"{source} item {item_id} has non-integer annotations: {raw!r}") from exc
    if len(votes) != 5 or any(vote not in {0, 1} for vote in votes):
        raise ValueError(f"{source} item {item_id} must have exactly five binary annotation votes.")
    return votes


def prepare_md_agreement(input_dir: Path, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "single_text_label_distribution",
        "label_mode": "soft",
        "labels": ["not_offensive", "offensive"],
        # Keep this manifest directly consumable by ``train --config``, just
        # like the text-pair exporters. Dataset provenance belongs in the
        # JSONL record metadata rather than the training configuration.
        "train_path": str(output_dir / "train.jsonl"),
        "dev_path": str(output_dir / "dev.jsonl"),
    }
    (output_dir / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        source = input_dir / f"MD-Agreement_{split}.json"
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Raw MD-Agreement split not found: {source}. Run download_data md_agreement first.") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{source} must contain an object keyed by item id.")
        records = []
        for item_id, row in payload.items():
            if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"{source} item {item_id} must contain non-empty text.")
            if row.get("split") != split:
                raise ValueError(f"{source} item {item_id} has split={row.get('split')!r}, expected {split!r}.")
            record = {
                "id": f"md_agreement:{split}:{item_id}",
                "text": row["text"],
                "annotation_labels": _annotation_votes(row, source, str(item_id)),
                "meta": {"domain": (row.get("other_info") or {}).get("domain"), "source_id": str(item_id)},
            }
            records.append(record)
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[split] = len(records)
        print(f"Wrote {len(records)} MD-Agreement samples -> {output_dir / f'{split}.jsonl'}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MD-Agreement annotation-vote JSONL files")
    parser.add_argument("--input-dir", type=Path, default=Path("data/external/md_agreement"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/single_text/md_agreement"))
    args = parser.parse_args()
    download_md_agreement(args.input_dir)
    prepare_md_agreement(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
