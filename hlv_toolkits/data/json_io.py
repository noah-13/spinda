"""I/O helpers for the public JSON-array dataset and prediction contracts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List


def load_records(path: Path, *, kind: str = "records") -> List[Any]:
    """Load a public JSON file whose top-level value must be an array."""
    if not path.is_file():
        raise FileNotFoundError(f"{kind.capitalize()} file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        try:
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid legacy JSONL in {path}: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a top-level JSON array of {kind}.")
    return payload


def split_path(directory: Path, split: str) -> Path:
    """Return the JSON split path, falling back to a legacy JSONL file."""
    path = directory / f"{split}.json"
    return path if path.is_file() or not (directory / f"{split}.jsonl").is_file() else directory / f"{split}.jsonl"

def write_records(path: Path, records: Iterable[Any]) -> None:
    """Write records as an indented JSON array, suitable for users to inspect."""
    path.write_text(json.dumps(list(records), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
