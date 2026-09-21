"""I/O helpers for SPInDa's public JSON-array data contract."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List


def load_records(path: Path, *, kind: str = "records") -> List[Any]:
    """Load a public ``.json`` file whose top-level value is an array."""
    if path.suffix != ".json":
        raise ValueError(f"{kind.capitalize()} must use a .json path, got {path}.")
    if not path.is_file():
        raise FileNotFoundError(f"{kind.capitalize()} file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a top-level JSON array of {kind}.")
    return payload


def split_path(directory: Path, split: str) -> Path:
    """Return the required public JSON split path."""
    return directory / f"{split}.json"

def write_records(path: Path, records: Iterable[Any]) -> None:
    """Write a public top-level JSON array to a ``.json`` file."""
    if path.suffix != ".json":
        raise ValueError(f"Public SPInDa records must use a .json path, got {path}.")
    path.write_text(json.dumps(list(records), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
