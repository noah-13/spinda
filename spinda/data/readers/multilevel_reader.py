from __future__ import annotations
import json
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence
from spinda.data.readers.base import BaseReader
from spinda.data.json_io import load_records, split_path
from spinda.data.tie_breaking import annotation_argmax
from spinda.data.schemas import MultilevelSample, Split

class TextPairMultilevelJSONReader(BaseReader):
    LEVEL_ORDER = ("level1", "level2", "level3")
    DATA_FORMAT = "text_pair_multidimensional_label_distribution"

    def __init__(
        self,
        data_path: Optional[str] = None,
        task: str = "multilevel",
        *,
        level_labels: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        super().__init__(task=task)
        if data_path is None:
            raise ValueError("data_path is required")
        self.data_path = Path(data_path)
        manifest_labels = self._load_manifest_level_labels() if level_labels is None else level_labels
        self.level_labels = self._validate_level_labels(manifest_labels)
        self.dimension_names = tuple(self.level_labels)

    def _load_manifest_level_labels(self) -> Any:
        manifest_path = (self.data_path if self.data_path.is_dir() else self.data_path.parent) / "dataset.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {manifest_path}: {exc}") from exc
        if manifest.get("format") != self.DATA_FORMAT:
            raise ValueError(f"{manifest_path} must contain format={self.DATA_FORMAT!r}.")
        return manifest.get("level_labels")

    @classmethod
    def _validate_level_labels(cls, level_labels: Any) -> dict[str, List[str]]:
        if not isinstance(level_labels, Mapping) or not level_labels:
            raise ValueError("dataset.json level_labels must be a non-empty mapping of dimension names to label lists.")
        validated = {}
        for level, labels in level_labels.items():
            if not isinstance(level, str) or not level:
                raise ValueError("dataset.json dimension names must be non-empty strings.")
            if not isinstance(labels, Sequence) or isinstance(labels, str) or len(labels) < 2:
                raise ValueError(f"dataset.json level_labels[{level!r}] must be a non-empty list of strings.")
            if any(not isinstance(label, str) or not label for label in labels) or len(set(labels)) != len(labels):
                raise ValueError(f"dataset.json level_labels[{level!r}] must contain unique, non-empty strings.")
            validated[level] = list(labels)
        return validated

    def load_split(self, split: Split) -> List[MultilevelSample]:
        path = split_path(self.data_path, str(split)) if self.data_path.is_dir() else self.data_path
        rows: List[MultilevelSample] = []
        for line_number, p in enumerate(load_records(path, kind="dataset records"), 1):
            row_split = "dev" if p.get("split") in {"valid", "validation"} else p.get("split", split)
            target = "dev" if split in {"valid", "validation"} else split
            if row_split != target:
                continue
            annotation_labels = p.get("annotation_labels")
            if not isinstance(annotation_labels, Mapping) or set(annotation_labels) != set(self.dimension_names):
                raise ValueError(f"Line {line_number} in {path} must contain annotation_labels for every manifest dimension.")
            votes = {}
            for level in self.dimension_names:
                values = annotation_labels[level]
                if not isinstance(values, list) or not values or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= len(self.level_labels[level]) for value in values):
                    raise ValueError(f"Line {line_number} in {path} {level} annotation_labels must be non-empty valid label indices.")
                votes[level] = list(values)
            if len({len(values) for values in votes.values()}) != 1:
                raise ValueError(f"Line {line_number} in {path} annotation_labels must have one aligned vote per dimension.")
            human_dists = {level: [values.count(index) / len(values) for index in range(len(self.level_labels[level]))] for level, values in votes.items()}
            sample_id = str(p["id"])
            rows.append(
                MultilevelSample(
                    id=sample_id,
                    task=p.get("task", self.task),
                    split=target,
                    source=p.get("source"),
                    meta=dict(p.get("meta") or {}),
                    text_a=str(p.get("text_a", "")),
                    text_b=str(p.get("text_b", "")),
                    hard_labels={
                        level: annotation_argmax(human_dists[level], p, f"{self.task}:{level}", level)
                        for level in self.dimension_names
                    },
                    human_dists=human_dists,
                    annotation_labels=votes,
                )
            )
        return rows
