from __future__ import annotations
import json
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence
from hlv_toolkits.data.readers.base import BaseReader
from hlv_toolkits.data.tie_breaking import tied_argmax
from hlv_toolkits.data.schemas import MultilevelSample, Split

class TextPairMultilevelJSONLReader(BaseReader):
    LEVEL_ORDER = ("level1", "level2", "level3")
    DATA_FORMAT = "text_pair_multilevel_label_distribution"

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
        if not isinstance(level_labels, Mapping) or set(level_labels) != set(cls.LEVEL_ORDER):
            raise ValueError("dataset.json level_labels must contain exactly level1, level2, and level3.")
        validated = {}
        for level in cls.LEVEL_ORDER:
            labels = level_labels[level]
            if not isinstance(labels, Sequence) or isinstance(labels, str) or not labels:
                raise ValueError(f"dataset.json level_labels[{level!r}] must be a non-empty list of strings.")
            if any(not isinstance(label, str) or not label for label in labels) or len(set(labels)) != len(labels):
                raise ValueError(f"dataset.json level_labels[{level!r}] must contain unique, non-empty strings.")
            validated[level] = list(labels)
        return validated

    def load_split(self, split: Split) -> List[MultilevelSample]:
        path = self.data_path / f"{split}.jsonl" if self.data_path.is_dir() else self.data_path
        rows: List[MultilevelSample] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            p = json.loads(line)
            row_split = "dev" if p.get("split") in {"valid", "validation"} else p.get("split", split)
            target = "dev" if split in {"valid", "validation"} else split
            if row_split != target:
                continue
            raw_distributions = dict(p.get("human_dists") or {})
            if set(raw_distributions) != set(self.LEVEL_ORDER):
                raise ValueError(f"Line {line_number} in {path} must contain distributions for all label levels.")
            human_dists = {}
            for level in self.LEVEL_ORDER:
                distribution = raw_distributions[level]
                if (
                    not isinstance(distribution, list)
                    or len(distribution) != len(self.level_labels[level])
                    or any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 for value in distribution)
                    or abs(sum(distribution) - 1.0) > 1e-6
                ):
                    raise ValueError(
                        f"Line {line_number} in {path} {level} distribution must contain "
                        f"{len(self.level_labels[level])} non-negative probabilities summing to 1."
                    )
                human_dists[level] = [float(value) for value in distribution]
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
                        level: tied_argmax(human_dists[level], sample_id, f"{self.task}:{level}")
                        for level in self.LEVEL_ORDER
                    },
                    human_dists=human_dists,
                )
            )
        return rows
