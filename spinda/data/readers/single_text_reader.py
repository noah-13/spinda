"""Reader for the public single-text annotation-label dataset contract."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Optional, Sequence

from spinda.data.readers.base import BaseReader
from spinda.data.json_io import load_records, split_path
from spinda.data.tie_breaking import tied_argmax
from spinda.data.schemas import SingleTextClassificationSample, SingleTextDistributionSample, Split

SINGLE_TEXT_TASK = "single_text_label_distribution"
LabelMode = Literal["hard", "soft", "soft_to_hard"]


class SingleTextClassificationJSONReader(BaseReader):
    """Read single-text examples while retaining annotation votes for soft labels."""

    def __init__(self, data_dir: Optional[str] = None, *, data_format: Optional[str] = None,
                 train_path: Optional[str] = None, dev_path: Optional[str] = None,
                 labels: Optional[Sequence[str]] = None, label_mode: Optional[LabelMode] = None) -> None:
        super().__init__(SINGLE_TEXT_TASK)
        if (data_dir is None) == (train_path is None):
            raise ValueError("Specify exactly one of data_dir or train_path.")
        if data_dir is not None:
            self.data_dir = Path(data_dir)
            manifest_path = self.data_dir / "dataset.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}") from exc
            if manifest.get("format") != SINGLE_TEXT_TASK:
                raise ValueError(f"{manifest_path} must contain format={SINGLE_TEXT_TASK!r}.")
            self.train_path, self.dev_path, self.test_path = split_path(self.data_dir, "train"), split_path(self.data_dir, "dev"), split_path(self.data_dir, "test")
            source_labels, source_mode, self.source = manifest.get("labels"), manifest.get("label_mode"), str(self.data_dir)
        else:
            if data_format != SINGLE_TEXT_TASK:
                raise ValueError(f"format is required and must be {SINGLE_TEXT_TASK!r}.")
            self.train_path, self.dev_path, self.test_path = Path(train_path), Path(dev_path) if dev_path else None, None  # type: ignore[arg-type]
            source_labels, source_mode, self.source = None, label_mode, str(self.train_path.parent)
        values = list(labels) if labels is not None else source_labels
        if not isinstance(values, list) or len(values) < 2 or any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
            raise ValueError("labels must be unique, non-empty strings with at least two values.")
        self.labels = values
        self.label_mode = label_mode or source_mode
        if self.label_mode not in {"hard", "soft", "soft_to_hard"}:
            raise ValueError("label_mode must be hard, soft, or soft_to_hard.")
        self.use_soft_labels = self.label_mode == "soft"

    @property
    def has_dev(self) -> bool:
        return self.dev_path is not None

    def load_split(self, split: Split) -> List[SingleTextClassificationSample]:
        name = "dev" if split in {"valid", "validation"} else str(split)
        path = {"train": self.train_path, "dev": self.dev_path, "test": self.test_path}.get(name)
        if path is None:
            raise FileNotFoundError(f"No {name}_path was provided for this dataset.")
        if not path.is_file():
            raise FileNotFoundError(f"Dataset split not found: {path}")
        samples: List[SingleTextClassificationSample] = []
        seen: set[str] = set()
        for line_number, row in enumerate(load_records(path, kind="dataset records"), 1):
            identifier, text, votes = row.get("id"), row.get("text"), row.get("annotation_labels")
            if not isinstance(identifier, str) or not identifier or identifier in seen or not isinstance(text, str):
                raise ValueError(f"Line {line_number} in {path} must contain a unique string id and string text.")
            if not isinstance(votes, list) or not votes or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < len(self.labels) for v in votes):
                raise ValueError(f"Line {line_number} in {path} annotation_labels must contain valid label indices.")
            if self.label_mode == "hard" and len(votes) != 1:
                raise ValueError("label_mode='hard' requires exactly one annotation label per example.")
            seen.add(identifier)
            counts = [votes.count(i) for i in range(len(self.labels))]
            label = tied_argmax(counts, identifier, SINGLE_TEXT_TASK)
            common = dict(id=identifier, task=self.task, split=name, source=self.source, text=text, label=label)
            if self.label_mode in {"soft", "soft_to_hard"}:
                samples.append(SingleTextDistributionSample(human_dist=[count / len(votes) for count in counts], annotation_labels=list(votes), **common))
            else:
                samples.append(SingleTextClassificationSample(**common))
        return samples
