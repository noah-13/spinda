"""Reader for single-text multi-label annotation distributions (used by MFRC)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

from hlv_toolkits.data.readers.base import BaseReader
from hlv_toolkits.data.json_io import load_records, split_path
from hlv_toolkits.data.tie_breaking import binary_threshold
from hlv_toolkits.data.schemas import SingleTextMultilabelDistributionSample, Split

SINGLE_TEXT_MULTILABEL_TASK = "single_text_multilabel_annotation_distribution"


class SingleTextMultilabelJSONLReader(BaseReader):
    def __init__(self, data_dir: Optional[str] = None, *, data_format: Optional[str] = None,
                 train_path: Optional[str] = None, dev_path: Optional[str] = None,
                 labels: Optional[Sequence[str]] = None) -> None:
        super().__init__(SINGLE_TEXT_MULTILABEL_TASK)
        if (data_dir is None) == (train_path is None):
            raise ValueError("Specify exactly one of data_dir or train_path.")
        if data_dir is not None:
            self.data_dir = Path(data_dir)
            manifest_path = self.data_dir / "dataset.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}") from exc
            if manifest.get("format") != SINGLE_TEXT_MULTILABEL_TASK:
                raise ValueError(f"{manifest_path} must contain format={SINGLE_TEXT_MULTILABEL_TASK!r}.")
            self.train_path, self.dev_path, self.test_path = split_path(self.data_dir, "train"), split_path(self.data_dir, "dev"), split_path(self.data_dir, "test")
            source_labels, self.source = manifest.get("labels"), str(self.data_dir)
        else:
            if data_format != SINGLE_TEXT_MULTILABEL_TASK:
                raise ValueError(f"format is required and must be {SINGLE_TEXT_MULTILABEL_TASK!r}.")
            self.train_path, self.dev_path, self.test_path = Path(train_path), Path(dev_path) if dev_path else None, None  # type: ignore[arg-type]
            source_labels, self.source = labels, str(self.train_path.parent)
        self.labels = list(labels) if labels is not None else source_labels
        if not isinstance(self.labels, list) or len(self.labels) < 2 or any(not isinstance(v, str) or not v for v in self.labels) or len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique, non-empty strings with at least two values.")

    @property
    def has_dev(self) -> bool:
        return self.dev_path is not None

    def load_split(self, split: Split) -> List[SingleTextMultilabelDistributionSample]:
        name = "dev" if split in {"valid", "validation"} else str(split)
        path = {"train": self.train_path, "dev": self.dev_path, "test": self.test_path}.get(name)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Dataset split not found: {path}")
        samples, seen = [], set()
        for line_number, row in enumerate(load_records(path, kind="dataset records"), 1):
            identifier, text, votes = row.get("id"), row.get("text"), row.get("annotation_label_sets")
            if not isinstance(identifier, str) or not identifier or identifier in seen or not isinstance(text, str) or not text.strip():
                raise ValueError(f"Line {line_number} in {path} must contain a unique id and non-empty text.")
            valid_vote = lambda vote: isinstance(vote, list) and all(isinstance(label, int) and not isinstance(label, bool) and 0 <= label < len(self.labels) for label in vote) and len(set(vote)) == len(vote)
            if not isinstance(votes, list) or not votes or not all(valid_vote(vote) for vote in votes):
                raise ValueError(f"Line {line_number} in {path} annotation_label_sets must be non-empty lists of valid, unique label indices.")
            seen.add(identifier)
            probs = [sum(label in vote for vote in votes) / len(votes) for label in range(len(self.labels))]
            samples.append(SingleTextMultilabelDistributionSample(id=identifier, task=self.task, split=name, source=self.source, text=text, labels=[binary_threshold(prob, identifier, f"{SINGLE_TEXT_MULTILABEL_TASK}:label:{index}") for index, prob in enumerate(probs)], human_probs=probs, annotation_label_sets=[list(vote) for vote in votes]))
        return samples
