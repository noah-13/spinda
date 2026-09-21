"""Reader for single-text datasets with one distribution per label level."""
from __future__ import annotations

import json
from typing import List

from hlv_toolkits.data.readers.multilevel_reader import TextPairMultilevelJSONLReader
from hlv_toolkits.data.json_io import load_records, split_path
from hlv_toolkits.data.schemas import SingleTextMultilevelSample, Split
from hlv_toolkits.data.tie_breaking import tied_argmax


class SingleTextMultilevelJSONLReader(TextPairMultilevelJSONLReader):
    """Read the single-text multilevel JSONL contract."""

    DATA_FORMAT = "single_text_multidimensional_label_distribution"

    def load_split(self, split: Split) -> List[SingleTextMultilevelSample]:
        path = split_path(self.data_path, str(split)) if self.data_path.is_dir() else self.data_path
        rows: List[SingleTextMultilevelSample] = []
        for line_number, payload in enumerate(load_records(path, kind="dataset records"), 1):
            row_split = "dev" if payload.get("split") in {"valid", "validation"} else payload.get("split", split)
            target = "dev" if split in {"valid", "validation"} else split
            if row_split != target:
                continue
            text = payload.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError(f"Line {line_number} in {path} must contain a non-empty string text.")
            annotation_labels = payload.get("annotation_labels")
            if not isinstance(annotation_labels, dict) or set(annotation_labels) != set(self.dimension_names):
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
            sample_id = payload.get("id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Line {line_number} in {path} must contain a non-empty string id.")
            rows.append(
                SingleTextMultilevelSample(
                    id=sample_id, task=payload.get("task", self.task), split=target,
                    source=payload.get("source"), meta=dict(payload.get("meta") or {}), text=text,
                    hard_labels={level: tied_argmax(human_dists[level], sample_id, f"{self.task}:{level}") for level in self.dimension_names},
                    human_dists=human_dists,
                    annotation_labels=votes,
                )
            )
        return rows
