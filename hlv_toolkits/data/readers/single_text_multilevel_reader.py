"""Reader for single-text datasets with one distribution per label level."""
from __future__ import annotations

import json
from typing import List

from hlv_toolkits.data.readers.multilevel_reader import TextPairMultilevelJSONLReader
from hlv_toolkits.data.schemas import SingleTextMultilevelSample, Split
from hlv_toolkits.data.tie_breaking import tied_argmax


class SingleTextMultilevelJSONLReader(TextPairMultilevelJSONLReader):
    """Read the single-text multilevel JSONL contract."""

    DATA_FORMAT = "single_text_multilevel_label_distribution"

    def load_split(self, split: Split) -> List[SingleTextMultilevelSample]:
        path = self.data_path / f"{split}.jsonl" if self.data_path.is_dir() else self.data_path
        rows: List[SingleTextMultilevelSample] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}: {exc}") from exc
            row_split = "dev" if payload.get("split") in {"valid", "validation"} else payload.get("split", split)
            target = "dev" if split in {"valid", "validation"} else split
            if row_split != target:
                continue
            text = payload.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError(f"Line {line_number} in {path} must contain a non-empty string text.")
            raw_distributions = payload.get("human_dists")
            if not isinstance(raw_distributions, dict) or set(raw_distributions) != set(self.LEVEL_ORDER):
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
            sample_id = payload.get("id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Line {line_number} in {path} must contain a non-empty string id.")
            rows.append(
                SingleTextMultilevelSample(
                    id=sample_id, task=payload.get("task", self.task), split=target,
                    source=payload.get("source"), meta=dict(payload.get("meta") or {}), text=text,
                    hard_labels={level: tied_argmax(human_dists[level], sample_id, f"{self.task}:{level}") for level in self.LEVEL_ORDER},
                    human_dists=human_dists,
                )
            )
        return rows
