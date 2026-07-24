from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from hlv_toolkits.data.readers.base import BaseReader
from hlv_toolkits.data.schemas import AnySample, DiscoGeMMultiLevelSample, NLIDistributionSample, NLISample, Split
from hlv_toolkits.data.serialization import sample_from_json_dict


class ProcessedJSONLReader(BaseReader):
    """Reader for canonical JSONL files produced by the preprocessing CLI."""

    def __init__(
        self,
        data_path: Optional[str] = None,
        task: str = "nli",
        source: str = "processed",
        label_level: str = "level2",
        label_mode: str = "soft",
        language: str = "en",
    ) -> None:
        super().__init__(task=task)
        self.data_path = Path(data_path) if data_path else None
        self.source = source
        self.label_level = label_level
        self.label_mode = label_mode
        self.language = language

    def _resolve_path(self, split: Split) -> Path:
        if self.data_path is None:
            raise ValueError("data_path must be provided for ProcessedJSONLReader")
        if self.data_path.is_dir():
            return self.data_path / f"{split}.jsonl"
        return self.data_path

    def _project_discogem(self, sample: DiscoGeMMultiLevelSample) -> AnySample:
        if self.label_level == "all":
            return sample

        if self.label_mode == "soft":
            dist = sample.human_dists.get(self.label_level, [])
            if not dist:
                raise ValueError(f"Missing {self.label_level} human_dist for sample {sample.id}.")
            label = max(range(len(dist)), key=lambda i: dist[i])
            return NLIDistributionSample(
                id=sample.id,
                task=sample.task,
                split=sample.split,
                source=sample.source,
                premise=sample.premise,
                hypothesis=sample.hypothesis,
                label=label,
                human_dist=list(dist),
                meta=sample.meta,
            )

        label = sample.hard_labels.get(self.label_level, -1)
        if label < 0:
            raise ValueError(f"Missing {self.label_level} hard label for sample {sample.id}.")
        return NLISample(
            id=sample.id,
            task=sample.task,
            split=sample.split,
            source=sample.source,
            premise=sample.premise,
            hypothesis=sample.hypothesis,
            label=label,
            meta=sample.meta,
        )

    def load_split(self, split: Split) -> List[AnySample]:
        path = self._resolve_path(split)
        if not path.exists():
            raise FileNotFoundError(f"Processed JSONL file not found: {path}")

        samples: List[AnySample] = []
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num} in {path}: {e}")
                sample = sample_from_json_dict(payload)

                # A single processed JSONL file can contain train/dev/test rows.
                # Filter by the requested split before projecting labels; otherwise
                # load_train/load_dev/load_test would all read the complete file.
                sample_split = getattr(sample, "split", None)
                target_split = "dev" if split in {"valid", "validation"} else str(split)
                normalized_sample_split = (
                    "dev" if sample_split in {"valid", "validation"} else str(sample_split)
                )
                if sample_split is not None and normalized_sample_split != target_split:
                    continue

                if self.task == "discogem" and isinstance(sample, DiscoGeMMultiLevelSample):
                    samples.append(self._project_discogem(sample))
                else:
                    samples.append(sample)
        return samples
