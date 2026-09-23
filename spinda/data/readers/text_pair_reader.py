"""Reader for the public text-pair classification dataset contract."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Iterator, List, Literal, Optional, Sequence

from spinda.data.readers.base import BaseReader
from spinda.data.json_io import load_records, split_path
from spinda.data.tie_breaking import tied_argmax, annotation_argmax
from spinda.data.schemas import Split, TextPairClassificationSample, TextPairDistributionSample

TEXT_PAIR_TASK = "text_pair_label_distribution"
MANIFEST_FILENAME = "dataset.json"
LabelMode = Literal["hard", "soft", "soft_to_hard"]


class TextPairClassificationJSONReader(BaseReader):
    """Load hard labels, soft distributions, or explicit soft-to-hard conversions.

    New training calls use ``format``, ``train_path`` and an optional ``dev_path``.
    ``data_dir`` remains available for prediction/evaluation of legacy exported
    datasets that carry their metadata in ``dataset.json``.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        *,
        data_format: Optional[str] = None,
        train_path: Optional[str] = None,
        dev_path: Optional[str] = None,
        labels: Optional[Sequence[str]] = None,
        label_mode: Optional[LabelMode] = None,
    ) -> None:
        super().__init__(task=TEXT_PAIR_TASK)
        if data_dir is not None and train_path is not None:
            raise ValueError("Specify either data_dir or train_path, not both.")
        if data_dir is None and train_path is None:
            raise ValueError("train_path is required when data_dir is not provided.")
        if label_mode is not None and label_mode not in {"hard", "soft", "soft_to_hard"}:
            raise ValueError("label_mode must be one of: hard, soft, soft_to_hard")

        self.data_dir = Path(data_dir) if data_dir is not None else None
        manifest_label_mode: Optional[LabelMode] = None
        manifest_labels: Optional[List[str]] = None
        if self.data_dir is not None:
            manifest_label_mode, manifest_labels = self._load_manifest()
            self.train_path = split_path(self.data_dir, "train")
            self.dev_path = split_path(self.data_dir, "dev")
            self.test_path = split_path(self.data_dir, "test")
            if data_format is not None and data_format != TEXT_PAIR_TASK:
                raise ValueError(f"format must be {TEXT_PAIR_TASK!r}.")
            self.source = str(self.data_dir)
        else:
            if data_format != TEXT_PAIR_TASK:
                raise ValueError(f"format is required and must be {TEXT_PAIR_TASK!r}.")
            self.train_path = Path(train_path)  # type: ignore[arg-type]
            self.dev_path = Path(dev_path) if dev_path is not None else None
            self.test_path = None
            self.source = str(self.train_path.parent)

        # CLI/config values override dataset.json. They still have to agree
        # with the actual source rows, except for explicit soft_to_hard.
        self.label_mode = label_mode if label_mode is not None else manifest_label_mode
        self.source_label_mode = self._detect_source_label_mode(self.train_path)
        self.label_mode = self._resolve_label_mode(self.label_mode, self.source_label_mode)
        self.use_soft_labels = self.label_mode == "soft"

        provided_labels = list(labels) if labels is not None else manifest_labels
        if provided_labels is None:
            num_labels = self._infer_num_labels(self.train_path, self.source_label_mode)
            self.labels = [f"label{index}" for index in range(num_labels)]
            warnings.warn(
                f"No labels were provided for {self.train_path}; using {self.labels}.",
                UserWarning,
                stacklevel=2,
            )
        else:
            self.labels = self._validate_labels(provided_labels, "labels")

    @property
    def has_dev(self) -> bool:
        return self.dev_path is not None

    def _load_manifest(self) -> tuple[Optional[LabelMode], Optional[List[str]]]:
        assert self.data_dir is not None
        manifest_path = self.data_dir / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {manifest_path}: {exc}") from exc
        if manifest.get("format") != TEXT_PAIR_TASK:
            raise ValueError(f"{manifest_path} must contain format={TEXT_PAIR_TASK!r}.")

        label_mode = manifest.get("label_mode")
        if label_mode is not None and label_mode not in {"hard", "soft", "soft_to_hard"}:
            raise ValueError(f"{manifest_path} label_mode must be one of: hard, soft, soft_to_hard.")
        if label_mode is None:
            warnings.warn(
                f"{manifest_path} does not define label_mode; detecting it from train.json.",
                UserWarning,
                stacklevel=2,
            )
        labels = manifest.get("labels")
        return label_mode, None if labels is None else self._validate_labels(labels, str(manifest_path))

    @staticmethod
    def _validate_labels(labels: Any, context: str) -> List[str]:
        if not isinstance(labels, list) or len(labels) < 2 or any(
            not isinstance(label, str) or not label for label in labels
        ):
            raise ValueError(f"{context} must contain at least two non-empty strings.")
        if len(set(labels)) != len(labels):
            raise ValueError(f"{context} must contain unique labels.")
        return list(labels)

    @staticmethod
    def _record_source_label_mode(record: Any, path: Path, line_number: int) -> Literal["hard", "soft"]:
        if not isinstance(record, dict):
            raise ValueError(f"Line {line_number} in {path} must be a JSON object.")
        annotations = record.get("annotation_labels")
        if not isinstance(annotations, list) or not annotations:
            raise ValueError(f"Line {line_number} in {path} annotation_labels must be a non-empty list.")
        if any(isinstance(label, bool) or not isinstance(label, int) or label < 0 for label in annotations):
            raise ValueError(f"Line {line_number} in {path} annotation_labels must contain non-negative integer indices.")
        return "hard" if len(annotations) == 1 else "soft"

    def _read_records(self, path: Path) -> Iterator[tuple[int, Any]]:
        if not path.is_file():
            raise FileNotFoundError(f"Dataset split not found: {path}")
        records: list[tuple[int, dict[str, Any]]] = []
        max_annotation = -1
        for line_number, record in enumerate(load_records(path, kind="dataset records"), 1):
                self._record_source_label_mode(record, path, line_number)
                annotations = record["annotation_labels"]
                max_annotation = max(max_annotation, max(annotations))
                records.append((line_number, record))
        num_labels = len(self.labels) if hasattr(self, "labels") else max_annotation + 1
        for line_number, record in records:
            annotations = record["annotation_labels"]
            if max(annotations) >= num_labels:
                raise ValueError(f"Line {line_number} in {path} annotation_labels contains an out-of-range index.")
            counts = [0] * num_labels
            for label in annotations:
                counts[label] += 1
            record["label"] = annotation_argmax(counts, record, TEXT_PAIR_TASK)
            record["label_distribution"] = [count / len(annotations) for count in counts]
            yield line_number, record

    def _detect_source_label_mode(self, path: Path) -> Literal["hard", "soft"]:
        detected_mode: Literal["hard", "soft"] = "hard"
        found_records = False
        for line_number, record in self._read_records(path):
            found_records = True
            row_mode = self._record_source_label_mode(record, path, line_number)
            if row_mode == "soft":
                detected_mode = "soft"
        if not found_records:
            raise ValueError(f"Dataset split is empty: {path}")
        return detected_mode

    @staticmethod
    def _resolve_label_mode(
        requested_mode: Optional[LabelMode], source_mode: Literal["hard", "soft"]
    ) -> LabelMode:
        if requested_mode is None:
            warnings.warn(
                f"Detected {source_mode} labels from train.json; using label_mode={source_mode!r}.",
                UserWarning,
                stacklevel=3,
            )
            return source_mode
        if requested_mode == "soft_to_hard":
            if source_mode != "soft":
                raise ValueError("label_mode='soft_to_hard' requires two or more annotation_labels per example.")
            return requested_mode
        if requested_mode != source_mode:
            raise ValueError(
                f"label_mode={requested_mode!r} does not match {source_mode!r} data. "
                "Use soft_to_hard only when converting soft distributions to hard labels."
            )
        return requested_mode

    def _infer_num_labels(self, path: Path, source_mode: Literal["hard", "soft"]) -> int:
        if source_mode == "soft":
            distribution_size: Optional[int] = None
            for line_number, record in self._read_records(path):
                self._record_source_label_mode(record, path, line_number)
                distribution = record["label_distribution"]
                if not isinstance(distribution, list) or not distribution:
                    raise ValueError(f"Line {line_number} in {path} label_distribution must be a non-empty list.")
                if distribution_size is not None and len(distribution) != distribution_size:
                    raise ValueError(f"Line {line_number} in {path} label_distribution has an inconsistent length.")
                distribution_size = len(distribution)
            if distribution_size is None or distribution_size < 2:
                raise ValueError(f"Could not infer at least two labels from {path}.")
            return distribution_size

        max_label = -1
        for line_number, record in self._read_records(path):
            self._record_source_label_mode(record, path, line_number)
            label = record["label"]
            if isinstance(label, bool) or not isinstance(label, int) or label < 0:
                raise ValueError(f"Line {line_number} in {path} label must be a non-negative integer.")
            max_label = max(max_label, label)
        if max_label < 1:
            raise ValueError(f"Could not infer at least two labels from {path}.")
        return max_label + 1

    @staticmethod
    def _hard_label_from_distribution(distribution: Sequence[float], sample_id: str, random_ties: bool) -> int:
        maximum = max(distribution)
        tied = [index for index, value in enumerate(distribution) if value == maximum]
        if len(tied) == 1 or not random_ties:
            return tied[0]
        return tied_argmax(distribution, sample_id, TEXT_PAIR_TASK)

    def _path_for_split(self, split: str) -> Path:
        if split == "train":
            return self.train_path
        if split == "dev":
            if self.dev_path is None:
                raise FileNotFoundError("No dev_path was provided for this training dataset.")
            return self.dev_path
        if split == "test" and self.test_path is not None:
            return self.test_path
        raise FileNotFoundError(f"No {split}_path was provided for this dataset.")

    def load_split(self, split: Split) -> List[TextPairClassificationSample]:
        normalized_split = "dev" if split in {"valid", "validation"} else str(split)
        if normalized_split not in {"train", "dev", "test"}:
            raise ValueError(f"Unsupported text-pair split: {split}")
        path = self._path_for_split(normalized_split)
        samples: List[TextPairClassificationSample] = []
        seen_ids: set[str] = set()
        required_fields = {"id", "text_a", "text_b"}
        for line_number, record in self._read_records(path):
            row_mode = self._record_source_label_mode(record, path, line_number)
            # A soft manifest may legitimately contain a one-vote example.
            # Its empirical distribution is one-hot, but it must remain part
            # of the soft dataset rather than being rejected as a hard row.
            one_vote_soft = self.source_label_mode == "soft" and self.label_mode in {"soft", "soft_to_hard"} and row_mode == "hard"
            if row_mode != self.source_label_mode and not one_vote_soft:
                raise ValueError(f"Line {line_number} in {path} label type does not match train.json.")
            if not required_fields.issubset(record):
                raise ValueError(f"Line {line_number} in {path} must contain {sorted(required_fields)}.")
            if not isinstance(record["id"], str) or not record["id"] or record["id"] in seen_ids:
                raise ValueError(f"Line {line_number} in {path} has a missing or duplicate id.")
            if not isinstance(record["text_a"], str) or not isinstance(record["text_b"], str):
                raise ValueError(f"Line {line_number} in {path} text_a and text_b must be strings.")
            seen_ids.add(record["id"])
            common = dict(
                id=record["id"], task=TEXT_PAIR_TASK, split=normalized_split,
                source=self.source, text_a=record["text_a"], text_b=record["text_b"], meta=dict(record.get("meta") or {}),
            )
            if self.source_label_mode == "soft":
                dist = record["label_distribution"]
                if (
                    not isinstance(dist, list)
                    or len(dist) != len(self.labels)
                    or any(isinstance(x, bool) or not isinstance(x, (int, float)) or x < 0 for x in dist)
                ):
                    raise ValueError(
                        f"Line {line_number} in {path} must have a non-negative label_distribution "
                        f"with {len(self.labels)} values, one per label."
                    )
                dist = [float(x) for x in dist]
                if abs(sum(dist) - 1.0) > 1e-6:
                    raise ValueError(f"Line {line_number} in {path} label_distribution must sum to 1.")
                hard_label = record["label"]
                samples.append(TextPairDistributionSample(label=hard_label, human_dist=dist, annotation_labels=list(record["annotation_labels"]), **common))
            else:
                label = record["label"]
                if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(self.labels):
                    raise ValueError(f"Line {line_number} in {path} has a label outside [0, {len(self.labels) - 1}].")
                samples.append(TextPairClassificationSample(label=label, **common))
        return samples
