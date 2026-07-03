from __future__ import annotations

import ast
import csv
import io
import tarfile
from pathlib import Path
from typing import Dict, List, Literal, Optional

from hlv_toolkits.data.readers.base import NLIDistributionReader
from hlv_toolkits.data.schemas import (
    DISCOGEM_LEVEL1_LABELS,
    DISCOGEM_LEVEL2_LABELS,
    DISCOGEM_LEVEL2_TO_LEVEL1,
    DISCOGEM_LEVEL3_LABELS,
    DISCOGEM_LEVEL3_TO_LEVEL2,
    DiscoGeMLabelLevel,
    DiscoGeMMultiLevelSample,
    NLIDistributionSample,
    Split,
    get_discogem_num_labels,
    map_discogem_label,
)

DiscoGeMVersion = Literal["auto", "2.0"]
DiscoGeMLabelMode = Literal["soft", "hard"]

LEVEL_ORDER = ("level1", "level2", "level3")
LEVEL_LABELS = {
    "level1": DISCOGEM_LEVEL1_LABELS,
    "level2": DISCOGEM_LEVEL2_LABELS,
    "level3": DISCOGEM_LEVEL3_LABELS,
}
LEVEL_LABEL2ID = {
    level: {label: idx for idx, label in enumerate(labels)}
    for level, labels in LEVEL_LABELS.items()
}


class DiscoGeMReader(NLIDistributionReader):
    """Reader for DiscoGeM 2.0."""

    def __init__(
        self,
        data_path: Optional[str] = None,
        version: DiscoGeMVersion = "auto",
        label_mode: DiscoGeMLabelMode = "soft",
        label_level: DiscoGeMLabelLevel = "level2",
        language: str = "en",
        source: str = "discogem",
    ) -> None:
        super().__init__(task="discogem")
        self.data_path = Path(data_path) if data_path else None
        self.version = version
        self.label_mode = label_mode
        self.label_level = label_level
        self.language = language
        self.source = source
        self.is_multilevel = label_level == "all"
        self.label2id = None if self.is_multilevel else LEVEL_LABEL2ID[label_level]
        self.num_labels = get_discogem_num_labels("all" if self.is_multilevel else label_level)

    def _infer_default_data_path(self) -> Path:
        return Path("data/external/DiscoGeM/DiscoGeM 2.0/DiscoGeM2.0_annotation.tgz")

    @staticmethod
    def _norm_split(split: Split) -> str:
        if split in {"dev", "valid", "validation"}:
            return "dev"
        return split

    @staticmethod
    def _discogem_split_to_norm(raw_split: str) -> str:
        s = str(raw_split).strip().lower()
        if s in {"dev", "valid", "validation"}:
            return "dev"
        return s

    @staticmethod
    def _safe_float(v: object) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def _dist_from_dict(self, obj: Dict[str, object], target_level: str) -> List[float]:
        dist = [0.0] * len(LEVEL_LABELS[target_level])
        for label, value in obj.items():
            mapped = map_discogem_label(str(label).strip(), target_level)  # type: ignore[arg-type]
            if mapped not in LEVEL_LABEL2ID[target_level]:
                continue
            dist[LEVEL_LABEL2ID[target_level][mapped]] += self._safe_float(value)
        return dist

    def _parse_soft_dist(self, text: str, target_level: str) -> List[float]:
        obj = ast.literal_eval(text)
        if not isinstance(obj, dict):
            raise ValueError(f"Invalid DiscoGeM soft distribution: {text}")
        return self._dist_from_dict(obj, target_level)

    @staticmethod
    def _normalize(dist: List[float]) -> List[float]:
        total = sum(dist)
        if total <= 0:
            return dist
        return [x / total for x in dist]

    def _soft_dist_for_row(self, row: Dict[str, str], sample_id: str, target_level: str) -> List[float]:
        raw = row.get(f"WAWA_dist_{self.language}", "").strip()
        if not raw:
            raise ValueError(f"Missing 2.0 soft distribution for sample {sample_id}.")
        dist = self._parse_soft_dist(raw, target_level)
        dist = self._normalize(dist)
        if sum(dist) <= 0:
            raise ValueError(f"Soft distribution sum <= 0 for sample {sample_id}.")
        return dist

    def _hard_label_from_raw(self, raw: str, target_level: str, sample_id: str) -> int:
        mapped = map_discogem_label(raw, target_level)
        if mapped not in LEVEL_LABEL2ID[target_level]:
            raise ValueError(f"Unknown hard label '{raw}' mapped to '{mapped}' for sample {sample_id}.")
        return LEVEL_LABEL2ID[target_level][mapped]

    def _raw_hard_label_from_row(self, row: Dict[str, str]) -> str:
        return row.get(f"WAWA_{self.language}", "").strip()

    def _multi_level_hard_labels(self, raw: str, sample_id: str) -> Dict[str, int]:
        return {
            level: self._hard_label_from_raw(raw, level, sample_id)
            for level in LEVEL_ORDER
        }

    def _multi_level_soft_dists(self, row: Dict[str, str], sample_id: str) -> Dict[str, List[float]]:
        base_level3 = self._soft_dist_for_row(row, sample_id, "level3")
        level3 = base_level3
        level2 = [0.0] * len(DISCOGEM_LEVEL2_LABELS)
        level1 = [0.0] * len(DISCOGEM_LEVEL1_LABELS)

        for idx, label in enumerate(DISCOGEM_LEVEL3_LABELS):
            value = level3[idx]
            level2_label = DISCOGEM_LEVEL3_TO_LEVEL2[label]
            level2[LEVEL_LABEL2ID["level2"][level2_label]] += value
            level1_label = DISCOGEM_LEVEL2_TO_LEVEL1[level2_label]
            level1[LEVEL_LABEL2ID["level1"][level1_label]] += value

        return {
            "level1": self._normalize(level1),
            "level2": self._normalize(level2),
            "level3": self._normalize(level3),
        }

    def _multi_level_hard_sample(self, row: Dict[str, str], sample_id: str, arg1: str, arg2: str, split: Split) -> DiscoGeMMultiLevelSample:
        raw = self._raw_hard_label_from_row(row)
        if not raw:
            raise ValueError(f"Missing hard label for sample {sample_id}.")
        return DiscoGeMMultiLevelSample(
            id=sample_id,
            task=self.task,
            split=split,
            source=self.source,
            premise=arg1,
            hypothesis=arg2,
            hard_labels=self._multi_level_hard_labels(raw, sample_id),
        )

    def _multi_level_soft_sample(self, row: Dict[str, str], sample_id: str, arg1: str, arg2: str, split: Split) -> DiscoGeMMultiLevelSample:
        return DiscoGeMMultiLevelSample(
            id=sample_id,
            task=self.task,
            split=split,
            source=self.source,
            premise=arg1,
            hypothesis=arg2,
            human_dists=self._multi_level_soft_dists(row, sample_id),
        )

    def _hard_label_from_row(self, row: Dict[str, str], sample_id: str) -> int:
        raw = self._raw_hard_label_from_row(row)
        if not raw:
            raise ValueError(f"Missing hard label for sample {sample_id}.")
        return self._hard_label_from_raw(raw, self.label_level, sample_id)

    def _soft_label_from_row(self, row: Dict[str, str], sample_id: str) -> List[float]:
        return self._soft_dist_for_row(row, sample_id, self.label_level)

    def _detect_version(self, fieldnames: List[str]) -> str:
        if self.version == "2.0":
            return "2.0"
        fields = set(fieldnames)
        if f"WAWA_{self.language}" in fields and f"WAWA_dist_{self.language}" in fields:
            return "2.0"
        raise ValueError("Cannot auto-detect DiscoGeM 2.0 from file header.")

    def _open_rows(self, data_path: Path) -> csv.DictReader:
        if data_path.suffix == ".tgz":
            with tarfile.open(data_path, "r:gz") as tf:
                member = "DiscoGeM2.0_annotation/DiscoGeM2.0_items.csv"
                raw = tf.extractfile(member)
                if raw is None:
                    raise FileNotFoundError(f"Could not find {member} inside {data_path}")
                text = raw.read().decode("utf-8-sig")
        else:
            text = data_path.read_text(encoding="utf-8")

        sio = io.StringIO(text)
        first_line = sio.readline()
        delimiter = "\t" if first_line.count("\t") > first_line.count(",") else ","
        sio.seek(0)
        return csv.DictReader(sio, delimiter=delimiter)

    def load_split(self, split: Split):
        data_path = self.data_path or self._infer_default_data_path()
        if not data_path.exists():
            raise FileNotFoundError(f"DiscoGeM data file not found: {data_path}")

        samples = []
        reader = self._open_rows(data_path)
        if not reader.fieldnames:
            raise ValueError(f"No header found in DiscoGeM file: {data_path}")
        version = self._detect_version(reader.fieldnames)

        target_split = self._norm_split(split)
        for idx, row in enumerate(reader, 1):
            sample_id = (row.get("itemid") or "").strip() or f"discogem_{idx}"
            row_split = self._discogem_split_to_norm(row.get("split", "unknown"))
            if target_split != "unknown" and row_split != target_split:
                continue

            arg1 = (row.get(f"arg1_context_{self.language}") or "").strip()
            arg2 = (row.get(f"arg2_context_{self.language}") or "").strip()
            if not arg1 or not arg2:
                continue

            if self.label_level == "all":
                if self.label_mode == "soft":
                    sample = self._multi_level_soft_sample(row, sample_id, arg1, arg2, split)
                else:
                    sample = self._multi_level_hard_sample(row, sample_id, arg1, arg2, split)
            elif self.label_mode == "soft":
                human_dist = self._soft_label_from_row(row, sample_id)
                label_id = max(range(len(human_dist)), key=lambda i: human_dist[i])
                sample = NLIDistributionSample(
                    id=sample_id,
                    task=self.task,
                    split=split,
                    source=self.source,
                    premise=arg1,
                    hypothesis=arg2,
                    label=label_id,
                    human_dist=human_dist,
                )
            else:
                label_id = self._hard_label_from_row(row, sample_id)
                sample = NLIDistributionSample(
                    id=sample_id,
                    task=self.task,
                    split=split,
                    source=self.source,
                    premise=arg1,
                    hypothesis=arg2,
                    label=label_id,
                    human_dist=[],
                )

            samples.append(sample)

        return samples
