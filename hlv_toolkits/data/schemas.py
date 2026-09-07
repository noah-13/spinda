# data/schemas.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Union

# -------------------------
# Common types
# -------------------------
Split = Literal["train", "dev", "valid", "validation", "test", "unknown"]


# =========================
# Label space (for NLI)
# =========================
# FIX the label order globally to avoid silent bugs across readers/datasets/metrics.
# 0: entailment
# 1: neutral
# 2: contradiction
NLI_LABELS: List[str] = ["entailment", "neutral", "contradiction"]
NLI_LABEL2ID = {label: i for i, label in enumerate(NLI_LABELS)}
NLI_ID2LABEL = {i: label for label, i in NLI_LABEL2ID.items()}
NLI_NUM_LABELS = len(NLI_LABELS)

# =========================
# Label space (for DiscoGeM)
# =========================
DiscoGeMLabelLevel = Literal["level1", "level2", "level3", "all"]

DISCOGEM_LEVEL3_LABELS: List[str] = [
    "asynchronous",
    "cause",
    "concession",
    "condition",
    "conjunction",
    "contrast",
    "disjunction",
    "equivalence",
    "exception",
    "instantiation",
    "level-of-detail",
    "manner",
    "norel",
    "purpose",
    "similarity",
    "substitution",
    "synchronous",
    "precedence",
    "succession",
    "reason",
    "result",
    "arg1-as-cond",
    "arg2-as-cond",
    "arg1-as-negcond",
    "arg2-as-negcond",
    "arg1-as-goal",
    "arg2-as-goal",
    "arg1-as-manner",
    "arg2-as-manner",
    "arg1-as-detail",
    "arg2-as-detail",
    "arg1-as-instance",
    "arg2-as-instance",
    "arg1-as-subst",
    "arg2-as-subst",
    "arg1-as-excpt",
    "arg2-as-excpt",
    "arg1-as-denier",
    "arg2-as-denier",
]
DISCOGEM_LEVEL2_LABELS: List[str] = [
    "asynchronous",
    "cause",
    "concession",
    "condition",
    "conjunction",
    "contrast",
    "disjunction",
    "equivalence",
    "exception",
    "instantiation",
    "level-of-detail",
    "manner",
    "norel",
    "purpose",
    "similarity",
    "substitution",
    "synchronous",
]
DISCOGEM_LEVEL1_LABELS: List[str] = [
    "temporal",
    "contingency",
    "comparison",
    "expansion",
    "norel",
]

DISCOGEM_LEVEL3_TO_LEVEL2: Dict[str, str] = {
    "asynchronous": "asynchronous",
    "synchronous": "synchronous",
    "cause": "cause",
    "concession": "concession",
    "condition": "condition",
    "conjunction": "conjunction",
    "contrast": "contrast",
    "disjunction": "disjunction",
    "equivalence": "equivalence",
    "exception": "exception",
    "instantiation": "instantiation",
    "level-of-detail": "level-of-detail",
    "manner": "manner",
    "norel": "norel",
    "purpose": "purpose",
    "similarity": "similarity",
    "substitution": "substitution",
    "precedence": "asynchronous",
    "succession": "asynchronous",
    "reason": "cause",
    "result": "cause",
    "arg1-as-goal": "purpose",
    "arg2-as-goal": "purpose",
    "arg1-as-cond": "condition",
    "arg1-as-negcond": "condition",
    "arg2-as-cond": "condition",
    "arg2-as-negcond": "condition",
    "arg1-as-denier": "concession",
    "arg2-as-denier": "concession",
    "arg1-as-instance": "instantiation",
    "arg2-as-instance": "instantiation",
    "arg1-as-detail": "level-of-detail",
    "arg2-as-detail": "level-of-detail",
    "arg1-as-excpt": "exception",
    "arg2-as-excpt": "exception",
    "arg1-as-manner": "manner",
    "arg2-as-manner": "manner",
    "arg1-as-subst": "substitution",
    "arg2-as-subst": "substitution",
}
DISCOGEM_LEVEL2_TO_LEVEL1: Dict[str, str] = {
    "synchronous": "temporal",
    "asynchronous": "temporal",
    "cause": "contingency",
    "purpose": "contingency",
    "condition": "contingency",
    "concession": "comparison",
    "contrast": "comparison",
    "similarity": "comparison",
    "equivalence": "expansion",
    "instantiation": "expansion",
    "level-of-detail": "expansion",
    "conjunction": "expansion",
    "disjunction": "expansion",
    "exception": "expansion",
    "manner": "expansion",
    "substitution": "expansion",
    "norel": "norel",
}

DISCOGEM_LABELS_BY_LEVEL: Dict[DiscoGeMLabelLevel, List[str]] = {
    "level1": DISCOGEM_LEVEL1_LABELS,
    "level2": DISCOGEM_LEVEL2_LABELS,
    "level3": DISCOGEM_LEVEL3_LABELS,
}

DISCOGEM_LEVEL_ORDER: List[str] = ["level1", "level2", "level3"]
DISCOGEM_LEVEL_NUM_LABELS: Dict[str, int] = {
    "level1": len(DISCOGEM_LEVEL1_LABELS),
    "level2": len(DISCOGEM_LEVEL2_LABELS),
    "level3": len(DISCOGEM_LEVEL3_LABELS),
}


def get_discogem_labels(level: DiscoGeMLabelLevel = "level3") -> List[str]:
    if level == "all":
        return [
            *(f"level1::{label}" for label in DISCOGEM_LEVEL1_LABELS),
            *(f"level2::{label}" for label in DISCOGEM_LEVEL2_LABELS),
            *(f"level3::{label}" for label in DISCOGEM_LEVEL3_LABELS),
        ]
    return list(DISCOGEM_LABELS_BY_LEVEL[level])


def get_discogem_label2id(level: DiscoGeMLabelLevel = "level3") -> Dict[str, int]:
    if level == "all":
        return {label: i for i, label in enumerate(get_discogem_labels(level))}
    labels = get_discogem_labels(level)
    return {label: i for i, label in enumerate(labels)}


def get_discogem_id2label(level: DiscoGeMLabelLevel = "level3") -> Dict[int, str]:
    label2id = get_discogem_label2id(level)
    return {i: label for label, i in label2id.items()}


def get_discogem_num_labels(level: DiscoGeMLabelLevel = "level3") -> int:
    if level == "all":
        return sum(DISCOGEM_LEVEL_NUM_LABELS[level_name] for level_name in DISCOGEM_LEVEL_ORDER)
    return len(DISCOGEM_LABELS_BY_LEVEL[level])


def get_discogem_level_num_labels() -> Dict[str, int]:
    return dict(DISCOGEM_LEVEL_NUM_LABELS)


def map_discogem_label(label: str, target_level: DiscoGeMLabelLevel) -> str:
    raw = str(label).strip()
    if not raw:
        raise ValueError("Cannot map empty DiscoGeM label.")

    if target_level == "level3":
        return raw
    if raw in DISCOGEM_LEVEL3_TO_LEVEL2:
        level2 = DISCOGEM_LEVEL3_TO_LEVEL2[raw]
    elif raw in DISCOGEM_LEVEL2_TO_LEVEL1:
        level2 = raw
    elif raw in DISCOGEM_LEVEL1_LABELS:
        level2 = raw
    else:
        raise ValueError(f"Unknown DiscoGeM label: {raw}")

    if target_level == "level2":
        return level2
    if level2 in DISCOGEM_LEVEL2_TO_LEVEL1:
        return DISCOGEM_LEVEL2_TO_LEVEL1[level2]
    if level2 in DISCOGEM_LEVEL1_LABELS:
        return level2
    raise ValueError(f"Cannot map DiscoGeM label '{raw}' to {target_level}.")


DISCOGEM_LABELS = get_discogem_labels("level3")
DISCOGEM_LABEL2ID = get_discogem_label2id("level3")
DISCOGEM_ID2LABEL = get_discogem_id2label("level3")
DISCOGEM_NUM_LABELS = get_discogem_num_labels("level3")


# =========================
# Base sample (task-agnostic)
# =========================
@dataclass
class BaseSample:
    """
    Task-agnostic sample representation.
    - Keep this stable. Everything else can evolve.
    """
    id: str
    task: str  # e.g., "nli", "qa", "ranking", "regression"
    split: Split = "unknown"
    source: Optional[str] = None  # e.g., "snli", "chaosnli-snli", "squad"
    meta: Dict[str, Any] = field(default_factory=dict)


# =========================
# NLI samples
# =========================
@dataclass
class MultilevelSample(BaseSample):
    """Text-pair sample with labels/distributions for all three hierarchy levels."""

    premise: str = ""
    hypothesis: str = ""
    hard_labels: Dict[str, int] = field(default_factory=dict)
    human_dists: Dict[str, List[float]] = field(default_factory=dict)


# =========================
# Prediction record (task-agnostic output)
# =========================
@dataclass
class PredictionRecord:
    """
    Standardized prediction dump format.
    Store only what you need for evaluation + debugging.
    """
    id: str
    task: str
    split: Split = "unknown"
    source: Optional[str] = None

    # Task outputs (keep flexible):
    outputs: Dict[str, Any] = field(default_factory=dict)

    # Optional debug extras:
    extras: Dict[str, Any] = field(default_factory=dict)


# Convenient unions (optional)
AnySample = Union[
    BaseSample,
    MultilevelSample,
]

# Current direct dataset contracts.
@dataclass
class TextPairClassificationSample(BaseSample):
    text_a: str = ""
    text_b: str = ""
    label: int = -1

@dataclass
class TextPairDistributionSample(TextPairClassificationSample):
    human_dist: List[float] = field(default_factory=list)
    annotation_labels: List[int] = field(default_factory=list)

@dataclass
class SingleTextClassificationSample(BaseSample):
    text: str = ""
    label: int = -1

@dataclass
class SingleTextDistributionSample(SingleTextClassificationSample):
    human_dist: List[float] = field(default_factory=list)
    annotation_labels: List[int] = field(default_factory=list)

@dataclass
class MultilevelSample(BaseSample):
    premise: str = ""
    hypothesis: str = ""
    hard_labels: Dict[str, int] = field(default_factory=dict)
    human_dists: Dict[str, List[float]] = field(default_factory=dict)
