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
class NLISample(BaseSample):
    premise: str = ""
    hypothesis: str = ""
    label: int = -1  # 0..NLI_NUM_LABELS-1


@dataclass
class NLIDistributionSample(NLISample):
    # Empirical human label distribution, length == NLI_NUM_LABELS, sums to ~1.0
    human_dist: List[float] = field(default_factory=list)


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
    NLISample,
    NLIDistributionSample,
]
