# data/schemas.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Union

# -------------------------
# Common types
# -------------------------
Split = Literal["train", "dev", "valid", "validation", "test", "unknown"]

# =========================
# Shared sample metadata
# =========================
@dataclass
class BaseSample:
    """Metadata shared by all normalized dataset rows.

    Concrete subclasses hold task-specific text and numeric labels. Label names,
    ordering, and hierarchy belong to the dataset-level ``dataset.json``
    manifest, rather than to individual samples or this module.
    """
    id: str
    task: str  # Reader/data format identifier used in prediction and evaluation output.
    split: Split = "unknown"
    source: Optional[str] = None  # Dataset directory or source provenance.
    meta: Dict[str, Any] = field(default_factory=dict)


# =========================
# Prediction record
# =========================
@dataclass
class PredictionRecord:
    """Standardized prediction dump aligned to a normalized sample.

    Label indices and probability-vector positions in ``outputs`` follow the
    corresponding dataset manifest.
    """
    id: str
    task: str
    split: Split = "unknown"
    source: Optional[str] = None

    outputs: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


# Normalized dataset rows. Numeric label values use the ordering declared in
# the dataset manifest.
@dataclass
class TextPairClassificationSample(BaseSample):
    """A text pair with one categorical label index."""
    text_a: str = ""
    text_b: str = ""
    label: int = -1

@dataclass
class TextPairDistributionSample(TextPairClassificationSample):
    """A text pair with an empirical categorical label distribution."""
    human_dist: List[float] = field(default_factory=list)
    annotation_labels: List[int] = field(default_factory=list)

@dataclass
class SingleTextClassificationSample(BaseSample):
    """One text with one categorical label index."""
    text: str = ""
    label: int = -1

@dataclass
class SingleTextDistributionSample(SingleTextClassificationSample):
    """One text with an empirical categorical label distribution."""
    human_dist: List[float] = field(default_factory=list)
    annotation_labels: List[int] = field(default_factory=list)


@dataclass
class SingleTextMultilabelDistributionSample(BaseSample):
    """One text with a binary probability for each manifest label."""
    text: str = ""
    labels: List[int] = field(default_factory=list)
    human_probs: List[float] = field(default_factory=list)
    annotation_label_sets: List[List[int]] = field(default_factory=list)
@dataclass
class MultilevelSample(BaseSample):
    """A text pair with one categorical distribution per annotation dimension."""
    text_a: str = ""
    text_b: str = ""
    hard_labels: Dict[str, int] = field(default_factory=dict)
    human_dists: Dict[str, List[float]] = field(default_factory=dict)
    annotation_labels: Dict[str, List[int]] = field(default_factory=dict)

@dataclass
class SingleTextMultilevelSample(BaseSample):
    """One text with one categorical distribution per annotation dimension."""
    text: str = ""
    hard_labels: Dict[str, int] = field(default_factory=dict)
    human_dists: Dict[str, List[float]] = field(default_factory=dict)
    annotation_labels: Dict[str, List[int]] = field(default_factory=dict)


AnySample = Union[
    TextPairClassificationSample,
    SingleTextMultilevelSample,
    TextPairDistributionSample,
    SingleTextClassificationSample,
    SingleTextDistributionSample,
    SingleTextMultilabelDistributionSample,
    MultilevelSample,
]
