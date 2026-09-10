"""Deterministic pseudo-random choices for ambiguous hard-label projections."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence


def tied_argmax(values: Sequence[float], identifier: str, namespace: str = "") -> int:
    """Choose uniformly among maximum values using a stable hash of the sample ID."""
    maximum = max(values)
    tied = [index for index, value in enumerate(values) if value == maximum]
    if len(tied) == 1:
        return tied[0]
    digest = hashlib.sha256(f"{namespace}\x1f{identifier}".encode("utf-8")).digest()
    return tied[int.from_bytes(digest[:8], "big") % len(tied)]


def binary_threshold(value: float, identifier: str, namespace: str = "") -> int:
    """Threshold at 0.5, choosing reproducibly at an exactly ambiguous 0.5."""
    if value != 0.5:
        return int(value > 0.5)
    return tied_argmax([0.5, 0.5], identifier, namespace)
