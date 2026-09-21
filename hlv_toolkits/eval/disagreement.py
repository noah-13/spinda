"""Disagreement-stratified and per-instance HLV evaluation utilities."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from hlv_toolkits.eval.metrics import compute_cross_entropy, compute_euclidean_distance, compute_jsd, compute_kl, compute_tvd, validate_and_fix_probs

INSTANCE_ERROR_METRICS = ("tvd", "jsd", "kl", "ce", "l2")


def compute_human_entropy(human_probs: np.ndarray, *, normalized: bool = True) -> np.ndarray:
    """Return per-example Shannon entropy in normalized bits (0=unanimous, 1=uniform)."""
    human_probs = validate_and_fix_probs(human_probs, name="human_probs")
    num_classes = human_probs.shape[1]
    if num_classes < 2:
        raise ValueError("Human disagreement requires at least two classes.")
    log_probs = np.zeros_like(human_probs)
    positive = human_probs > 0
    log_probs[positive] = np.log2(human_probs[positive])
    entropy = -np.sum(human_probs * log_probs, axis=1)
    return entropy / np.log2(num_classes) if normalized else entropy


def compute_instance_distribution_errors(pred_probs: np.ndarray, human_probs: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute every supported lower-is-better distribution error per example."""
    pred_probs = validate_and_fix_probs(pred_probs, name="pred_probs")
    human_probs = validate_and_fix_probs(human_probs, name="human_probs")
    if pred_probs.shape != human_probs.shape:

        raise ValueError("pred_probs and human_probs must have the same shape")
    return {"tvd": compute_tvd(pred_probs, human_probs), "jsd": compute_jsd(pred_probs, human_probs, base=2), "kl": compute_kl(human_probs, pred_probs), "ce": compute_cross_entropy(human_probs, pred_probs), "l2": compute_euclidean_distance(pred_probs, human_probs)}

def assign_disagreement_strata(
    entropy: np.ndarray,
    *,
    num_groups: int = 3,
    boundaries: Optional[Sequence[float]] = None,
) -> tuple[np.ndarray, Dict[str, object]]:
    """Assign entropy values to configurable disagreement strata.

    With no ``boundaries``, empirical quantiles create equally sized groups.
    Explicit boundaries are normalized entropy cutoffs in [0, 1]. Ties stay
    in the lower stratum, so equal human distributions are never separated.
    """
    entropy = np.asarray(entropy, dtype=float)
    if entropy.ndim != 1 or entropy.size == 0 or not np.isfinite(entropy).all():
        raise ValueError("entropy must be a non-empty finite one-dimensional array")
    if num_groups < 2:
        raise ValueError("num_groups must be at least 2")
    if boundaries is None:
        cutoffs = np.quantile(entropy, np.arange(1, num_groups) / num_groups)
        method = "empirical_quantiles"
    else:
        cutoffs = np.asarray(boundaries, dtype=float)
        if cutoffs.ndim != 1 or len(cutoffs) != num_groups - 1:
            raise ValueError("boundaries must contain exactly num_groups - 1 values")
        if not np.isfinite(cutoffs).all() or np.any(cutoffs < 0) or np.any(cutoffs > 1) or np.any(np.diff(cutoffs) <= 0):
            raise ValueError("boundaries must be strictly increasing finite values in [0, 1]")
        method = "explicit_boundaries"

    names = ("low", "medium", "high") if num_groups == 3 else tuple(f"group_{index}" for index in range(1, num_groups + 1))
    indices = np.searchsorted(cutoffs, entropy, side="left")
    strata = np.asarray([names[index] for index in indices], dtype=f"U{max(map(len, names))}")
    metadata: Dict[str, object] = {
        "method": method,
        "num_groups": num_groups,
        "boundaries": [float(value) for value in cutoffs],
        "group_names": list(names),
    }
    if num_groups == 3:
        metadata.update({"low_max_entropy": float(cutoffs[0]), "medium_max_entropy": float(cutoffs[1])})
    return strata, metadata

def _describe(values: np.ndarray) -> Dict[str, float]:
    return {"mean": float(np.mean(values)), "median": float(np.median(values)), "p25": float(np.quantile(values, .25)), "p75": float(np.quantile(values, .75)), "p90": float(np.quantile(values, .9)), "max": float(np.max(values))}


def disagreement_stratified_evaluation(
    pred_probs: np.ndarray, human_probs: np.ndarray, pred_labels: np.ndarray, true_labels: np.ndarray,
    *, num_groups: int = 3, boundaries: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Report accuracy and distribution errors separately by human entropy."""
    entropy = compute_human_entropy(human_probs)
    strata, thresholds = assign_disagreement_strata(entropy, num_groups=num_groups, boundaries=boundaries)
    errors = compute_instance_distribution_errors(pred_probs, human_probs)
    pred_labels, true_labels = np.asarray(pred_labels), np.asarray(true_labels)
    if len(pred_labels) != len(entropy) or len(true_labels) != len(entropy):
        raise ValueError("labels must have one entry per distribution")
    groups: Dict[str, Dict[str, object]] = {}
    for name in thresholds["group_names"]:
        mask = strata == name
        group: Dict[str, object] = {"n": int(mask.sum())}
        if mask.any():
            group["entropy"] = _describe(entropy[mask])
            group["metrics"] = {"accuracy": float(np.mean(pred_labels[mask] == true_labels[mask])), **{metric: float(np.mean(values[mask])) for metric, values in errors.items()}}
        else:
            group["entropy"], group["metrics"] = None, None
        groups[name] = group
    return {"stratification": "normalized_human_entropy", "entropy_unit": "normalized_bits", "thresholds": thresholds, "groups": groups}


def instance_error_records(
    ids: Sequence[str], pred_probs: np.ndarray, human_probs: np.ndarray,
    pred_labels: np.ndarray, true_labels: np.ndarray,
    *, num_groups: int = 3, boundaries: Optional[Sequence[float]] = None,
) -> list[Dict[str, object]]:
    """Return one sortable, serializable error record for every example."""
    entropy = compute_human_entropy(human_probs)
    if len(ids) != len(entropy):
        raise ValueError("ids must have one entry per distribution")
    strata, _ = assign_disagreement_strata(entropy, num_groups=num_groups, boundaries=boundaries)
    errors = compute_instance_distribution_errors(pred_probs, human_probs)
    pred_labels, true_labels = np.asarray(pred_labels), np.asarray(true_labels)
    return [
        {
            "id": str(sample_id), "human_entropy": float(entropy[index]),
            "disagreement_group": str(strata[index]), "pred_label": int(pred_labels[index]),
            "true_label": int(true_labels[index]), "is_correct": bool(pred_labels[index] == true_labels[index]),
            **{metric: float(values[index]) for metric, values in errors.items()},
        }
        for index, sample_id in enumerate(ids)
    ]

def instance_error_summary(errors: np.ndarray, *, metric: str) -> Dict[str, object]:
    """Summarize an error distribution, including its upper tail."""
    if metric not in INSTANCE_ERROR_METRICS:
        raise ValueError(f"Unsupported instance error metric: {metric}")
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 1 or errors.size == 0 or not np.isfinite(errors).all():
        raise ValueError("errors must be a non-empty finite one-dimensional array")
    return {"metric": metric, "n": int(errors.size), **_describe(errors)}


def analyze_distributional_disagreement(
    pred_probs: np.ndarray, human_probs: np.ndarray, pred_labels: np.ndarray, true_labels: np.ndarray,
    *, num_groups: int = 3, boundaries: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Return the complete stratified report plus summaries for every error."""
    errors = compute_instance_distribution_errors(pred_probs, human_probs)
    return {
        "disagreement_stratified": disagreement_stratified_evaluation(pred_probs, human_probs, pred_labels, true_labels, num_groups=num_groups, boundaries=boundaries),
        "instance_error_distributions": {
            metric: instance_error_summary(values, metric=metric) for metric, values in errors.items()
        },
    }
