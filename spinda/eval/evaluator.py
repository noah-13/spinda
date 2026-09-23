"""
Evaluator for computing calibration metrics on NLI predictions.
"""

from dataclasses import dataclass
from typing import Collection, Dict, List, Optional, Tuple, Union

import numpy as np

from spinda.data.schemas import (
    SingleTextDistributionSample,
    SingleTextMultilabelDistributionSample,
    TextPairClassificationSample,
    TextPairDistributionSample,
    PredictionRecord,
)
from spinda.eval.metrics import (
    compute_tvd,
    validate_and_fix_probs,
    compute_distance_correlation,
    compute_entropy_correlation,
    compute_euclidean_distance,
    compute_jsd,
    compute_pojsd,
    compute_kl,
    compute_cross_entropy,
    compute_soft_macro_f1,
    compute_soft_micro_f1,
    compute_multilabel_entropy_correlation,
    compute_multilabel_pojsd,
)


@dataclass
class EvaluationArtifacts:
    metrics: Dict[str, float]
    pred_probs: Optional[np.ndarray] = None
    human_probs: Optional[np.ndarray] = None
    pred_labels: Optional[np.ndarray] = None
    true_labels: Optional[np.ndarray] = None


DEFAULT_CATEGORICAL_DISTRIBUTION_METRICS = (
    "accuracy",
    "tvd",
    "jsd",
    "kl",
    "ce",
    "l2",
    "entropy_correlation",
    "distance_correlation",
)
"""The standard metrics reported for normalized, categorical distributions."""


_CATEGORICAL_DISTRIBUTION_METRICS = frozenset(
    {
        *DEFAULT_CATEGORICAL_DISTRIBUTION_METRICS,
        # Backward-compatible, mathematically derived metrics; opt-in only.
        "macro_f1",
        "l1",
        "pojsd",
        "soft_micro_f1",
        "soft_accuracy",
        "soft_macro_f1",
    }
)


def _macro_f1(predicted, targets, num_labels):
    scores = []
    for label in range(num_labels):
        pred, gold = predicted == label, targets == label
        denominator = pred.sum() + gold.sum()
        scores.append(float(2 * np.logical_and(pred, gold).sum() / denominator) if denominator else 0.0)
    return float(np.mean(scores))


class Evaluator:
    """
    Evaluator for computing calibration metrics.

    Supports both single-label and distribution-based evaluation.
    """

    def __init__(
        self,
        distribution_metrics: Optional[Collection[str]] = None,
    ) -> None:
        """Create an evaluator with selected categorical distribution metrics.

        The default is the standard seven-metric table. Derived and legacy
        names such as ``l1``, ``pojsd``, and ``soft_accuracy`` are opt-in.
        """
        requested_metrics = (
            DEFAULT_CATEGORICAL_DISTRIBUTION_METRICS
            if distribution_metrics is None
            else tuple(distribution_metrics)
        )
        unknown_metrics = set(requested_metrics) - _CATEGORICAL_DISTRIBUTION_METRICS
        if unknown_metrics:
            raise ValueError(
                "Unsupported categorical distribution metrics: "
                + ", ".join(sorted(unknown_metrics))
            )
        self.distribution_metrics = requested_metrics



    @staticmethod
    def validate_prediction_coverage(predictions: List[PredictionRecord], ground_truth: List[object]) -> None:
        """Require a one-to-one mapping between prediction and gold IDs."""
        prediction_ids = [prediction.id for prediction in predictions]
        ground_truth_ids = [getattr(sample, "id") for sample in ground_truth]
        if len(set(prediction_ids)) != len(prediction_ids):
            raise ValueError("Predictions contain duplicate IDs.")
        if len(set(ground_truth_ids)) != len(ground_truth_ids):
            raise ValueError("Ground truth contains duplicate IDs.")
        unknown_ids = set(prediction_ids) - set(ground_truth_ids)
        missing_ids = set(ground_truth_ids) - set(prediction_ids)
        if unknown_ids or missing_ids:
            details = []
            if unknown_ids:
                details.append(f"unknown prediction IDs: {sorted(unknown_ids)[:3]}")
            if missing_ids:
                details.append(f"missing prediction IDs: {sorted(missing_ids)[:3]}")
            raise ValueError("Prediction/ground-truth ID mismatch (" + "; ".join(details) + ").")

    def prepare_distribution_arrays(
        self,
        predictions: List[PredictionRecord],
        ground_truth: List[Union[TextPairDistributionSample]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Align predictions and distribution ground truth by ID and return arrays.

        Returns:
            pred_probs: [N, num_labels]
            human_probs: [N, num_labels]
            pred_labels: [N]
            true_labels: [N]
        """
        gt_dict = {gt.id: gt for gt in ground_truth}
        num_labels = len(ground_truth[0].human_dist) if ground_truth and ground_truth[0].human_dist else 0

        pred_probs_list = []
        human_probs_list = []
        pred_labels = []
        true_labels = []

        for pred in predictions:
            gt = gt_dict.get(pred.id)
            if gt is None:
                continue

            pred_label = pred.outputs.get("pred", -1)
            probs = pred.outputs.get("probs", [])

            if not isinstance(pred_label, int) or isinstance(pred_label, bool) or not isinstance(probs, list):
                raise ValueError(f"Prediction {pred.id} must contain integer pred and list probs.")
            if pred_label < 0 or pred_label >= num_labels or num_labels <= 0 or len(probs) != num_labels:
                raise ValueError(f"Prediction {pred.id} has an invalid categorical output.")
            if len(gt.human_dist) != num_labels:
                raise ValueError(f"Ground truth {gt.id} has an invalid distribution length.")

            pred_probs_list.append(probs)
            human_probs_list.append(gt.human_dist)
            pred_labels.append(pred_label)
            true_labels.append(gt.label)

        if len(pred_probs_list) == 0:
            empty_probs = np.empty((0, num_labels), dtype=float)
            empty_labels = np.empty((0,), dtype=int)
            return empty_probs, empty_probs.copy(), empty_labels, empty_labels.copy()

        pred_probs = np.array(pred_probs_list, dtype=float)
        human_probs = np.array(human_probs_list, dtype=float)
        pred_probs = validate_and_fix_probs(pred_probs, name="pred_probs")
        human_probs = validate_and_fix_probs(human_probs, name="human_probs")
        pred_labels_arr = np.array(pred_labels, dtype=int)
        true_labels_arr = np.array(true_labels, dtype=int)
        return pred_probs, human_probs, pred_labels_arr, true_labels_arr

    def evaluate_single_label(
        self,
        predictions: List[PredictionRecord],
        ground_truth: List[Union[TextPairClassificationSample]],
    ) -> EvaluationArtifacts:
        # Match predictions to ground truth by ID
        gt_dict = {gt.id: gt for gt in ground_truth}

        pred_labels = []
        true_labels = []

        for pred in predictions:
            gt = gt_dict.get(pred.id)
            if gt is None:
                continue

            pred_label = pred.outputs.get("pred", -1)
            probs = pred.outputs.get("probs", [])

            if not isinstance(pred_label, int) or isinstance(pred_label, bool) or not isinstance(probs, list):
                raise ValueError(f"Prediction {pred.id} must contain integer pred and list probs.")
            if pred_label < 0 or len(probs) == 0:
                raise ValueError(f"Prediction {pred.id} has an invalid categorical output.")
            if pred_label >= len(probs) or gt.label < 0 or gt.label >= len(probs):
                raise ValueError(f"Prediction {pred.id} or ground truth {gt.id} has an out-of-range label.")

            pred_labels.append(pred_label)
            true_labels.append(gt.label)

        if len(pred_labels) == 0:
            empty_labels = np.empty((0,), dtype=int)
            return EvaluationArtifacts(
                metrics={"accuracy": 0.0},
                pred_labels=empty_labels,
                true_labels=empty_labels.copy(),
            )

        pred_labels_arr = np.array(pred_labels, dtype=int)
        true_labels_arr = np.array(true_labels, dtype=int)

        accuracy = (pred_labels_arr == true_labels_arr).mean()

        return EvaluationArtifacts(
            metrics={"accuracy": float(accuracy), "macro_f1": _macro_f1(pred_labels_arr, true_labels_arr, len(predictions[0].outputs["probs"]))},
            pred_labels=pred_labels_arr,
            true_labels=true_labels_arr,
        )

    def evaluate_with_distribution(
        self,
        predictions: List[PredictionRecord],
        ground_truth: List[Union[TextPairDistributionSample]],
    ) -> EvaluationArtifacts:
        pred_probs, human_probs, pred_labels, true_labels = self.prepare_distribution_arrays(
            predictions,
            ground_truth,
        )
        if len(pred_probs) == 0:
            raise ValueError("No valid predictions to evaluate.")

        functions = {
            "accuracy": lambda: float((pred_labels == true_labels).mean()),
            "macro_f1": lambda: _macro_f1(pred_labels, true_labels, pred_probs.shape[1]),
            "tvd": lambda: float(np.mean(compute_tvd(pred_probs, human_probs))),
            "l1": lambda: 2 * float(np.mean(compute_tvd(pred_probs, human_probs))),
            "jsd": lambda: float(np.mean(compute_jsd(pred_probs, human_probs, base=2))),
            "pojsd": lambda: float(np.mean(compute_pojsd(pred_probs, human_probs))),
            "kl": lambda: float(np.mean(compute_kl(human_probs, pred_probs))),
            "soft_micro_f1": lambda: float(compute_soft_micro_f1(pred_probs, human_probs)),
            "soft_accuracy": lambda: float(compute_soft_micro_f1(pred_probs, human_probs)),
            "soft_macro_f1": lambda: float(compute_soft_macro_f1(pred_probs, human_probs)),
            "distance_correlation": lambda: compute_distance_correlation(pred_probs, human_probs),
            "entropy_correlation": lambda: compute_entropy_correlation(pred_probs, human_probs),
            "l2": lambda: float(np.mean(compute_euclidean_distance(pred_probs, human_probs))),
            "ce": lambda: float(np.mean(compute_cross_entropy(human_probs, pred_probs))),
        }
        all_metrics = {name: functions[name]() for name in self.distribution_metrics}

        return EvaluationArtifacts(
            metrics={name: all_metrics[name] for name in self.distribution_metrics},
            pred_probs=pred_probs,
            human_probs=human_probs,
            pred_labels=pred_labels,
            true_labels=true_labels,
        )

    def evaluate_multilabel(
        self, predictions: List[PredictionRecord], ground_truth: List[SingleTextMultilabelDistributionSample]
    ) -> EvaluationArtifacts:
        """Evaluate independent per-label probabilities and binary decisions."""
        pred_probs, human_probs, pred_labels, true_labels = [], [], [], []
        ground_truth_by_id = {sample.id: sample for sample in ground_truth}
        for prediction in predictions:
            sample = ground_truth_by_id[prediction.id]
            probs, labels = prediction.outputs.get("probs"), prediction.outputs.get("pred")
            width = len(sample.human_probs)
            if not isinstance(probs, list) or not isinstance(labels, list) or len(probs) != width or len(labels) != width:
                raise ValueError(f"Prediction {prediction.id} has an invalid multilabel output.")
            if any(not isinstance(label, int) or isinstance(label, bool) or label not in {0, 1} for label in labels):
                raise ValueError(f"Prediction {prediction.id} must use binary integer multilabel predictions.")
            pred_probs.append(probs)
            human_probs.append(sample.human_probs)
            pred_labels.append(labels)
            true_labels.append(sample.labels)

        pred_probs_array = np.asarray(pred_probs, dtype=float)
        human_probs_array = np.asarray(human_probs, dtype=float)
        if not np.isfinite(pred_probs_array).all() or np.any((pred_probs_array < 0) | (pred_probs_array > 1)):
            raise ValueError("Multilabel probabilities must be finite values in [0, 1].")
        if not np.isfinite(human_probs_array).all() or np.any((human_probs_array < 0) | (human_probs_array > 1)):
            raise ValueError("Ground-truth multilabel probabilities must be finite values in [0, 1].")
        pred_labels_array = np.asarray(pred_labels, dtype=int)
        true_labels_array = np.asarray(true_labels, dtype=int)
        denominator = pred_labels_array.sum() + true_labels_array.sum()
        micro_f1 = float(2 * np.logical_and(pred_labels_array, true_labels_array).sum() / denominator) if denominator else 1.0
        macro_denominators = pred_labels_array.sum(axis=0) + true_labels_array.sum(axis=0)
        macro_f1 = float(np.mean(np.divide(
            2 * np.logical_and(pred_labels_array, true_labels_array).sum(axis=0),
            macro_denominators,
            out=np.zeros_like(macro_denominators, dtype=float),
            where=macro_denominators != 0,
        )))
        return EvaluationArtifacts(
            metrics={
                "accuracy": float(np.all(pred_labels_array == true_labels_array, axis=-1).mean()),
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
                "soft_micro_f1": float(compute_soft_micro_f1(pred_probs_array, human_probs_array)),
                "soft_macro_f1": float(compute_soft_macro_f1(pred_probs_array, human_probs_array)),
                "multilabel_pojsd": compute_multilabel_pojsd(pred_probs_array, human_probs_array),
                "multilabel_entropy_correlation": compute_multilabel_entropy_correlation(pred_probs_array, human_probs_array),
            },
            pred_probs=pred_probs_array, human_probs=human_probs_array,
            pred_labels=pred_labels_array, true_labels=true_labels_array,
        )

    def evaluate(
        self,
        predictions: List[PredictionRecord],
        ground_truth: List[Union[TextPairClassificationSample, TextPairDistributionSample]],
    ) -> EvaluationArtifacts:
        if len(ground_truth) == 0:
            return EvaluationArtifacts(metrics={})

        self.validate_prediction_coverage(predictions, ground_truth)

        if isinstance(ground_truth[0], SingleTextMultilabelDistributionSample):
            multilabel_ground_truth = [sample for sample in ground_truth if isinstance(sample, SingleTextMultilabelDistributionSample)]
            if len(multilabel_ground_truth) != len(ground_truth):
                raise ValueError("Ground truth cannot mix multilabel and categorical samples.")
            return self.evaluate_multilabel(predictions, multilabel_ground_truth)

        if isinstance(ground_truth[0], (TextPairDistributionSample, SingleTextDistributionSample)):
            return self.evaluate_with_distribution(
                predictions,
                [gt for gt in ground_truth if isinstance(gt, (TextPairDistributionSample, SingleTextDistributionSample))],
            )

        return self.evaluate_single_label(
            predictions,
            [gt for gt in ground_truth if isinstance(gt, (TextPairClassificationSample, TextPairDistributionSample, SingleTextDistributionSample))],
        )
