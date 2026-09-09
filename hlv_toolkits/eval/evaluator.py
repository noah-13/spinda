"""
Evaluator for computing calibration metrics on NLI predictions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from hlv_toolkits.data.schemas import (
    SingleTextDistributionSample,
    TextPairClassificationSample,
    TextPairDistributionSample,
    NLI_NUM_LABELS,
    PredictionRecord,
)
from hlv_toolkits.eval.metrics import (
    compute_tvd,
    compute_distance_correlation,
    compute_euclidean_distance,
    compute_jsd,
    compute_pojsd,
    compute_kl,
    compute_cross_entropy,
    compute_soft_macro_f1,
    compute_soft_micro_f1,
)


@dataclass
class EvaluationArtifacts:
    metrics: Dict[str, float]
    pred_probs: Optional[np.ndarray] = None
    human_probs: Optional[np.ndarray] = None
    pred_labels: Optional[np.ndarray] = None
    true_labels: Optional[np.ndarray] = None


class Evaluator:
    """
    Evaluator for computing calibration metrics.

    Supports both single-label and distribution-based evaluation.
    """

    def __init__(self) -> None:
        pass

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

            if pred_label < 0 or num_labels <= 0 or len(probs) != num_labels:
                continue
            if len(gt.human_dist) != num_labels:
                continue

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

            if pred_label < 0 or not isinstance(probs, list) or len(probs) == 0:
                continue
            if pred_label >= len(probs) or gt.label < 0 or gt.label >= len(probs):
                continue

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
            metrics={"accuracy": float(accuracy)},
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

        accuracy = (pred_labels == true_labels).mean()
        tvd = compute_tvd(pred_probs, human_probs)
        jsd = compute_jsd(pred_probs, human_probs, base=2)
        pojsd = compute_pojsd(pred_probs, human_probs)
        kl = compute_kl(human_probs, pred_probs)
        soft_micro_f1 = compute_soft_micro_f1(pred_probs, human_probs)
        soft_macro_f1 = compute_soft_macro_f1(pred_probs, human_probs)
        distance_correlation = compute_distance_correlation(pred_probs, human_probs)
        euclidean_distance = compute_euclidean_distance(pred_probs, human_probs)
        cross_entropy = compute_cross_entropy(human_probs, pred_probs)

        return EvaluationArtifacts(
            metrics={
                "accuracy": float(accuracy),
                "tvd": float(np.mean(tvd)),
                "jsd": float(np.mean(jsd)),
                "pojsd": float(np.mean(pojsd)),
                "kl": float(np.mean(kl)),
                "soft_micro_f1": float(soft_micro_f1),
                "soft_macro_f1": float(soft_macro_f1),
                "distance_correlation": float(distance_correlation),
                # Keep result keys aligned with notebooks/metrics_tutorial.ipynb.
                "l2": float(np.mean(euclidean_distance)),
                "ce": float(np.mean(cross_entropy)),
            },
            pred_probs=pred_probs,
            human_probs=human_probs,
            pred_labels=pred_labels,
            true_labels=true_labels,
        )

    def evaluate(
        self,
        predictions: List[PredictionRecord],
        ground_truth: List[Union[TextPairClassificationSample, TextPairDistributionSample]],
    ) -> EvaluationArtifacts:
        if len(ground_truth) == 0:
            return EvaluationArtifacts(metrics={})

        if isinstance(ground_truth[0], (TextPairDistributionSample, SingleTextDistributionSample)):
            return self.evaluate_with_distribution(
                predictions,
                [gt for gt in ground_truth if isinstance(gt, (TextPairDistributionSample, SingleTextDistributionSample))],
            )

        return self.evaluate_single_label(
            predictions,
            [gt for gt in ground_truth if isinstance(gt, (TextPairClassificationSample, TextPairDistributionSample, SingleTextDistributionSample))],
        )
