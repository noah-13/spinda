"""
Evaluator for computing calibration metrics on NLI predictions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from nli_toolkits.data.schemas import (
    NLIDistributionSample,
    NLISample,
    NLI_NUM_LABELS,
    PredictionRecord,
)
from nli_toolkits.eval.metrics import (
    compute_tvd,
    compute_distance_correlation,
    compute_jsd,
    compute_kl_human_to_pred,
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
        ground_truth: List[NLIDistributionSample],
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

            if pred_label < 0 or len(probs) != NLI_NUM_LABELS:
                continue
            if len(gt.human_dist) != NLI_NUM_LABELS:
                continue

            pred_probs_list.append(probs)
            human_probs_list.append(gt.human_dist)
            pred_labels.append(pred_label)
            true_labels.append(gt.label)

        if len(pred_probs_list) == 0:
            empty_probs = np.empty((0, NLI_NUM_LABELS), dtype=float)
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
        ground_truth: List[NLISample],
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

            if pred_label < 0 or len(probs) != NLI_NUM_LABELS:
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
        ground_truth: List[NLIDistributionSample],
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
        kl = compute_kl_human_to_pred(pred_probs, human_probs)
        soft_micro_f1 = compute_soft_micro_f1(pred_probs, human_probs)
        soft_macro_f1 = compute_soft_macro_f1(pred_probs, human_probs)
        distance_correlation = compute_distance_correlation(pred_probs, human_probs)

        return EvaluationArtifacts(
            metrics={
                "accuracy": float(accuracy),
                "tvd_mean": float(np.mean(tvd)),
                "jsd_mean": float(np.mean(jsd)),
                "kl_mean": float(np.mean(kl)),
                "soft_micro_f1": float(soft_micro_f1),
                "soft_macro_f1": float(soft_macro_f1),
                "distance_correlation": float(distance_correlation),
            },
            pred_probs=pred_probs,
            human_probs=human_probs,
            pred_labels=pred_labels,
            true_labels=true_labels,
        )

    def evaluate(
        self,
        predictions: List[PredictionRecord],
        ground_truth: List[Union[NLISample, NLIDistributionSample]],
    ) -> EvaluationArtifacts:
        if len(ground_truth) == 0:
            return EvaluationArtifacts(metrics={})

        if isinstance(ground_truth[0], NLIDistributionSample):
            return self.evaluate_with_distribution(
                predictions,
                [gt for gt in ground_truth if isinstance(gt, NLIDistributionSample)],
            )

        return self.evaluate_single_label(
            predictions,
            [gt for gt in ground_truth if isinstance(gt, NLISample)],
        )
