"""
Evaluator for computing calibration metrics on NLI predictions.
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np

from nli_toolkits.data.schemas import (
    NLIDistributionSample,
    NLISample,
    NLI_NUM_LABELS,
    PredictionRecord,
)
from nli_toolkits.eval.metrics import (
    compute_distce,   
)


class Evaluator:
    """
    Evaluator for computing calibration metrics.
    
    Supports both single-label evaluation (using ECE) and
    distribution-based evaluation (using EntCE, RankCS, DistCE).
    """

    def __init__(self) -> None:
        pass

    def evaluate_single_label(
        self,
        predictions: List[PredictionRecord],
        ground_truth: List[NLISample],
    ) -> Dict[str, float]:
        """
        Evaluate predictions against single ground-truth labels.
        
        Computes:
        - Accuracy
        - ECE (Expected Calibration Error)
        
        Args:
            predictions: List of prediction records
            ground_truth: List of ground truth samples
            
        Returns:
            Dictionary of metric names to values
        """
        # Match predictions to ground truth by ID
        gt_dict = {gt.id: gt for gt in ground_truth}
        
        pred_labels = []
        confidences = []
        true_labels = []
        
        for pred in predictions:
            if pred.id not in gt_dict:
                continue
            
            gt = gt_dict[pred.id]
            pred_label = pred.outputs.get("pred", -1)
            probs = pred.outputs.get("probs", [])
            
            if pred_label < 0 or len(probs) != NLI_NUM_LABELS:
                continue
            
            pred_labels.append(pred_label)
            confidences.append(max(probs))
            true_labels.append(gt.label)
        
        if len(pred_labels) == 0:
            return {"accuracy": 0.0, "ece": 0.0}
        
        pred_labels = np.array(pred_labels)
        confidences = np.array(confidences)
        true_labels = np.array(true_labels)
        
        # Compute accuracy
        accuracy = (pred_labels == true_labels).mean()
        
        # Compute ECE
        ece = compute_ece(pred_labels, confidences, true_labels)
        
        return {
            "accuracy": float(accuracy),
            "ece": float(ece),
        }

    def evaluate_with_distribution(
        self,
        predictions: List[PredictionRecord],
        ground_truth: List[NLIDistributionSample],
    ) -> Dict[str, float]:
        """
        Evaluate predictions against human annotation distributions.
        
        Computes:
        - Accuracy (majority vote)
        - ECE (Expected Calibration Error)
        - EntCE (Human Entropy Calibration Error) - mean and median
        - RankCS (Human Ranking Calibration Score)
        - DistCE (Human Distribution Calibration Error) - mean and median
        
        Args:
            predictions: List of prediction records
            ground_truth: List of ground truth samples with human distributions
            
        Returns:
            Dictionary of metric names to values
        """
        # Match predictions to ground truth by ID
        gt_dict = {gt.id: gt for gt in ground_truth}
        
        pred_probs_list = []
        human_probs_list = []
        pred_labels = []
        true_labels = []
        
        for pred in predictions:
            if pred.id not in gt_dict:
                continue
            
            gt = gt_dict[pred.id]
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
            return {
                "accuracy": 0.0,
                # "ece": 0.0,
                # "entce_mean": 0.0,
                # "entce_median": 0.0,
                # "rankcs": 0.0,
                "distce_mean": 0.0,
                # "distce_median": 0.0,
            }
        
        pred_probs = np.array(pred_probs_list)
        human_probs = np.array(human_probs_list)
        pred_labels = np.array(pred_labels)
        true_labels = np.array(true_labels)
        # confidences = pred_probs.max(axis=1)
        
        # Compute accuracy
        accuracy = (pred_labels == true_labels).mean()
        
        # Compute ECE
        # ece = compute_ece(pred_labels, confidences, true_labels)
        
        # Compute EntCE
        # entce = compute_entce(pred_probs, human_probs)
        # entce_mean = float(np.mean(entce))
        # entce_median = float(np.median(entce))
        
        # Compute RankCS
        # rankcs = compute_rankcs(pred_probs, human_probs)
        
        # Compute DistCE
        distce = compute_distce(pred_probs, human_probs)
        distce_mean = float(np.mean(distce))
        # distce_median = float(np.median(distce))
        
        return {
            "accuracy": float(accuracy),
            # "ece": float(ece),
            # "entce_mean": entce_mean,
            # "entce_median": entce_median,
            # "rankcs": float(rankcs),
            "distce_mean": distce_mean,
            # "distce_median": distce_median,
        }

    def evaluate(
        self,
        predictions: List[PredictionRecord],
        ground_truth: List[Union[NLISample, NLIDistributionSample]],
    ) -> Dict[str, float]:
        """
        Evaluate predictions. Automatically detects evaluation type.
        
        If ground_truth contains NLIDistributionSample, uses distribution-based evaluation.
        Otherwise, uses single-label evaluation.
        
        Args:
            predictions: List of prediction records
            ground_truth: List of ground truth samples
            
        Returns:
            Dictionary of metric names to values
        """
        if len(ground_truth) == 0:
            return {}
        
        # Check if we have distribution samples
        if isinstance(ground_truth[0], NLIDistributionSample):
            return self.evaluate_with_distribution(
                predictions, [gt for gt in ground_truth if isinstance(gt, NLIDistributionSample)]
            )
        else:
            return self.evaluate_single_label(
                predictions, [gt for gt in ground_truth if isinstance(gt, NLISample)]
            )
