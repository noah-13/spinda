#!/usr/bin/env python3
"""
Evaluation script for NLI predictions.

Example usage:
    python -m nli_toolkits.scripts.evaluate \
        --predictions predictions.jsonl \
        --ground_truth_source snli_test.jsonl \
        --output_file results.json
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from nli_toolkits.data import (
    ChaosNLIReader,
    NLIDistributionSample,
    NLISample,
    SNLIReader,
)
from nli_toolkits.data.schemas import PredictionRecord
from nli_toolkits.data.schemas import NLI_NUM_LABELS
from nli_toolkits.eval import Evaluator
from nli_toolkits.eval.metrics import compute_distce


def load_predictions(file_path: Path) -> List[PredictionRecord]:
    """Load predictions from JSONL file."""
    predictions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            pred = PredictionRecord(
                id=data["id"],
                task=data.get("task", "nli"),
                split=data.get("split", "unknown"),
                source=data.get("source"),
                outputs=data["outputs"],
            )
            predictions.append(pred)
    return predictions


def save_distce_plot_sns(
    distce: np.ndarray,
    output_path: Path,
    title: str | None = None,
    bins: int = 30,
) -> bool:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return False

    values = distce.astype(float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with plt.style.context("seaborn-v0_8-darkgrid"):
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    except Exception:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

    sns.histplot(
    values,
    binwidth=1 / bins,
    binrange=(0, 1),
    kde=True,
    stat="probability",
    ax=axes,
    )


    axes.set(xlabel="DistCE (TVD)")
    axes.set(title=title)
    axes.set(ylim=(0, 0.135))

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return True



def compute_distce_values(
    predictions: List[PredictionRecord],
    ground_truth: List[NLIDistributionSample],
) -> np.ndarray:
    gt_dict = {gt.id: gt for gt in ground_truth}

    pred_probs_list = []
    human_probs_list = []

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

    if not pred_probs_list:
        return np.array([], dtype=float)

    pred_probs = np.array(pred_probs_list, dtype=float)
    human_probs = np.array(human_probs_list, dtype=float)
    return compute_distce(pred_probs, human_probs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NLI predictions")
    
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSONL file",
    )
    
    parser.add_argument(
        "--ground_truth_source",
        type=str,
        required=True,
        choices=["snli", "chaosnli"],
        help="Source of ground truth data",
    )
    
    parser.add_argument(
        "--ground_truth_split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Split of ground truth data",
    )
    
    parser.add_argument(
        "--chaosnli_path",
        type=str,
        default=None,
        help="Path to ChaosNLI JSONL file (required if ground_truth_source=chaosnli)",
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="Output file for evaluation results",
    )

    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save DistCE distribution plot (default: enabled)",
    )

    parser.add_argument(
        "--plot_dir",
        type=str,
        default="outputs/figures",
        help="Directory to save plots (default: outputs/figures)",
    )

    parser.add_argument(
        "--plot_title",
        type=str,
        default=None,
        help="Optional title for the plot (default: predictions file stem)",
    )
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading predictions from {args.predictions}...")
    predictions = load_predictions(Path(args.predictions))
    print(f"Loaded {len(predictions)} predictions")
    
    # Load ground truth
    print(f"Loading ground truth from {args.ground_truth_source}...")
    if args.ground_truth_source == "snli":
        reader = SNLIReader()
        ground_truth = reader.load_split(args.ground_truth_split)
    elif args.ground_truth_source == "chaosnli":
        if args.chaosnli_path is None:
            raise ValueError("--chaosnli_path is required when ground_truth_source=chaosnli")
        reader = ChaosNLIReader(data_path=args.chaosnli_path)
        ground_truth = reader.load_split(args.ground_truth_split)
    else:
        raise ValueError(f"Unknown ground truth source: {args.ground_truth_source}")
    
    print(f"Loaded {len(ground_truth)} ground truth samples")

    # Diagnostics: how many predictions can be matched to ground truth?
    gt_dict = {gt.id: gt for gt in ground_truth}
    overlap = sum(1 for p in predictions if p.id in gt_dict)
    valid = 0
    if ground_truth and isinstance(ground_truth[0], NLIDistributionSample):
        for p in predictions:
            gt = gt_dict.get(p.id)
            if gt is None or not isinstance(gt, NLIDistributionSample):
                continue
            pred_label = p.outputs.get("pred", -1)
            probs = p.outputs.get("probs", [])
            if pred_label < 0 or len(probs) != NLI_NUM_LABELS:
                continue
            if len(getattr(gt, "human_dist", [])) != NLI_NUM_LABELS:
                continue
            valid += 1
    else:
        for p in predictions:
            gt = gt_dict.get(p.id)
            if gt is None or not isinstance(gt, NLISample):
                continue
            pred_label = p.outputs.get("pred", -1)
            probs = p.outputs.get("probs", [])
            if pred_label < 0 or len(probs) != NLI_NUM_LABELS:
                continue
            valid += 1
    print(f"Matched prediction IDs: {overlap}/{len(predictions)}")
    print(f"Valid pairs used for metrics: {valid}/{len(predictions)}")
    
    # Evaluate
    print("Computing metrics...")
    evaluator = Evaluator()
    results = evaluator.evaluate(predictions, ground_truth)
    
    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for metric_name, value in sorted(results.items()):
        print(f"{metric_name:20s}: {value:.4f}")
    print("=" * 50)
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

    # Optional plot (DistCE distribution)
    if args.plot and ground_truth and isinstance(ground_truth[0], NLIDistributionSample):
        distce = compute_distce_values(
            predictions,
            [gt for gt in ground_truth if isinstance(gt, NLIDistributionSample)],
        )

        plot_dir = Path(args.plot_dir)
        plot_path = plot_dir / f"{Path(args.predictions).stem}_distce.png"
        title = args.plot_title or Path(args.predictions).stem

        ok = save_distce_plot_sns(distce, plot_path, title=title)
        if ok:
            print(f"DistCE plot saved to {plot_path}")
        else:
            print(
                "Plotting skipped: matplotlib is not installed. "
                "Install with `pip install -e \".[plot]\"` or use `--no-plot`."
            )


if __name__ == "__main__":
    main()
