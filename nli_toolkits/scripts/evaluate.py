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
import re
from pathlib import Path
from typing import List, Optional

import numpy as np

from nli_toolkits.data import (
    ChaosNLIReader,
    NLIDistributionSample,
    NLISample,
    SNLIReader,
    PredictionRecord,
    NLI_NUM_LABELS
)
from nli_toolkits.eval import Evaluator, compute_distce


def _parse_chaosnli_preds_line(
    line: str,
    line_num: int,
    source: str,
) -> Optional[PredictionRecord]:
    parts = line.rstrip("\n").split("\t")
    if not parts:
        return None

    prob_cell = parts[0].strip()
    if "=" not in prob_cell:
        return None

    probs_map = {}
    for chunk in prob_cell.split("|"):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        key = key.strip().lower()
        try:
            prob = float(value)
        except ValueError:
            continue
        if key in ("entailment", "e"):
            probs_map["entailment"] = prob
        elif key in ("neutral", "n"):
            probs_map["neutral"] = prob
        elif key in ("contradiction", "c"):
            probs_map["contradiction"] = prob

    if not probs_map:
        return None

    probs = [
        probs_map.get("entailment", 0.0),
        probs_map.get("neutral", 0.0),
        probs_map.get("contradiction", 0.0),
    ]
    pred = int(max(range(len(probs)), key=lambda i: probs[i]))

    pred_id = None
    if len(parts) > 8 and parts[8] != "_":
        pred_id = parts[8]
    if pred_id is None:
        for cell in parts:
            if not cell or cell == "_":
                continue
            if ".jpg#" in cell or ".png#" in cell or re.search(r"#\\d", cell):
                pred_id = cell
                break
    if pred_id is None:
        pred_id = f"chaosnli_pred_{line_num}"

    return PredictionRecord(
        id=str(pred_id),
        task="nli",
        split="unknown",
        source=source,
        outputs={
            "probs": probs,
            "pred": pred,
        },
    )


def load_predictions(
    file_path: Path,
    predictions_format: str = "jsonl",
) -> List[PredictionRecord]:
    """Load predictions from JSONL file or ChaosNLI preds TSV-like format."""
    predictions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            if predictions_format == "jsonl":
                try:
                    data = json.loads(line)
                    pred = PredictionRecord(
                        id=data["id"],
                        task=data.get("task", "nli"),
                        split=data.get("split", "unknown"),
                        source=data.get("source"),
                        outputs=data["outputs"],
                    )
                    predictions.append(pred)
                    continue
                except json.JSONDecodeError:
                    raise ValueError(
                        f"Invalid JSONL on line {line_num} "
                        f"while predictions_format=jsonl"
                    )

            if predictions_format == "machamp":
                pred = _parse_chaosnli_preds_line(
                    line,
                    line_num,
                    source="machamp",
                )
                if pred is not None:
                    predictions.append(pred)
                    continue

            raise ValueError(
                f"Could not parse line {line_num} with "
                f"predictions_format={predictions_format}"
            )
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
        "--predictions_format",
        type=str,
        default="jsonl",
        choices=["jsonl", "machamp"],
        help="Predictions file format (default: jsonl)",
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
    predictions = load_predictions(
        Path(args.predictions),
        predictions_format=args.predictions_format,
    )
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
