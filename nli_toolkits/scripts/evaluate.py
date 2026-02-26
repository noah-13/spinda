#!/usr/bin/env python3
"""
Evaluation script for NLI predictions.

Example usage:
    uv run python -m nli_toolkits.scripts.evaluate\
        --predictions outputs/predictions/toolkit_preds.jsonl\
        --ground_truth_source chaosnli
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

from nli_toolkits.data import (
    ChaosNLIReader,
    NLIDistributionSample,
    SNLIReader,
    PredictionRecord,
)
from nli_toolkits.eval import Evaluator
from nli_toolkits.visualization import save_tvd_plot, save_ternary_plot
from nli_toolkits.eval.metrics import compute_tvd


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
        default="chaosNLI_v1.0/chaosNLI_snli.jsonl",
        help="Path to ChaosNLI JSONL file (required if ground_truth_source=chaosnli)",
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/results/evaluation_results.json",
        help="Output file for evaluation results",
    )

    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save plots (default: enabled)",
    )

    parser.add_argument(
        "--plots",
        nargs="+",
        default=["tvd", "ternary"],
        choices=["tvd", "ternary"],
        help=(
            "Plot types to save (default: tvd, ternary). "
            "Use as: --plots tvd ternary"
        ),
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
    print(f"Matched prediction IDs: {overlap}/{len(predictions)}")
    
    # Evaluate
    print("Computing metrics...")
    evaluator = Evaluator()
    eval_output = evaluator.evaluate(predictions, ground_truth)
    results = eval_output.metrics

    if ground_truth and isinstance(ground_truth[0], NLIDistributionSample):
        valid = 0 if eval_output.pred_probs is None else len(eval_output.pred_probs)
    else:
        valid = 0 if eval_output.pred_labels is None else len(eval_output.pred_labels)
    print(f"Valid pairs used for metrics: {valid}/{len(predictions)}")
    
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

    # Optional plots
    if args.plot and ground_truth and isinstance(ground_truth[0], NLIDistributionSample):
        selected_plots = set(args.plots)
        plot_dir = Path(args.plot_dir)
        title = args.plot_title or Path(args.predictions).stem

        if "tvd" in selected_plots:
            tvd = compute_tvd(eval_output.pred_probs, eval_output.human_probs)
            if tvd is None:
                print("DistCE plot skipped: no distribution artifacts were returned.")
                return
            plot_path = plot_dir / f"{Path(args.predictions).stem}_tvd.png"
            ok = save_tvd_plot(tvd, plot_path, title=title)
            if ok:
                print(f"DistCE plot saved to {plot_path}")
            else:
                print(
                    "Plotting skipped: matplotlib is not installed. "
                    "Install with `pip install -e \".[plot]\"` or use `--no-plot`."
                )

        if "ternary" in selected_plots:
            plot_path = plot_dir / f"{Path(args.predictions).stem}_ternary.png"
            ok = save_ternary_plot(
                distributions=[eval_output.pred_probs, eval_output.human_probs],
                output_path=plot_path,
                dataset_names=["Model", "Human"],
                colors=["k", "tab:orange"],
                markers=["D", "v"],
                title=title,
            )
            if ok:
                print(f"Ternary plot saved to {plot_path}")
            else:
                print(
                    "Ternary plotting skipped: install `python-ternary` "
                    "and matplotlib, or use --plots tvd."
                )


if __name__ == "__main__":
    main()
