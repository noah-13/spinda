#!/usr/bin/env python3
"""
Evaluation script for HLV predictions.

Example usage:
    uv run python -m hlv_toolkits.scripts.evaluate\
        --predictions outputs/predictions/toolkit_preds.jsonl\
        --data_dir data/processed/text_pair/chaosnli
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from hlv_toolkits.data import (
    SingleTextClassificationJSONLReader,
    SingleTextDistributionSample,
    TextPairClassificationSample,
    TextPairDistributionSample,
    PredictionRecord,
    TextPairClassificationJSONLReader,
)
from hlv_toolkits.eval import Evaluator
from hlv_toolkits.visualization import (
    save_distribution_ternary_plot,
    save_interactive_distribution_ternary_plot,
    save_tvd_plot,
)
from hlv_toolkits.eval.metrics import compute_tvd



def load_predictions(
    file_path: Path,
    predictions_format: str = "jsonl",
) -> List[PredictionRecord]:
    """Load predictions from JSONL."""
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

            raise ValueError(
                f"Could not parse line {line_num} with "
                f"predictions_format={predictions_format}"
            )
    return predictions


def load_ground_truth(file_path: Path) -> List[TextPairClassificationSample]:
    """Load lightweight external ground truth JSONL, ignoring extra metadata."""
    rows: List[tuple[str, int, Optional[List[float]]]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid ground-truth JSONL on line {line_num}") from error
            try:
                sample_id = data["id"]
                label = data["label"]
            except KeyError as error:
                raise ValueError(
                    f"Ground truth on line {line_num} requires {error.args[0]!r}."
                ) from error
            if not isinstance(sample_id, str) or not isinstance(label, int):
                raise ValueError(f"Ground truth on line {line_num} requires string id and integer label.")
            human_dist = data.get("human_dist")
            if human_dist is not None and (
                not isinstance(human_dist, list)
                or not all(isinstance(value, (int, float)) for value in human_dist)
            ):
                raise ValueError(f"Ground truth human_dist on line {line_num} must be a numeric list.")
            rows.append((sample_id, label, human_dist))

    has_distributions = any(human_dist is not None for _, _, human_dist in rows)
    if has_distributions and any(human_dist is None for _, _, human_dist in rows):
        raise ValueError("External ground truth must provide human_dist for every record or for none.")
    if has_distributions:
        return [
            TextPairDistributionSample(
                id=sample_id,
                task="external_ground_truth",
                label=label,
                human_dist=human_dist or [],
            )
            for sample_id, label, human_dist in rows
        ]
    return [
        TextPairClassificationSample(id=sample_id, task="external_ground_truth", label=label)
        for sample_id, label, _ in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HLV predictions")
    
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
        choices=["jsonl"],
        help="Predictions file format (default: jsonl)",
    )
    
    
    parser.add_argument(
        "--ground_truth_split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Split of ground truth data",
    )
    
    ground_truth_input = parser.add_mutually_exclusive_group(required=True)
    ground_truth_input.add_argument(
        "--data_dir",
        type=str,
        help="Processed dataset directory containing dataset.json",
    )
    ground_truth_input.add_argument(
        "--ground_truth",
        type=str,
        help="Lightweight external ground-truth JSONL file",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help=(
            "Output file for evaluation results. If omitted, save to "
            "outputs/evaluation/results/<predictions_stem>__eval.json"
        ),
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
        default=None,
        help="Directory to save plots (default: outputs/<ground_truth_source>/figures)",
    )

    parser.add_argument(
        "--plot_title",
        type=str,
        default=None,
        help="Optional title for the plot (default: predictions file stem)",
    )

    parser.add_argument(
        "--ternary_source",
        type=str,
        default="both",
        choices=["model", "human", "both"],
        help="Which distribution source to visualize in ternary plot (default: both).",
    )

    parser.add_argument(
        "--ternary_browser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to also save an interactive browser ternary plot as HTML (default: enabled).",
    )
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading predictions from {args.predictions}...")
    predictions = load_predictions(
        Path(args.predictions),
        predictions_format=args.predictions_format,
    )
    print(f"Loaded {len(predictions)} predictions")
    
    multilevel_eval = False
    if args.ground_truth:
        ground_truth = load_ground_truth(Path(args.ground_truth))
    else:
        manifest = json.loads((Path(args.data_dir) / "dataset.json").read_text(encoding="utf-8"))
        data_format = manifest.get("format")
        if data_format == "text_pair_label_distribution":
            ground_truth = TextPairClassificationJSONLReader(args.data_dir).load_split(args.ground_truth_split)
        elif data_format == "single_text_label_distribution":
            ground_truth = SingleTextClassificationJSONLReader(args.data_dir).load_split(args.ground_truth_split)
        else:
            raise ValueError(f"Unsupported evaluation format: {data_format!r}")

    print(f"Loaded {len(ground_truth)} ground truth samples")

    # Diagnostics: how many predictions can be matched to ground truth?
    gt_dict = {gt.id: gt for gt in ground_truth}
    overlap = sum(1 for p in predictions if p.id in gt_dict)
    print(f"Matched prediction IDs: {overlap}/{len(predictions)}")
    
    # Evaluate
    evaluator = Evaluator()
    eval_output = None
    if multilevel_eval:
        print("Computing multilevel metrics...")
        if not ground_truth or not isinstance(ground_truth[0], MultilevelSample):
            raise ValueError("Multilevel DiscoGeM evaluation requires MultilevelSample ground truth.")

        multilevel_ground_truth = [gt for gt in ground_truth if isinstance(gt, MultilevelSample)]
        if not any(_get_multilevel_prediction_payload(pred, "level1") is not None for pred in predictions):
            raise ValueError("Multilevel DiscoGeM evaluation requires prediction records with nested 'levels' outputs.")

        results: Dict[str, Any] = {}
        aggregate_metrics: Dict[str, List[float]] = {}
        for level in ["level1", "level2", "level3"]:
            level_predictions, level_ground_truth = _build_multilevel_eval_inputs(
                predictions,
                multilevel_ground_truth,
                level,
            )
            if not level_predictions:
                raise ValueError(f"No valid multilevel predictions found for {level}.")
            level_eval_output = evaluator.evaluate(level_predictions, level_ground_truth)
            results[level] = level_eval_output.metrics
            for metric_name, value in level_eval_output.metrics.items():
                aggregate_metrics.setdefault(metric_name, []).append(float(value))
            valid = len(level_predictions)
            print(f"{level}: valid pairs used for metrics = {valid}/{len(predictions)}")

        results["overall"] = {
            metric_name: float(sum(values) / len(values))
            for metric_name, values in aggregate_metrics.items()
            if values
        }
    else:
        print("Computing metrics...")
        eval_output = evaluator.evaluate(predictions, ground_truth)
        results = eval_output.metrics

        if ground_truth and isinstance(ground_truth[0], (TextPairDistributionSample, SingleTextDistributionSample)):
            valid = 0 if eval_output.pred_probs is None else len(eval_output.pred_probs)
        else:
            valid = 0 if eval_output.pred_labels is None else len(eval_output.pred_labels)
        print(f"Valid pairs used for metrics: {valid}/{len(predictions)}")

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    if multilevel_eval:
        for level in ["level1", "level2", "level3"]:
            print(f"[{level}]")
            for metric_name, value in sorted(results[level].items()):
                print(f"  {metric_name:18s}: {value:.4f}")
        if "overall" in results:
            print("[overall]")
            for metric_name, value in sorted(results["overall"].items()):
                print(f"  {metric_name:18s}: {value:.4f}")
    else:
        for metric_name, value in sorted(results.items()):
            print(f"{metric_name:20s}: {value:.4f}")
    print("=" * 50)
    
    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = Path("outputs/evaluation/results") / f"{Path(args.predictions).stem}__eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

    # Optional plots are only available for soft-label evaluation artifacts.
    if args.plot and eval_output is not None and eval_output.pred_probs is not None:
        selected_plots = set(args.plots)
        plot_dir = Path(args.plot_dir) if args.plot_dir else Path("outputs/evaluation/figures")
        title = args.plot_title or Path(args.predictions).stem
        if "tvd" in selected_plots:
            tvd = compute_tvd(eval_output.pred_probs, eval_output.human_probs)
            if tvd is not None:
                save_tvd_plot(tvd, plot_dir / f"{Path(args.predictions).stem}_tvd.png", title=title)
        if "ternary" in selected_plots and eval_output.pred_probs.shape[1] == 3:
            save_distribution_ternary_plot(eval_output.pred_probs, eval_output.human_probs, plot_dir / f"{Path(args.predictions).stem}_ternary.png", args.ternary_source, title)


if __name__ == "__main__":
    main()
