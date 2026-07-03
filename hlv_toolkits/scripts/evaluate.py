#!/usr/bin/env python3
"""
Evaluation script for NLI predictions.

Example usage:
    uv run python -m hlv_toolkits.scripts.evaluate\
        --predictions outputs/predictions/toolkit_preds.jsonl\
        --ground_truth_source chaosnli
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from hlv_toolkits.data import (
    ChaosNLIReader,
    DiscoGeMReader,
    DiscoGeMMultiLevelSample,
    ProcessedJSONLReader,
    NLIDistributionSample,
    NLISample,
    SNLIReader,
    PredictionRecord,
)
from hlv_toolkits.eval import Evaluator
from hlv_toolkits.visualization import (
    save_distribution_ternary_plot,
    save_interactive_distribution_ternary_plot,
    save_tvd_plot,
)
from hlv_toolkits.eval.metrics import compute_tvd


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


def _collect_distribution_plot_metadata(
    predictions: List[PredictionRecord],
    ground_truth: List[NLIDistributionSample],
) -> tuple[list[str], list[str], list[str]]:
    gt_dict = {gt.id: gt for gt in ground_truth}
    ids: list[str] = []
    premises: list[str] = []
    hypotheses: list[str] = []

    for pred in predictions:
        gt = gt_dict.get(pred.id)
        if gt is None:
            continue

        pred_label = pred.outputs.get("pred", -1)
        probs = pred.outputs.get("probs", [])
        if pred_label < 0:
            continue
        if len(probs) != len(gt.human_dist):
            continue

        ids.append(str(gt.id))
        premises.append(str(gt.premise))
        hypotheses.append(str(gt.hypothesis))

    return ids, premises, hypotheses


def _get_multilevel_prediction_payload(pred: PredictionRecord, level: str) -> Optional[Dict[str, Any]]:
    levels = pred.outputs.get("levels")
    if isinstance(levels, dict):
        payload = levels.get(level)
        if isinstance(payload, dict):
            return payload
    return None


def _build_multilevel_eval_inputs(
    predictions: List[PredictionRecord],
    ground_truth: List[DiscoGeMMultiLevelSample],
    level: str,
) -> tuple[List[PredictionRecord], List[NLISample]]:
    gt_dict = {gt.id: gt for gt in ground_truth}
    level_predictions: List[PredictionRecord] = []
    level_ground_truth: List[NLISample] = []

    for pred in predictions:
        gt = gt_dict.get(pred.id)
        if gt is None:
            continue
        payload = _get_multilevel_prediction_payload(pred, level)
        if payload is None:
            continue

        human_dist = gt.human_dists.get(level, [])
        hard_label = gt.hard_labels.get(level, -1)
        if hard_label < 0 and not human_dist:
            continue

        pred_label = int(payload.get("pred", -1))
        level_predictions.append(
            PredictionRecord(
                id=pred.id,
                task=pred.task,
                split=pred.split,
                source=pred.source,
                outputs={
                    "probs": payload.get("probs", []),
                    "pred": pred_label,
                },
            )
        )

        if human_dist:
            label = hard_label if hard_label >= 0 else int(max(range(len(human_dist)), key=lambda i: human_dist[i]))
            level_ground_truth.append(
                NLIDistributionSample(
                    id=gt.id,
                    task=gt.task,
                    split=gt.split,
                    source=gt.source,
                    premise=gt.premise,
                    hypothesis=gt.hypothesis,
                    label=label,
                    human_dist=list(human_dist),
                )
            )
        else:
            level_ground_truth.append(
                NLISample(
                    id=gt.id,
                    task=gt.task,
                    split=gt.split,
                    source=gt.source,
                    premise=gt.premise,
                    hypothesis=gt.hypothesis,
                    label=hard_label,
                )
            )

    return level_predictions, level_ground_truth


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
        choices=["snli", "chaosnli", "discogem", "processed"],
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
        default="data/external/chaosnli/chaosNLI_snli.jsonl",
        help="Path to ChaosNLI JSONL file (required if ground_truth_source=chaosnli)",
    )
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="",
        help="Directory or file containing canonical JSONL samples",
    )
    parser.add_argument(
        "--processed_task",
        type=str,
        default="nli",
        choices=["nli", "discogem"],
        help="Task type stored in processed_data_dir",
    )

    parser.add_argument(
        "--discogem_path",
        type=str,
        default="",
        help="Path to the DiscoGeM 2.0 annotation archive. If empty, infer the default local path.",
    )

    parser.add_argument(
        "--discogem_version",
        type=str,
        default="auto",
        choices=["auto", "2.0"],
        help="DiscoGeM schema version",
    )

    parser.add_argument(
        "--discogem_label_mode",
        type=str,
        default="soft",
        choices=["soft", "hard"],
        help="Label mode for DiscoGeM evaluation ground truth",
    )
    parser.add_argument(
        "--discogem_label_level",
        type=str,
        default="level2",
        choices=["level1", "level2", "level3", "all"],
        help="DiscoGeM label granularity",
    )
    parser.add_argument(
        "--discogem_language",
        type=str,
        default="en",
        choices=["en", "de", "fr", "cs"],
        help="DiscoGeM language slice to use for version 2.0",
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help=(
            "Output file for evaluation results. If omitted, save to "
            "outputs/<ground_truth_source>/results/<predictions_stem>__eval.json"
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
    
    multilevel_eval = (
        (args.ground_truth_source == "discogem")
        or (args.ground_truth_source == "processed" and args.processed_task == "discogem")
    ) and args.discogem_label_level == "all"

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
    elif args.ground_truth_source == "discogem":
        reader = DiscoGeMReader(
            data_path=args.discogem_path or None,
            version=args.discogem_version,
            label_mode=args.discogem_label_mode,
            label_level=args.discogem_label_level,
            language=args.discogem_language,
        )
        ground_truth = reader.load_split(args.ground_truth_split)
    elif args.ground_truth_source == "processed":
        if not args.processed_data_dir:
            raise ValueError("--processed_data_dir is required when ground_truth_source=processed")
        reader = ProcessedJSONLReader(
            data_path=args.processed_data_dir,
            task=args.processed_task,
            label_level=args.discogem_label_level,
            label_mode=args.discogem_label_mode,
            language=args.discogem_language,
        )
        ground_truth = reader.load_split(args.ground_truth_split)
    else:
        raise ValueError(f"Unknown ground truth source: {args.ground_truth_source}")
    
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
        if not ground_truth or not isinstance(ground_truth[0], DiscoGeMMultiLevelSample):
            raise ValueError("Multilevel DiscoGeM evaluation requires DiscoGeMMultiLevelSample ground truth.")

        multilevel_ground_truth = [gt for gt in ground_truth if isinstance(gt, DiscoGeMMultiLevelSample)]
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

        if ground_truth and isinstance(ground_truth[0], NLIDistributionSample):
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
        output_path = Path(f"outputs/{args.ground_truth_source}/results") / f"{Path(args.predictions).stem}__eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

    # Optional plots
    if args.plot and ground_truth and isinstance(ground_truth[0], NLIDistributionSample):
        selected_plots = set(args.plots)
        plot_dir = Path(args.plot_dir) if args.plot_dir else Path(f"outputs/{args.ground_truth_source}/figures")
        title = args.plot_title or Path(args.predictions).stem
        ids, premises, hypotheses = _collect_distribution_plot_metadata(predictions, ground_truth)

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

        if "ternary" in selected_plots and eval_output.pred_probs is not None and eval_output.pred_probs.shape[1] == 3:
            plot_path = plot_dir / f"{Path(args.predictions).stem}_ternary.png"
            ok = save_distribution_ternary_plot(
                model_distributions=eval_output.pred_probs,
                human_distributions=eval_output.human_probs,
                output_path=plot_path,
                distribution_source=args.ternary_source,
                title=title,
            )
            if ok:
                print(f"Ternary plot saved to {plot_path}")
            else:
                print(
                    "Ternary plotting skipped: install `python-ternary` "
                    "and matplotlib, or use --plots tvd."
                )

            if args.ternary_browser:
                html_path = plot_dir / f"{Path(args.predictions).stem}_ternary_interactive.html"
                ok_html = save_interactive_distribution_ternary_plot(
                    model_distributions=eval_output.pred_probs,
                    human_distributions=eval_output.human_probs,
                    output_path=html_path,
                    distribution_source=args.ternary_source,
                    title=title,
                    ids=ids,
                    premises=premises,
                    hypotheses=hypotheses,
                )
                if ok_html:
                    print(f"Interactive ternary plot saved to {html_path}")
                else:
                    print(
                        "Interactive ternary plotting skipped: install `plotly` "
                        "or use --no-ternary_browser."
                    )
        elif "ternary" in selected_plots:
            print("Ternary plot skipped: only supported for 3-label distributions.")


if __name__ == "__main__":
    main()
