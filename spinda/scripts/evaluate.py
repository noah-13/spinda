#!/usr/bin/env python3
"""
Evaluation script for HLV predictions.

Example usage:
    uv run python -m spinda.scripts.evaluate\
        --predictions outputs/predictions/toolkit_preds.json\
        --data_dir data/datasets/text_pair/chaosnli
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from spinda.data import (
    SingleTextClassificationJSONReader,
    SingleTextClassificationSample,
    SingleTextDistributionSample,
    SingleTextMultilabelJSONReader,
    SingleTextMultilabelDistributionSample,
    SingleTextMultilevelJSONReader,
    SingleTextMultilevelSample,
    TextPairClassificationSample,
    TextPairDistributionSample,
    TextPairMultilevelJSONReader,
    MultilevelSample,
    PredictionRecord,
    TextPairClassificationJSONReader,
)
from spinda.eval import Evaluator, analyze_distributional_disagreement, instance_error_records
from spinda.visualization import (
    save_distribution_ternary_plot,
    save_interactive_distribution_ternary_plot,
)
from spinda.data.json_io import load_records


MultilevelGroundTruthSample = Union[MultilevelSample, SingleTextMultilevelSample]


EVALUATION_CONFIG_KEYS = {
    "predictions", "predictions_format", "input_file", "human_labels",
    "output_file", "metrics", "ternary_plot", "ternary_plot_dir", "ternary_plot_title",
    "ternary_source", "ternary_browser", "analysis", "disagreement_groups",
    "disagreement_boundaries", "analysis_output_file", "instance_errors_file",
}


def _load_json_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Evaluation config not found: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in evaluation config {config_path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError(f"Evaluation config {config_path} must be a JSON object.")
    unknown = sorted(set(config) - EVALUATION_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unknown evaluation config keys in {config_path}: {', '.join(unknown)}")
    return config


def _load_merged_json_configs(paths: list[str]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        merged.update(_load_json_config(path))
    return merged


def _infer_prediction_kind(predictions: Sequence[PredictionRecord]) -> str:
    """Infer the output contract; format remains training-only metadata."""
    if not predictions:
        raise ValueError("Prediction file is empty.")
    kinds: set[str] = set()
    for prediction in predictions:
        outputs = prediction.outputs
        if isinstance(outputs.get("dimensions"), dict):
            kinds.add("multilevel")
        elif isinstance(outputs.get("pred"), list):
            kinds.add("multilabel")
        elif isinstance(outputs.get("pred"), int):
            kinds.add("categorical")
        else:
            raise ValueError(f"Prediction {prediction.id} has no recognized output shape.")
    if len(kinds) != 1:
        raise ValueError("Prediction file mixes categorical, multilabel, and multilevel output shapes.")
    return kinds.pop()


def _filter_metrics(results: Dict[str, Any], metrics: Optional[Sequence[str]]) -> Dict[str, Any]:
    if metrics is None:
        return results
    requested = set(metrics)
    if not requested or any(not isinstance(metric, str) or not metric for metric in metrics):
        raise ValueError("metrics must be a non-empty list of metric names when provided.")

    def select(values: Dict[str, Any]) -> Dict[str, Any]:
        return {name: value for name, value in values.items() if name in requested}

    if "overall" in results and all(isinstance(value, dict) for value in results.values()):
        available = {name for values in results.values() for name in values}
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(f"Unsupported metrics for these predictions: {', '.join(unknown)}")
        return {level: select(values) for level, values in results.items()}
    unknown = sorted(requested - set(results))
    if unknown:
        raise ValueError(f"Unsupported metrics for these predictions: {', '.join(unknown)}")
    return select(results)


def _get_multilevel_prediction_payload(
    prediction: PredictionRecord,
    level: str,
) -> Optional[Dict[str, Any]]:
    """Return one level's ``{probs, pred}`` payload from a prediction record."""
    levels = prediction.outputs.get("dimensions")
    if not isinstance(levels, dict):
        return None
    payload = levels.get(level)
    return payload if isinstance(payload, dict) else None


def _build_multilevel_eval_inputs(
    predictions: Sequence[PredictionRecord],
    ground_truth: Sequence[MultilevelGroundTruthSample],
    level: str,
) -> Tuple[List[PredictionRecord], List[TextPairDistributionSample]]:
    """Adapt one level of the multilevel JSON contract for ``Evaluator``."""
    ground_truth_by_id = {sample.id: sample for sample in ground_truth}
    level_predictions: List[PredictionRecord] = []
    level_ground_truth: List[TextPairDistributionSample] = []

    for prediction in predictions:
        sample = ground_truth_by_id.get(prediction.id)
        payload = _get_multilevel_prediction_payload(prediction, level)
        if sample is None:
            raise ValueError(f"Prediction {prediction.id} has no matching multilevel ground truth.")
        if payload is None:
            raise ValueError(f"Prediction {prediction.id} is missing outputs.dimensions.{level}.")
        level_predictions.append(
            PredictionRecord(
                id=prediction.id,
                task=prediction.task,
                split=prediction.split,
                source=prediction.source,
                outputs=payload,
            )
        )
        level_ground_truth.append(
            TextPairDistributionSample(
                id=sample.id,
                task=sample.task,
                split=sample.split,
                source=sample.source,
                label=sample.hard_labels[level],
                human_dist=sample.human_dists[level],
            )
        )
    return level_predictions, level_ground_truth


def load_predictions(
    file_path: Path,
    predictions_format: str = "json",
) -> List[PredictionRecord]:
    """Load predictions from a JSON array."""
    if predictions_format != "json":
        raise ValueError(f"Unsupported predictions_format={predictions_format!r}")
    if file_path.suffix != ".json":
        raise ValueError(f"Prediction input must be a .json file containing a top-level array, got {file_path}.")
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in prediction file {file_path}: {error}") from error
    records = payload.get("predictions") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{file_path} must contain a prediction array or an object with predictions.")
    predictions = []
    for index, data in enumerate(records, 1):
        if not isinstance(data, dict):
            raise ValueError(f"Prediction {index} in {file_path} must be a JSON object.")
        try:
            predictions.append(PredictionRecord(id=data["id"], task=data.get("task", "nli"), split=data.get("split", "unknown"), source=data.get("source"), outputs=data["outputs"]))
        except KeyError as error:
            raise ValueError(f"Prediction {index} in {file_path} requires {error.args[0]!r}.") from error
    return predictions


def load_human_labels(
    file_path: Path, prediction_kind: str,
) -> list[Union[TextPairClassificationSample, TextPairDistributionSample, SingleTextClassificationSample, SingleTextDistributionSample, SingleTextMultilabelDistributionSample, MultilevelSample, SingleTextMultilevelSample]]:
    """Load a test JSON that contains human annotations, inferring its supported contract."""
    rows = list(load_records(file_path, kind="human-label test records"))
    if not rows:
        raise ValueError(f"Human-label test file is empty: {file_path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{file_path} must be a top-level JSON array of objects.")
    records = [row for row in rows if isinstance(row, dict)]
    seen: set[str] = set()
    shapes: set[str] = set()
    for index, row in enumerate(records, 1):
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f"Record {index} in {file_path} requires a unique non-empty string id.")
        seen.add(identifier)
        if isinstance(row.get("text_a"), str) and isinstance(row.get("text_b"), str) and "text" not in row:
            shapes.add("pair")
        elif isinstance(row.get("text"), str) and "text_a" not in row and "text_b" not in row:
            shapes.add("single")
        else:
            raise ValueError(f"Record {index} in {file_path} must contain either text_a/text_b or text.")
    if len(shapes) != 1:
        raise ValueError(f"{file_path} mixes text-pair and single-text records.")
    shape = shapes.pop()

    if prediction_kind == "categorical":
        votes_by_row: list[list[int]] = []
        for index, row in enumerate(records, 1):
            votes = row.get("annotation_labels")
            if not isinstance(votes, list) or not votes or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in votes):
                raise ValueError(f"Record {index} in {file_path} must contain non-empty integer annotation_labels for categorical evaluation.")
            votes_by_row.append(votes)
        width = max(max(votes) for votes in votes_by_row) + 1
        samples = []
        for row, votes in zip(records, votes_by_row):
            counts = [votes.count(index) for index in range(width)]
            common = dict(id=row["id"], task="human_labels", split="test", source=str(file_path), label=max(range(width), key=counts.__getitem__))
            if shape == "pair":
                samples.append(TextPairDistributionSample(text_a=row["text_a"], text_b=row["text_b"], human_dist=[count / len(votes) for count in counts], annotation_labels=votes, **common))
            else:
                samples.append(SingleTextDistributionSample(text=row["text"], human_dist=[count / len(votes) for count in counts], annotation_labels=votes, **common))
        return samples

    if prediction_kind == "multilabel":
        votes_by_row = []
        for index, row in enumerate(records, 1):
            votes = row.get("annotation_label_sets")
            valid = isinstance(votes, list) and bool(votes) and all(isinstance(vote, list) and all(isinstance(label, int) and not isinstance(label, bool) and label >= 0 for label in vote) for vote in votes)
            if not valid:
                raise ValueError(f"Record {index} in {file_path} must contain annotation_label_sets for multilabel evaluation.")
            votes_by_row.append(votes)
        width = max((label for votes in votes_by_row for vote in votes for label in vote), default=-1) + 1
        if width < 2:
            raise ValueError(f"{file_path} multilabel annotations must cover at least two labels.")
        return [SingleTextMultilabelDistributionSample(
            id=row["id"], task="human_labels", split="test", source=str(file_path), text=row["text"],
            labels=[int(sum(label in vote for vote in votes) / len(votes) >= 0.5) for label in range(width)],
            human_probs=[sum(label in vote for vote in votes) / len(votes) for label in range(width)], annotation_label_sets=votes,
        ) for row, votes in zip(records, votes_by_row)]

    if prediction_kind == "multilevel":
        annotations = []
        for index, row in enumerate(records, 1):
            value = row.get("annotation_labels")
            if not isinstance(value, dict) or not value or any(not isinstance(level, str) or not isinstance(votes, list) or not votes or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in votes) for level, votes in value.items()):
                raise ValueError(f"Record {index} in {file_path} must contain a non-empty dimension-to-votes annotation_labels object for multilevel evaluation.")
            annotations.append(value)
        levels = tuple(annotations[0])
        if any(tuple(value) != levels for value in annotations):
            raise ValueError(f"{file_path} multilevel records must use the same ordered dimensions.")
        widths = {level: max(max(value[level]) for value in annotations) + 1 for level in levels}
        samples = []
        for row, value in zip(records, annotations):
            distributions = {level: [value[level].count(index) / len(value[level]) for index in range(widths[level])] for level in levels}
            hard = {level: max(range(widths[level]), key=lambda index, level=level: distributions[level][index]) for level in levels}
            common = dict(id=row["id"], task="human_labels", split="test", source=str(file_path), hard_labels=hard, human_dists=distributions, annotation_labels=value)
            if shape == "pair":
                samples.append(MultilevelSample(text_a=row["text_a"], text_b=row["text_b"], **common))
            else:
                samples.append(SingleTextMultilevelSample(text=row["text"], **common))
        return samples
    raise ValueError(f"Unsupported prediction contract: {prediction_kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HLV predictions")
    parser.add_argument("--config", nargs="+", action="extend", default=[], metavar="PATH", help="JSON configs merged left to right; CLI values override them.")
    
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Path to predictions JSON file (config or CLI).",
    )

    parser.add_argument(
        "--predictions_format", type=str, default="json", choices=["json"],
        help="Predictions file format (default: json)",
    )
    parser.add_argument("--metrics", nargs="+", default=None, help="Optional metric names to retain in results.")
    
    
    parser.add_argument("--input_file", type=str, default=None, help="Input JSON aligned with predictions; may be the same file as --human_labels.")
    parser.add_argument("--human_labels", type=str, default=None, help="Test JSON containing human annotation labels or distributions.")

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
        "--ternary-plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write ternary PNG diagnostics for three-class soft-label data (default: disabled)",
    )

    parser.add_argument(
        "--ternary-plot-dir",
        type=str,
        default=None,
        help="Directory to save ternary diagnostics (default: outputs/<ground_truth_source>/figures)",
    )

    parser.add_argument(
        "--ternary-plot-title",
        type=str,
        default=None,
        help="Optional ternary-plot title (default: predictions file stem)",
    )

    parser.add_argument(
        "--ternary-source",
        type=str,
        default="both",
        choices=["model", "human", "both"],
        help="Which distribution source to visualize in ternary plot (default: both).",
    )

    parser.add_argument(
        "--ternary-browser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With --ternary-plot, also write an interactive HTML ternary plot (default: enabled).",
    )

    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Write disagreement-stratified metrics and instance-error summaries for categorical soft labels.",
    )
    parser.add_argument(
        "--disagreement-groups",
        type=int,
        default=3,
        help="Number of entropy strata; defaults to low/medium/high tertiles.",
    )
    parser.add_argument(
        "--disagreement-boundaries",
        nargs="+",
        type=float,
        default=None,
        help="Explicit normalized-entropy cutoffs; must contain groups minus one values.",
    )
    parser.add_argument(
        "--analysis-output-file",
        type=str,
        default=None,
        help="JSON path for the aggregate analysis (default: next to evaluation output).",
    )
    parser.add_argument(
        "--instance-errors-file",
        type=str,
        default=None,
        help="CSV path for per-instance errors (default: next to aggregate analysis).",
    )
    
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", nargs="+", action="extend", default=[])
    bootstrap_args, _ = bootstrap_parser.parse_known_args()
    config_values = _load_merged_json_configs(bootstrap_args.config)
    args = parser.parse_args(namespace=argparse.Namespace(**config_values))
    if not args.predictions:
        parser.error("--predictions is required (in --config or on the command line).")
    if not args.input_file:
        parser.error("--input_file is required (in --config or on the command line).")
    if not args.human_labels:
        parser.error("--human_labels is required (in --config or on the command line).")
    
    # Load predictions
    print(f"Loading predictions from {args.predictions}...")
    predictions = load_predictions(
        Path(args.predictions),
        predictions_format=args.predictions_format,
    )
    print(f"Loaded {len(predictions)} predictions")
    prediction_kind = _infer_prediction_kind(predictions)
    print(f"Inferred prediction contract: {prediction_kind}")
    
    from spinda.scripts.predict import _load_inputs
    _, input_samples = _load_inputs(Path(args.input_file))
    input_ids = {sample["id"] for sample in input_samples}
    prediction_ids = {prediction.id for prediction in predictions}
    if input_ids != prediction_ids:
        raise ValueError("input_file IDs must exactly match prediction IDs.")
    multilevel_eval = prediction_kind == "multilevel"
    ground_truth = load_human_labels(Path(args.human_labels), prediction_kind)

    print(f"Loaded {len(ground_truth)} ground truth samples")

    Evaluator.validate_prediction_coverage(predictions, ground_truth)
    print(f"Matched prediction IDs: {len(predictions)}/{len(ground_truth)}")
    
    # Evaluate
    evaluator = Evaluator()
    eval_output = None
    if multilevel_eval:
        print("Computing multilevel metrics...")
        if not ground_truth or not isinstance(ground_truth[0], (MultilevelSample, SingleTextMultilevelSample)):
            raise ValueError("Multilevel evaluation requires multilevel ground truth.")

        multilevel_ground_truth = [
            gt for gt in ground_truth if isinstance(gt, (MultilevelSample, SingleTextMultilevelSample))
        ]
        if not any(_get_multilevel_prediction_payload(pred, next(iter(multilevel_ground_truth[0].human_dists))) is not None for pred in predictions):
            raise ValueError("Multilevel evaluation requires prediction records with nested 'outputs.dimensions' payloads.")

        results: Dict[str, Any] = {}
        aggregate_metrics: Dict[str, List[float]] = {}
        for level in multilevel_ground_truth[0].human_dists:
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

        if ground_truth and isinstance(ground_truth[0], (TextPairDistributionSample, SingleTextDistributionSample, SingleTextMultilabelDistributionSample)):
            valid = 0 if eval_output.pred_probs is None else len(eval_output.pred_probs)
        else:
            valid = 0 if eval_output.pred_labels is None else len(eval_output.pred_labels)
        print(f"Valid pairs used for metrics: {valid}/{len(predictions)}")

    results = _filter_metrics(results, args.metrics)

    analysis_payload = None
    instance_records: List[Dict[str, object]] = []
    if args.analysis:
        if multilevel_eval:
            analysis_payload = {}
            for level in multilevel_ground_truth[0].human_dists:
                level_predictions, level_ground_truth = _build_multilevel_eval_inputs(
                    predictions, multilevel_ground_truth, level
                )
                level_artifacts = evaluator.evaluate(level_predictions, level_ground_truth)
                if level_artifacts.pred_probs is None:
                    raise ValueError("Disagreement analysis requires categorical distribution labels.")
                analysis_payload[level] = analyze_distributional_disagreement(
                    level_artifacts.pred_probs, level_artifacts.human_probs,
                    level_artifacts.pred_labels, level_artifacts.true_labels,
                    num_groups=args.disagreement_groups,
                    boundaries=args.disagreement_boundaries,
                )
                for row in instance_error_records(
                    [prediction.id for prediction in level_predictions],
                    level_artifacts.pred_probs, level_artifacts.human_probs,
                    level_artifacts.pred_labels, level_artifacts.true_labels,
                    num_groups=args.disagreement_groups,
                    boundaries=args.disagreement_boundaries,
                ):
                    instance_records.append({"level": level, **row})
        else:
            if eval_output is None or eval_output.pred_probs is None:
                raise ValueError("Disagreement analysis requires categorical distribution labels.")
            analysis_payload = analyze_distributional_disagreement(
                eval_output.pred_probs, eval_output.human_probs,
                eval_output.pred_labels, eval_output.true_labels,
                num_groups=args.disagreement_groups,
                boundaries=args.disagreement_boundaries,
            )
            instance_records = instance_error_records(
                [prediction.id for prediction in predictions],
                eval_output.pred_probs, eval_output.human_probs,
                eval_output.pred_labels, eval_output.true_labels,
                num_groups=args.disagreement_groups,
                boundaries=args.disagreement_boundaries,
            )

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    if multilevel_eval:
        for level in multilevel_ground_truth[0].human_dists:
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

    if analysis_payload is not None:
        analysis_path = Path(args.analysis_output_file) if args.analysis_output_file else output_path.with_name(output_path.stem + "__analysis.json")
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis_payload, f, indent=2)
        errors_path = Path(args.instance_errors_file) if args.instance_errors_file else analysis_path.with_name(analysis_path.stem + "__instance_errors.csv")
        errors_path.parent.mkdir(parents=True, exist_ok=True)
        with open(errors_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(instance_records[0]) if instance_records else [])
            if instance_records:
                writer.writeheader()
                writer.writerows(instance_records)
        print(f"Analysis saved to {analysis_path}")
        print(f"Per-instance errors saved to {errors_path}")

    # Ternary diagnostics are optional and apply only to three-class soft-label data.
    if args.ternary_plot and eval_output is not None and eval_output.pred_probs is not None and eval_output.pred_probs.shape[1] == 3:
        plot_dir = Path(args.ternary_plot_dir) if args.ternary_plot_dir else Path("outputs/evaluation/figures")
        title = args.ternary_plot_title or Path(args.predictions).stem
        stem = Path(args.predictions).stem
        save_distribution_ternary_plot(
            model_distributions=eval_output.pred_probs,
            human_distributions=eval_output.human_probs,
            output_path=plot_dir / f"{stem}_ternary.png",
            distribution_source=args.ternary_source,
            title=title,
        )
        if args.ternary_browser:
            samples_by_id = {sample.id: sample for sample in ground_truth}
            matched_samples = [samples_by_id[prediction.id] for prediction in predictions]
            premises = [getattr(sample, "text_a", "") for sample in matched_samples]
            hypotheses = [getattr(sample, "text_b", getattr(sample, "text", "")) for sample in matched_samples]
            save_interactive_distribution_ternary_plot(
                model_distributions=eval_output.pred_probs,
                human_distributions=eval_output.human_probs,
                output_path=plot_dir / f"{stem}_ternary.html",
                distribution_source=args.ternary_source,
                title=title,
                ids=[prediction.id for prediction in predictions],
                premises=premises,
                hypotheses=hypotheses,
            )


if __name__ == "__main__":
    main()
