"""Aggregate seed runs and create SPInDa disagreement-analysis figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


STRATEGY_COLORS = ("#4C78A8", "#E45756", "#54A24B", "#B279A2")


def _read_analysis(path: Path, level: str | None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if level is not None:
        try:
            payload = payload[level]
        except KeyError as error:
            raise ValueError(f"{path} has no analysis for level {level!r}.") from error
    try:
        return payload["disagreement_stratified"]
    except KeyError as error:
        raise ValueError(f"{path} is not a categorical SPInDa analysis JSON.") from error


def _read_metric_errors(path: Path, level: str | None, metric: str) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if level is not None:
        rows = [row for row in rows if row.get("level") == level]
    if not rows:
        raise ValueError(f"{path} has no instance errors" + (f" for {level!r}." if level else "."))
    try:
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path} must contain a numeric {metric} column.") from error
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite {metric} values.")
    return values


def _save_disagreement_metric(
    values: dict[str, dict[str, list[float]]],
    groups: tuple[str, ...],
    output: Path,
    title: str | None,
    metric: str,
) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.2, 3.1))
    x = np.arange(len(groups), dtype=float)
    for index, (label, series) in enumerate(values.items()):
        means = np.asarray([np.mean(series[group]) for group in groups], dtype=float)
        deviations = np.asarray([np.std(series[group], ddof=0) for group in groups], dtype=float)
        axis.errorbar(x, means, yerr=deviations, marker="o", capsize=3, linewidth=1.8, label=label, color=STRATEGY_COLORS[index % len(STRATEGY_COLORS)])
    axis.set_xticks(x, tuple(group.replace("_", " ").title() for group in groups))
    axis.set_xlabel("Human disagreement group")
    axis.set_ylabel(metric.upper())
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    if title:
        axis.set_title(title)
    if len(values) > 1:
        figure.legend(
            *axis.get_legend_handles_labels(), frameon=False, loc="upper center",
            bbox_to_anchor=(0.5, 0.99), ncol=min(3, len(values)),
        )
        figure.tight_layout(rect=(0, 0, 1, 0.84))
    else:
        figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _save_instance_metric_violin(values: dict[str, list[float]], output: Path, title: str | None, metric: str) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    labels = list(values)
    figure, axis = plt.subplots(figsize=(4.6, 2.8))
    violin = axis.violinplot([values[label] for label in labels], showmeans=False, showmedians=False, showextrema=False)
    series_by_label = [np.asarray(values[label], dtype=float) for label in labels]
    for index, body in enumerate(violin["bodies"]):
        color = STRATEGY_COLORS[index % len(STRATEGY_COLORS)]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.8)
    box = axis.boxplot(
        series_by_label, positions=range(1, len(labels) + 1), widths=0.18,
        patch_artist=True, showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.1},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
        medianprops={"color": "black", "linewidth": 1.3},
    )
    for patch in box["boxes"]:
        patch.set_zorder(3)
    for index, series in enumerate(series_by_label, start=1):
        axis.scatter(index, np.mean(series), color="black", s=18, zorder=5)
    axis.set_xticks(range(1, len(labels) + 1), labels)
    axis.set_ylabel(f"Instance-level {metric.upper()}")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    if title:
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation artifacts across seeds and create disagreement-analysis figures."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--run-dirs", nargs="+", type=Path,
        help="One strategy root per label; discovers seed artifacts automatically.",
    )
    input_group.add_argument(
        "--analysis-files", nargs="+", type=Path,
        help="Advanced: explicit analysis JSON files.",
    )
    parser.add_argument(
        "--instance-errors-files", nargs="+", type=Path,
        help="Advanced: one instance-error CSV per explicit analysis file.",
    )
    parser.add_argument(
        "--labels", nargs="+",
        help="One display label per run directory, or per explicit artifact pair.",
    )
    parser.add_argument(
        "--metric", choices=("tvd", "jsd", "kl", "ce", "l2"), default="tvd",
        help="Per-instance and disagreement-stratified metric to plot (default: tvd).",
    )
    parser.add_argument(
        "--plots", nargs="+", choices=("stratified", "instance"),
        default=("stratified", "instance"),
        help="Plot types to write: disagreement strata, instance distribution, or both.",
    )
    parser.add_argument("--level", help="Dimension to plot for multi-dimensional analyses.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation/disagreement_analysis"))
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    artifact_pairs: list[tuple[Path, Path | None, str]] = []
    if args.run_dirs:
        labels = args.labels or [run_dir.name for run_dir in args.run_dirs]
        if len(labels) != len(args.run_dirs):
            parser.error("--labels must provide one label per --run-dirs entry.")
        for run_dir, label in zip(args.run_dirs, labels):
            analysis_paths = sorted(run_dir.glob("seed_*/test/evaluation__analysis.json"))
            if not analysis_paths:
                parser.error(f"{run_dir} has no seed_*/test/evaluation__analysis.json artifacts. Run evaluate --analysis first.")
            for analysis_path in analysis_paths:
                errors_path = analysis_path.with_name(analysis_path.stem + "__instance_errors.csv")
                if "instance" in args.plots and not errors_path.is_file():
                    parser.error(f"Missing instance-error CSV beside {analysis_path}.")
                artifact_pairs.append((analysis_path, errors_path if errors_path.is_file() else None, label))
    else:
        if not args.analysis_files or not args.labels:
            parser.error("--analysis-files requires --labels.")
        if len(args.analysis_files) != len(args.labels):
            parser.error("--analysis-files and --labels must have the same length.")
        if "instance" in args.plots:
            if not args.instance_errors_files:
                parser.error("--instance-errors-files is required when --plots includes instance.")
            if len(args.analysis_files) != len(args.instance_errors_files):
                parser.error("--analysis-files and --instance-errors-files must have the same length.")
            artifact_pairs = list(zip(args.analysis_files, args.instance_errors_files, args.labels))
        else:
            artifact_pairs = [(analysis_path, None, label) for analysis_path, label in zip(args.analysis_files, args.labels)]

    by_label_group: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_label_errors: dict[str, list[float]] = defaultdict(list)
    group_names: tuple[str, ...] | None = None
    for analysis_path, errors_path, label in artifact_pairs:
        report = _read_analysis(analysis_path, args.level)
        groups = report.get("groups", {})
        current_group_names = tuple(groups)
        if not current_group_names:
            raise ValueError(f"{analysis_path} has no disagreement groups.")
        if group_names is None:
            group_names = current_group_names
        elif current_group_names != group_names:
            raise ValueError("All analysis artifacts must use the same disagreement grouping.")
        for group in current_group_names:
            metrics = groups[group].get("metrics")
            if not metrics or args.metric not in metrics:
                raise ValueError(f"{analysis_path} has no {args.metric} result for the {group} group.")
            by_label_group[label][group].append(float(metrics[args.metric]))
        if "instance" in args.plots:
            if errors_path is None:
                raise RuntimeError("Instance plots require an instance-error CSV.")
            by_label_errors[label].extend(_read_metric_errors(errors_path, args.level, args.metric).tolist())

    assert group_names is not None
    stratified_stem = f"disagreement_{args.metric}"
    instance_stem = f"instance_{args.metric}_violin"
    for suffix in ("png", "pdf"):
        if "stratified" in args.plots:
            _save_disagreement_metric(
                by_label_group, group_names, args.output_dir / f"{stratified_stem}.{suffix}",
                args.title, args.metric,
            )
        if "instance" in args.plots:
            _save_instance_metric_violin(
                by_label_errors, args.output_dir / f"{instance_stem}.{suffix}",
                args.title, args.metric,
            )
    print(f"Disagreement-analysis plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
