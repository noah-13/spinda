"""Create the paper-standard SPINDA disagreement-analysis figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PAPER_GROUPS = ("low", "medium", "high")
PAPER_COLORS = ("#4C78A8", "#E45756", "#54A24B", "#B279A2")


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
        raise ValueError(f"{path} is not a categorical SPINDA analysis JSON.") from error


def _read_tvd_errors(path: Path, level: str | None) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if level is not None:
        rows = [row for row in rows if row.get("level") == level]
    if not rows:
        raise ValueError(f"{path} has no instance errors" + (f" for {level!r}." if level else "."))
    try:
        values = np.asarray([float(row["tvd"]) for row in rows], dtype=float)
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path} must contain a numeric tvd column.") from error
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite TVD values.")
    return values


def _save_disagreement_tvd(values: dict[str, dict[str, list[float]]], output: Path, title: str | None) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.2, 3.1))
    x = np.arange(len(PAPER_GROUPS), dtype=float)
    for index, (label, series) in enumerate(values.items()):
        means = np.asarray([np.mean(series[group]) for group in PAPER_GROUPS], dtype=float)
        deviations = np.asarray([np.std(series[group], ddof=0) for group in PAPER_GROUPS], dtype=float)
        axis.errorbar(x, means, yerr=deviations, marker="o", capsize=3, linewidth=1.8, label=label, color=PAPER_COLORS[index % len(PAPER_COLORS)])
    axis.set_xticks(x, ("Low", "Medium", "High"))
    axis.set_xlabel("Human disagreement level (entropy tertiles)")
    axis.set_ylabel("TVD")
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


def _save_instance_tvd_violin(values: dict[str, list[float]], output: Path, title: str | None) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    labels = list(values)
    figure, axis = plt.subplots(figsize=(4.6, 2.8))
    violin = axis.violinplot([values[label] for label in labels], showmeans=False, showmedians=False, showextrema=False)
    series_by_label = [np.asarray(values[label], dtype=float) for label in labels]
    for index, body in enumerate(violin["bodies"]):
        color = PAPER_COLORS[index % len(PAPER_COLORS)]
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
    axis.set_ylabel("Instance-level TVD")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    if title:
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the two paper-standard disagreement-analysis figures from SPINDA evaluation artifacts."
    )
    parser.add_argument(
        "--analysis-files", nargs="+", required=True, type=Path,
        help="One or more JSON files written by evaluate --analysis; repeat labels for seed runs.",
    )
    parser.add_argument(
        "--instance-errors-files", nargs="+", required=True, type=Path,
        help="One CSV per analysis file, written by evaluate --analysis.",
    )
    parser.add_argument(
        "--labels", nargs="+", required=True,
        help="Display label for every input pair, for example: Hard_CE Hard_CE ReL ReL.",
    )
    parser.add_argument("--level", help="Dimension to plot for multi-dimensional analyses.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation/paper_analysis"))
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    if not (len(args.analysis_files) == len(args.instance_errors_files) == len(args.labels)):
        parser.error("--analysis-files, --instance-errors-files, and --labels must have the same length.")

    by_label_group: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_label_errors: dict[str, list[float]] = defaultdict(list)
    for analysis_path, errors_path, label in zip(args.analysis_files, args.instance_errors_files, args.labels):
        report = _read_analysis(analysis_path, args.level)
        groups = report.get("groups", {})
        if tuple(groups) != PAPER_GROUPS:
            raise ValueError(
                f"{analysis_path} must use the paper default of three entropy tertiles; "
                "rerun evaluate with --analysis and no custom disagreement grouping."
            )
        for group in PAPER_GROUPS:
            metrics = groups[group].get("metrics")
            if not metrics or "tvd" not in metrics:
                raise ValueError(f"{analysis_path} has no TVD result for the {group} group.")
            by_label_group[label][group].append(float(metrics["tvd"]))
        by_label_errors[label].extend(_read_tvd_errors(errors_path, args.level).tolist())

    for suffix in ("png", "pdf"):
        _save_disagreement_tvd(by_label_group, args.output_dir / f"disagreement_tvd.{suffix}", args.title)
        _save_instance_tvd_violin(by_label_errors, args.output_dir / f"instance_tvd_violin.{suffix}", args.title)
    print(f"Paper-standard plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
