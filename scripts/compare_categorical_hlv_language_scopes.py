#!/usr/bin/env python3
"""Compare English and multilingual categorical-HLV strategy summaries.

This comparison is restricted to datasets with both scopes (currently
DiscoGeM and MultiPICo).  It compares within-setting strategy ranks and
hard-baseline improvements, rather than raw metrics across label spaces.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english-dir", type=Path, default=Path("outputs/analysis/english_categorical_hlv"))
    parser.add_argument("--multilingual-dir", type=Path, default=Path("outputs/analysis/multilingual_categorical_hlv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/multilingual_categorical_hlv"))
    return parser.parse_args()


def scope_table(directory: Path, scope: str) -> pd.DataFrame:
    table = pd.read_csv(directory / "strategy_by_dataset.csv")
    table = table.loc[table.dataset.isin(["discogem", "multipico"])].copy()
    table = table.rename(columns={column: f"{scope}_{column}" for column in table.columns if column not in {"dataset", "metric", "training_strategy"}})
    return table


def report(table: pd.DataFrame) -> str:
    tvd = table.loc[table.metric.eq("tvd")]
    accuracy = table.loc[table.metric.eq("accuracy")]
    def best(frame: pd.DataFrame, column: str) -> str:
        row = frame.loc[frame[column].idxmin()]
        return f"{row.training_strategy} ({column}={row[column]:.2f})"
    lines = [
        "# English-to-multilingual categorical HLV transfer",
        "",
        "## Design",
        "",
        "- Restricts the cross-language comparison to the shared DiscoGeM and MultiPICo families.",
        "- Uses average within-setting × model rank and improvement relative to `soft_to_hard/ce`; raw scores are not pooled across datasets or label spaces.",
        "- Positive `*_mean_improvement_vs_hard` means better than hard-label training, for every metric.",
        "",
        "## Transfer snapshot",
        "",
        f"- Best shared-family TVD rank: English {best(tvd, 'english_mean_rank')}; multilingual {best(tvd, 'multilingual_mean_rank')}.",
        f"- Best shared-family Accuracy rank: English {best(accuracy, 'english_mean_rank')}; multilingual {best(accuracy, 'multilingual_mean_rank')}.",
        "- Inspect `language_transfer_shared_dataset_summary.csv` for dataset-specific rank and hard-baseline changes. Coverage differs by strategy, particularly for ReL, so this is evidence of broad transfer rather than a fully balanced language-by-strategy factorial comparison.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    english = scope_table(args.english_dir, "english")
    multilingual = scope_table(args.multilingual_dir, "multilingual")
    keys = ["dataset", "metric", "training_strategy"]
    table = english.merge(multilingual, on=keys, how="inner", validate="one_to_one")
    table["rank_change_multilingual_minus_english"] = table.multilingual_mean_rank - table.english_mean_rank
    for scope in ("english", "multilingual"):
        table[f"{scope}_improves_vs_hard"] = table[f"{scope}_mean_improvement_vs_hard"].gt(0)
    table["improvement_sign_transfers"] = table.english_improves_vs_hard.eq(table.multilingual_improves_vs_hard)
    table = table.sort_values(["metric", "dataset", "training_strategy"])
    table.to_csv(args.output_dir / "language_transfer_shared_dataset_summary.csv", index=False)
    (args.output_dir / "language_transfer_summary.md").write_text(report(table), encoding="utf-8")
    print(f"Wrote {len(table)} shared-family English-to-multilingual comparisons to {args.output_dir}")


if __name__ == "__main__":
    main()
