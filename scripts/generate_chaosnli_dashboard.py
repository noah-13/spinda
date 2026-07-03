#!/usr/bin/env python3
"""Generate a Jupyter notebook for inspecting ChaosNLI experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }


def build_notebook(screen_root: str, outputs_root: str) -> dict:
    cells = [
        markdown_cell(
            "# HLV ChaosNLI Experiment Dashboard\n\n"
            "这个 notebook 用来查看当前仓库里的 ChaosNLI 实验结果。\n\n"
            "默认分成两类：\n"
            "- Screening：从 `trainer_state.json` 读取的 dev 选择结果\n"
            "- Test results：单独导出的 test 评测结果\n"
        ),
        code_cell(
            "from pathlib import Path\n"
            "import json\n\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "import seaborn as sns\n\n"
            "pd.set_option('display.max_columns', 100)\n"
            "pd.set_option('display.width', 200)\n"
            "sns.set_theme(style='whitegrid')\n\n"
            "def find_repo_root(start: Path) -> Path:\n"
            "    for candidate in [start, *start.parents]:\n"
            "        if (candidate / '.git').exists() or (candidate / 'pyproject.toml').exists():\n"
            "            return candidate\n"
            "    return start\n\n"
            "ROOT = find_repo_root(Path.cwd())\n"
            f"OUTPUTS = ROOT / {outputs_root!r}\n"
            f"SCREEN_ROOT = ROOT / {screen_root!r}\n"
            "print('ROOT =', ROOT)\n"
            "print('SCREEN_ROOT exists =', SCREEN_ROOT.exists())\n"
        ),
        code_cell(
            "def parse_run_name(name: str) -> dict:\n"
            "    parts = name.split('__')\n"
            "    if len(parts) != 3:\n"
            "        return {'model': name, 'head': 'unknown', 'loss_kind': 'unknown', 'loss_value': 'unknown'}\n\n"
            "    model, head, loss_blob = parts\n"
            "    if loss_blob.startswith('soft_label_loss_'):\n"
            "        loss_kind = 'soft_label_loss'\n"
            "        loss_value = loss_blob[len('soft_label_loss_'):]\n"
            "    elif loss_blob.startswith('regression_loss_'):\n"
            "        loss_kind = 'regression_loss'\n"
            "        loss_value = loss_blob[len('regression_loss_'):]\n"
            "    else:\n"
            "        loss_kind = 'unknown'\n"
            "        loss_value = loss_blob\n\n"
            "    return {'model': model, 'head': head, 'loss_kind': loss_kind, 'loss_value': loss_value}\n\n"
            "def load_json(path: Path):\n"
            "    with open(path, 'r', encoding='utf-8') as f:\n"
            "        return json.load(f)\n\n"
            "def extract_eval_history(trainer_state: dict) -> pd.DataFrame:\n"
            "    rows = []\n"
            "    for item in trainer_state.get('log_history', []):\n"
            "        if 'eval_tvd' in item or 'eval_accuracy' in item or 'eval_kl_divergence' in item:\n"
            "            rows.append(item)\n"
            "    return pd.DataFrame(rows)\n\n"
            "def collect_screening_runs(screen_root: Path) -> tuple[pd.DataFrame, dict]:\n"
            "    summary_rows = []\n"
            "    histories = {}\n\n"
            "    for state_path in sorted(screen_root.glob('*/seed_*/checkpoint-*/trainer_state.json')):\n"
            "        run_dir = state_path.parents[2]\n"
            "        seed_dir = state_path.parents[1]\n"
            "        run_name = run_dir.name\n"
            "        seed = seed_dir.name.replace('seed_', '')\n"
            "        key = (run_name, seed)\n"
            "        step = int(state_path.parent.name.replace('checkpoint-', ''))\n"
            "        if key in histories and histories[key]['checkpoint_step'] > step:\n"
            "            continue\n\n"
            "        trainer_state = load_json(state_path)\n"
            "        eval_df = extract_eval_history(trainer_state)\n"
            "        if eval_df.empty:\n"
            "            continue\n\n"
            "        histories[key] = {'checkpoint_step': step, 'trainer_state': trainer_state, 'eval_df': eval_df}\n\n"
            "    for (run_name, seed), bundle in sorted(histories.items()):\n"
            "        trainer_state = bundle['trainer_state']\n"
            "        eval_df = bundle['eval_df'].sort_values('step').reset_index(drop=True)\n"
            "        meta = parse_run_name(run_name)\n"
            "        best_step = trainer_state.get('best_global_step')\n"
            "        best_row = eval_df.loc[eval_df['step'] == best_step]\n"
            "        if best_row.empty:\n"
            "            best_row = eval_df.nsmallest(1, 'eval_tvd')\n"
            "        best_row = best_row.iloc[0].to_dict()\n"
            "        final_row = eval_df.iloc[-1].to_dict()\n\n"
            "        summary_rows.append({\n"
            "            **meta,\n"
            "            'run_name': run_name,\n"
            "            'seed': int(seed),\n"
            "            'best_global_step': best_step,\n"
            "            'best_metric': trainer_state.get('best_metric'),\n"
            "            'best_checkpoint': trainer_state.get('best_model_checkpoint'),\n"
            "            'num_eval_points': len(eval_df),\n"
            "            'best_epoch': best_row.get('epoch'),\n"
            "            'best_eval_tvd': best_row.get('eval_tvd'),\n"
            "            'best_eval_kl_divergence': best_row.get('eval_kl_divergence'),\n"
            "            'best_eval_accuracy': best_row.get('eval_accuracy'),\n"
            "            'final_epoch': final_row.get('epoch'),\n"
            "            'final_eval_tvd': final_row.get('eval_tvd'),\n"
            "            'final_eval_kl_divergence': final_row.get('eval_kl_divergence'),\n"
            "            'final_eval_accuracy': final_row.get('eval_accuracy'),\n"
            "        })\n\n"
            "    summary_df = pd.DataFrame(summary_rows)\n"
            "    if not summary_df.empty:\n"
            "        summary_df = summary_df.sort_values(['best_eval_tvd', 'best_eval_kl_divergence', 'best_eval_accuracy'], ascending=[True, True, False]).reset_index(drop=True)\n"
            "    return summary_df, histories\n\n"
            "def infer_eval_split(path: Path) -> str:\n"
            "    name = path.name\n"
            "    path_text = '/'.join(path.parts)\n"
            "    if '__test_eval.json' in name or 'chaosnli_test' in path_text:\n"
            "        return 'test'\n"
            "    if '__dev_eval.json' in name or 'chaosnli_confirm' in path_text or 'script_evals' in path_text or 'evals' in path_text:\n"
            "        return 'dev'\n"
            "    return 'unknown'\n\n"
            "def summarize_eval_name(path: Path) -> str:\n"
            "    name = path.name\n"
            "    for suffix in ['__test_eval.json', '__dev_eval.json', '__eval.json']:\n"
            "        if name.endswith(suffix):\n"
            "            return name[:-len(suffix)]\n"
            "    return path.stem\n\n"
            "def parse_export_result_name(name: str) -> dict:\n"
            "    parts = name.split('__')\n"
            "    if len(parts) < 4:\n"
            "        return {'model': name, 'head': 'unknown', 'loss': 'unknown', 'seed': 'unknown'}\n"
            "    model, head, loss_blob, seed_blob = parts[:4]\n"
            "    if loss_blob.startswith('soft_label_loss_'):\n"
            "        loss = loss_blob[len('soft_label_loss_'):]\n"
            "    elif loss_blob.startswith('regression_loss_'):\n"
            "        loss = loss_blob[len('regression_loss_'):]\n"
            "    else:\n"
            "        loss = loss_blob\n"
            "    seed = seed_blob[len('seed_'):] if seed_blob.startswith('seed_') else seed_blob\n"
            "    return {'model': model, 'head': head, 'loss': loss, 'seed': seed}\n\n"
            "def collect_eval_jsons(outputs_root: Path) -> pd.DataFrame:\n"
            "    rows = []\n"
            "    for path in sorted(outputs_root.glob('**/*.json')):\n"
            "        if 'legacy' in path.parts:\n"
            "            continue\n"
            "        if (\n"
            "            'evals' not in path.parts\n"
            "            and 'script_evals' not in path.parts\n"
            "            and not path.name.endswith('__eval.json')\n"
            "            and not path.name.endswith('__dev_eval.json')\n"
            "            and not path.name.endswith('__test_eval.json')\n"
            "        ):\n"
            "            continue\n"
            "        try:\n"
            "            data = load_json(path)\n"
            "        except Exception:\n"
            "            continue\n"
            "        if not isinstance(data, dict):\n"
            "            continue\n"
            "        metric_keys = {'accuracy', 'tvd_mean', 'jsd_mean', 'kl_mean', 'soft_micro_f1', 'soft_macro_f1', 'distance_correlation'}\n"
            "        if not (metric_keys & set(data.keys())):\n"
            "            continue\n"
            "        result_name = summarize_eval_name(path)\n"
            "        rows.append({\n"
            "            'result_name': result_name,\n"
            "            'split': infer_eval_split(path),\n"
            "            **parse_export_result_name(result_name),\n"
            "            **data,\n"
            "        })\n"
            "    return pd.DataFrame(rows)\n"
        ),
        code_cell(
            "screen_df, screen_histories = collect_screening_runs(SCREEN_ROOT)\n"
            "eval_json_df = collect_eval_jsons(OUTPUTS)\n"
            "test_eval_df = eval_json_df[eval_json_df['split'] == 'test'].copy() if not eval_json_df.empty else pd.DataFrame()\n\n"
            "print('screen runs =', len(screen_df))\n"
            "print('test result files =', len(test_eval_df))\n"
            "screen_df.head()\n"
        ),
        code_cell(
            "if screen_df.empty:\n"
            "    print('No screening runs found under', SCREEN_ROOT)\n"
            "else:\n"
            "    display(screen_df[['run_name', 'model', 'head', 'loss_value', 'seed', 'best_eval_tvd', 'best_eval_kl_divergence', 'best_eval_accuracy', 'best_epoch']].sort_values(['best_eval_tvd', 'best_eval_kl_divergence', 'best_eval_accuracy'], ascending=[True, True, False]).reset_index(drop=True))\n"
        ),
        code_cell(
            "if not screen_df.empty:\n"
            "    topn = min(10, len(screen_df))\n"
            "    top_df = screen_df.nsmallest(topn, 'best_eval_tvd')[['run_name', 'model', 'head', 'loss_value', 'best_eval_tvd', 'best_eval_kl_divergence', 'best_eval_accuracy']]\n"
            "    display(top_df.reset_index(drop=True))\n\n"
            "    plt.figure(figsize=(12, 5))\n"
            "    sns.barplot(data=top_df, x='best_eval_tvd', y='run_name', hue='model', dodge=False, palette='viridis')\n"
            "    plt.title('Top runs by best dev TVD from trainer_state.json')\n"
            "    plt.xlabel('best_eval_tvd')\n"
            "    plt.ylabel('run_name')\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
        ),
        code_cell(
            "if not screen_df.empty:\n"
            "    pivot = screen_df.pivot_table(index=['head', 'loss_value'], columns='model', values='best_eval_tvd', aggfunc='mean')\n"
            "    display(pivot.sort_index())\n\n"
            "    plt.figure(figsize=(9, 4))\n"
            "    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='mako_r')\n"
            "    plt.title('Best dev TVD by model / head / loss')\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
        ),
        code_cell(
            "if not screen_df.empty:\n"
            "    summary = (\n"
            "        screen_df.groupby(['model', 'head', 'loss_value'], as_index=False)\n"
            "        .agg(\n"
            "            best_eval_tvd=('best_eval_tvd', 'mean'),\n"
            "            best_eval_kl_divergence=('best_eval_kl_divergence', 'mean'),\n"
            "            best_eval_accuracy=('best_eval_accuracy', 'mean'),\n"
            "            runs=('run_name', 'count'),\n"
            "        )\n"
            "        .sort_values(['best_eval_tvd', 'best_eval_kl_divergence', 'best_eval_accuracy'], ascending=[True, True, False])\n"
            "        .reset_index(drop=True)\n"
            "    )\n"
            "    display(summary)\n"
        ),
        code_cell(
            "def plot_curves(run_names, metric='eval_tvd'):\n"
            "    plt.figure(figsize=(12, 5))\n"
            "    for run_name in run_names:\n"
            "        key = None\n"
            "        for candidate in screen_histories:\n"
            "            if candidate[0] == run_name:\n"
            "                key = candidate\n"
            "                break\n"
            "        if key is None:\n"
            "            print('missing', run_name)\n"
            "            continue\n"
            "        df = screen_histories[key]['eval_df'].sort_values('step')\n"
            "        plt.plot(df['epoch'], df[metric], marker='o', label=run_name)\n"
            "    plt.title(f'{metric} over epochs')\n"
            "    plt.xlabel('epoch')\n"
            "    plt.ylabel(metric)\n"
            "    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n\n"
            "if not screen_df.empty:\n"
            "    top_runs = screen_df.nsmallest(min(5, len(screen_df)), 'best_eval_tvd')['run_name'].tolist()\n"
            "    print('Top runs:', top_runs)\n"
            "    plot_curves(top_runs, metric='eval_tvd')\n"
            "    plot_curves(top_runs, metric='eval_kl_divergence')\n"
            "    plot_curves(top_runs, metric='eval_accuracy')\n"
        ),
        code_cell(
            "if test_eval_df.empty:\n"
            "    print('No test result JSON files found yet.')\n"
            "else:\n"
            "    test_display_df = test_eval_df[['model', 'head', 'loss', 'seed', 'accuracy', 'tvd_mean', 'jsd_mean', 'kl_mean', 'soft_micro_f1', 'soft_macro_f1', 'distance_correlation']].copy()\n"
            "    display(test_display_df.sort_values(['tvd_mean', 'kl_mean', 'accuracy'], ascending=[True, True, False]).reset_index(drop=True))\n"
        ),
        code_cell(
            "RUN_TO_INSPECT = None\n\n"
            "if RUN_TO_INSPECT is None and not screen_df.empty:\n"
            "    RUN_TO_INSPECT = screen_df.iloc[0]['run_name']\n\n"
            "if RUN_TO_INSPECT is not None:\n"
            "    matched = [k for k in screen_histories if k[0] == RUN_TO_INSPECT]\n"
            "    if matched:\n"
            "        detail_df = screen_histories[matched[0]]['eval_df'].sort_values('step').reset_index(drop=True)\n"
            "        display(detail_df)\n"
            "    else:\n"
            "        print('Run not found:', RUN_TO_INSPECT)\n"
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a notebook for ChaosNLI experiment inspection.")
    parser.add_argument("--screen-root", default="outputs/chaosnli/runs", help="Relative path to screening run directory.")
    parser.add_argument("--outputs-root", default="outputs/chaosnli", help="Relative path to outputs root for ChaosNLI.")
    parser.add_argument("--output-notebook", default="notebooks/chaosnli_experiment_dashboard.ipynb", help="Notebook path to write.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook = build_notebook(screen_root=args.screen_root, outputs_root=args.outputs_root)
    output_path = Path(args.output_notebook)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
