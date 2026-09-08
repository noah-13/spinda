#!/usr/bin/env python3
"""Generate a notebook that summarizes completed ChaosNLI experiments.

Run: uv run python scripts/generate_chaosnli_results_notebook.py
The notebook scans outputs/chaosnli when it is executed, so later results are included.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def cell(kind: str, source: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def notebook() -> dict:
    cells = [
        cell("markdown", """# ChaosNLI results dashboard

This report scans `outputs/chaosnli` at execution time and summarizes only the three-seed-average test metrics for each configuration. Checkpoints are selected by minimum development TVD, and lower TVD is better throughout the report.

Default settings: `soft/ce`, `soft/mse`, `soft/jsd`, `soft/rel`, and `soft_to_hard/ce`.
"""),
        cell("code", """from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
try:
    from IPython.display import display
except ImportError:
    def display(value): print(value)

pd.set_option('display.max_columns', 100)
sns.set_theme(style='whitegrid', context='notebook')

def repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / 'pyproject.toml').exists(): return path
    raise FileNotFoundError('No repository root found')

ROOT = repo_root(Path.cwd())
RUN_ROOT = ROOT / 'outputs' / 'chaosnli'
EXPECTED_MODELS = {'microsoft_deberta-v3-large', 'roberta-base', 'bert-base-uncased', 'xlm-roberta-base', 'Twitter_twhin-bert-base', 'bert-base-multilingual-cased'}
EXPECTED_SETTINGS = {'soft__ce', 'soft__mse', 'soft__jsd', 'soft__rel', 'soft_to_hard__ce'}
EXPECTED_SEEDS = {42, 43, 44}
print(f'Repository: {ROOT}\\nResults: {RUN_ROOT} (exists={RUN_ROOT.exists()})')
"""),
        cell("code", """def parse_run(seed_dir: Path) -> dict:
    model, label_mode, strategy = seed_dir.parent.name.rsplit('__', 2)
    fold = seed_dir.parent.parent.name.removeprefix('fold_')
    return {'subset': seed_dir.parent.parent.parent.name, 'fold': int(fold) if fold.isdigit() else fold, 'model': model, 'label_mode': label_mode, 'strategy': strategy, 'setting': f'{label_mode}/{strategy}', 'seed': int(seed_dir.name.removeprefix('seed_'))}

def best_dev_tvd(seed_dir: Path):
    states = sorted(seed_dir.glob('checkpoint-*/trainer_state.json'))
    for path in states:
        state = json.loads(path.read_text())
        if state.get('best_model_checkpoint', '').rstrip('/').endswith(path.parent.name):
            return state.get('best_metric'), path.parent.name
    return (None, None) if not states else (json.loads(states[-1].read_text()).get('best_metric'), None)

rows = []
for path in sorted(RUN_ROOT.glob('*/*/*/seed_*/test/evaluation.json')):
    seed_dir = path.parents[1]
    row = parse_run(seed_dir); row.update(json.loads(path.read_text()))
    row['dev_best_tvd'], row['best_checkpoint'] = best_dev_tvd(seed_dir)
    row['path'] = str(seed_dir.relative_to(ROOT)); rows.append(row)
results = pd.DataFrame(rows)
if results.empty: raise FileNotFoundError(f'No evaluation.json under {RUN_ROOT}')
metrics = ['accuracy', 'tvd', 'jsd', 'kl', 'soft_micro_f1', 'soft_macro_f1', 'distance_correlation', 'l2', 'ce']
results = results.sort_values(['subset', 'fold', 'model', 'setting', 'seed']).reset_index(drop=True)
print(f'Loaded {len(results)} seed-level evaluations; displays use three-seed averages only.')
"""),
        cell("code", """# Completion: default = 6 models × 5 settings × 3 seeds = 90 runs per subset/fold.
coverage = results.groupby(['subset', 'fold'], as_index=False).agg(completed_runs=('seed', 'size'), models=('model', 'nunique'), settings=('setting', 'nunique'), seeds=('seed', lambda x: ', '.join(map(str, sorted(set(x))))))
coverage['expected_runs'] = len(EXPECTED_MODELS) * len(EXPECTED_SETTINGS) * len(EXPECTED_SEEDS)
coverage['complete_default_sweep'] = coverage.completed_runs.eq(coverage.expected_runs)
display(coverage)

missing = []
for (subset, fold), group in results.groupby(['subset', 'fold']):
    actual = set(zip(group.model, group.label_mode + '__' + group.strategy, group.seed))
    for model in EXPECTED_MODELS:
        for setting in EXPECTED_SETTINGS:
            for seed in EXPECTED_SEEDS:
                if (model, setting, seed) not in actual: missing.append((subset, fold, model, setting.replace('__', '/'), seed))
print('Default sweep is complete.' if not missing else f'Missing {len(missing)} runs.')
if missing: display(pd.DataFrame(missing, columns=['subset', 'fold', 'model', 'setting', 'seed']))
"""),
        cell("code", """# Aggregate seeds first. The rank is by mean test TVD (lower is better).
summary = results.groupby(['subset', 'fold', 'model', 'setting'], as_index=False).agg(seeds=('seed', 'nunique'), test_tvd_mean=('tvd', 'mean'), test_accuracy_mean=('accuracy', 'mean'), test_jsd_mean=('jsd', 'mean'), test_kl_mean=('kl', 'mean'), dev_tvd_mean=('dev_best_tvd', 'mean')).sort_values(['subset', 'fold', 'test_tvd_mean', 'test_accuracy_mean'], ascending=[True, True, True, False]).reset_index(drop=True)
display(summary)

print('Best configuration per subset/fold:')
display(summary.groupby(['subset', 'fold'], group_keys=False).head(1))
print('Top ten configurations per subset/fold:')
display(summary.groupby(['subset', 'fold'], group_keys=False).head(10))
"""),
        cell("code", """# Strategy comparison using three-seed means for each model.
strategy_summary = results.groupby(['subset', 'fold', 'setting'], as_index=False).agg(test_tvd_mean=('tvd', 'mean'), test_accuracy_mean=('accuracy', 'mean'), dev_tvd_mean=('dev_best_tvd', 'mean')).sort_values(['subset', 'fold', 'test_tvd_mean'])
display(strategy_summary)
"""),
        cell("code", """# Strategy comparison based on three-seed averages; no individual-seed values are shown.
g = sns.catplot(data=strategy_summary, x='setting', y='test_tvd_mean', col='subset', kind='bar', sharey=True, height=4.5, aspect=1.3, color='#77aadd')
for ax, (_, frame) in zip(g.axes.flat, results.groupby('subset', sort=True)):
    
    ax.set(xlabel='training setting', ylabel='mean test TVD (lower is better)'); ax.tick_params(axis='x', rotation=35)
g.fig.suptitle('ChaosNLI test TVD by training strategy', y=1.05)
plt.show()
"""),
        cell("code", """# Mean test TVD for every model/strategy combination.
subsets = sorted(summary.subset.unique())
fig, axes = plt.subplots(1, len(subsets), figsize=(7 * len(subsets), 5), squeeze=False)
for ax, subset in zip(axes.flat, subsets):
    matrix = summary[summary.subset.eq(subset)].pivot(index='model', columns='setting', values='test_tvd_mean')
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlGnBu_r', linewidths=.5, ax=ax)
    ax.set(title=f'{subset}: mean test TVD', xlabel='training setting', ylabel='model')
plt.tight_layout(); plt.show()
"""),
        cell("code", """# How the development selection metric transfers to the test set.
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=summary, x='dev_tvd_mean', y='test_tvd_mean', hue='setting', style='subset', s=70, ax=ax)
low = min(summary.dev_tvd_mean.min(), summary.test_tvd_mean.min()); high = max(summary.dev_tvd_mean.max(), summary.test_tvd_mean.max())
ax.plot([low, high], [low, high], '--', color='grey', linewidth=1)
ax.set(xlabel='best development TVD', ylabel='test TVD', title='Three-seed-average development versus test TVD')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left'); plt.tight_layout(); plt.show()
"""),
        cell("code", """# Optional CSV exports (created when this cell is run).
report_dir = RUN_ROOT / 'reports'; report_dir.mkdir(parents=True, exist_ok=True)
summary.to_csv(report_dir / 'summary_by_model_strategy.csv', index=False)
strategy_summary.to_csv(report_dir / 'summary_by_strategy.csv', index=False)
print(f'Wrote CSV tables to {report_dir}')
"""),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('notebooks/chaosnli_results_dashboard.ipynb'))
    args = parser.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook(), indent=2) + '\n')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
