#!/usr/bin/env python3
"""Generate a Jupyter notebook for inspecting DiscoGeM experiment outputs."""

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


def build_notebook(run_root: str, result_root: str, multilevel_result_root: str, discogem_path: str, discogem_language: str) -> dict:
    cells = [
        markdown_cell(
            """# HLV DiscoGeM Experiment Dashboard

This notebook summarizes both the legacy single-level DiscoGeM runs and the new multilevel runs.

It covers:
- training traces and best dev metrics
- single-level test metrics
- multilevel test metrics with level1/level2/level3/overall breakdowns
- wall-clock reconstruction
- dataset split / label statistics
"""
        ),
        code_cell(
            """from __future__ import annotations
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 220)
sns.set_theme(style='whitegrid')


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / '.git').exists() or (candidate / 'pyproject.toml').exists():
            return candidate
    return start


ROOT = find_repo_root(Path.cwd())
RUN_ROOT = ROOT / __RUN_ROOT__
SINGLE_RUN_ROOT = ROOT / 'outputs/discogem/runs/single'
RESULT_ROOT = RUN_ROOT
MULTILEVEL_RESULT_ROOT = RUN_ROOT / 'multilevel'
DISCOGEM_PATH = ROOT / __DISCOGEM_PATH__
DISCOGEM_LANGUAGE = __DISCOGEM_LANGUAGE__
print('ROOT =', ROOT)
print('SCREEN_ROOT exists =', RUN_ROOT.exists())
print('SINGLE_RUN_ROOT exists =', SINGLE_RUN_ROOT.exists())
print('RESULT_ROOT exists =', RESULT_ROOT.exists())
print('MULTILEVEL_RESULT_ROOT exists =', MULTILEVEL_RESULT_ROOT.exists())
print('DISCOGEM_PATH exists =', DISCOGEM_PATH.exists())


def parse_result_name(name: str) -> dict:
    stem = name.removesuffix('__test_eval.json').removesuffix('__test.jsonl')
    level, model, head, objective_blob, seed_blob = stem.split('__', 4)
    if objective_blob.startswith('soft_label_loss_'):
        objective = 'soft_label_loss'
        loss = objective_blob[len('soft_label_loss_'):]
    elif objective_blob.startswith('regression_loss_'):
        objective = 'regression_loss'
        loss = objective_blob[len('regression_loss_'):]
    else:
        objective = 'unknown'
        loss = objective_blob
    seed = int(seed_blob.removeprefix('seed_'))
    result_mode = 'multilevel' if level == 'multilevel' else 'single'
    return {
        'level': level,
        'result_mode': result_mode,
        'safe_model': model,
        'head': head,
        'objective': objective,
        'loss': loss,
        'seed': seed,
        'run_name': f'{model}__{head}__{objective_blob}',
    }


def parse_new_result_path(path: Path) -> dict:
    # New layout: .../<level>/<run>/seed_<n>/test/evaluation.json
    seed_dir = path.parent.parent
    run_dir = seed_dir.parent
    level = run_dir.parent.name
    return parse_result_name(f'{level}__{run_dir.name}__{seed_dir.name}__test_eval.json')


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def flatten_metrics(payload: dict, prefix: str = '') -> dict:
    flat = {}
    for key, value in payload.items():
        name = f'{prefix}{key}' if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, prefix=f'{name}_'))
        else:
            flat[name] = value
    return flat


def extract_eval_history(trainer_state: dict) -> pd.DataFrame:
    rows = []
    for item in trainer_state.get('log_history', []):
        if 'eval_tvd' in item or 'eval_accuracy' in item or 'eval_kl_divergence' in item:
            rows.append(item)
    return pd.DataFrame(rows)


def collect_screening_runs(run_root: Path) -> tuple[pd.DataFrame, dict]:
    summary_rows = []
    histories = {}
    if not run_root.exists():
        return pd.DataFrame(), histories

    for level_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        level = level_dir.name
        for run_dir in sorted(p for p in level_dir.iterdir() if p.is_dir()):
            for seed_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith('seed_')):
                trainer_states = sorted(seed_dir.glob('checkpoint-*/trainer_state.json'))
                if not trainer_states:
                    continue
                latest_state_path = max(trainer_states, key=lambda p: int(p.parent.name.replace('checkpoint-', '')))
                trainer_state = load_json(latest_state_path)
                eval_df = extract_eval_history(trainer_state)
                if eval_df.empty:
                    continue
                meta = parse_result_name(f'{level}__{run_dir.name}__{seed_dir.name}__test_eval.json')
                best_step = trainer_state.get('best_global_step')
                best_row = eval_df.loc[eval_df['step'] == best_step]
                if best_row.empty and 'eval_tvd' in eval_df.columns:
                    best_row = eval_df.nsmallest(1, 'eval_tvd')
                if best_row.empty:
                    best_row = eval_df.iloc[[-1]]
                best_row = best_row.iloc[0].to_dict()
                final_row = eval_df.sort_values('step').iloc[-1].to_dict()
                histories[(level, run_dir.name, seed_dir.name)] = eval_df.sort_values('step').reset_index(drop=True)
                summary_rows.append({
                    **meta,
                    'best_global_step': best_step,
                    'best_metric': trainer_state.get('best_metric'),
                    'best_checkpoint': trainer_state.get('best_model_checkpoint'),
                    'num_eval_points': len(eval_df),
                    'train_batch_size': trainer_state.get('train_batch_size'),
                    'num_input_tokens_seen': trainer_state.get('num_input_tokens_seen'),
                    'best_epoch': best_row.get('epoch'),
                    'best_eval_tvd': best_row.get('eval_tvd'),
                    'best_eval_kl_divergence': best_row.get('eval_kl_divergence'),
                    'best_eval_accuracy': best_row.get('eval_accuracy'),
                    'best_eval_samples_per_second': best_row.get('eval_samples_per_second'),
                    'final_epoch': final_row.get('epoch'),
                    'final_eval_tvd': final_row.get('eval_tvd'),
                    'final_eval_kl_divergence': final_row.get('eval_kl_divergence'),
                    'final_eval_accuracy': final_row.get('eval_accuracy'),
                })
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ['result_mode', 'level', 'best_eval_tvd', 'best_eval_kl_divergence', 'best_eval_accuracy'],
            ascending=[True, True, True, True, False],
        ).reset_index(drop=True)
    return summary_df, histories


def collect_test_results(result_roots: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for result_mode, result_root in result_roots.items():
        if not result_root.exists():
            continue
        for path in sorted(result_root.glob('**/test/evaluation.json')):
            meta = parse_new_result_path(path)
            payload = load_json(path)
            row = {**meta, 'source_root': result_mode}
            if meta['result_mode'] == 'multilevel' and all(key in payload for key in ('level1', 'level2', 'level3', 'overall')):
                for block_name in ('level1', 'level2', 'level3', 'overall'):
                    block = payload.get(block_name, {})
                    if isinstance(block, dict):
                        for metric_name, value in block.items():
                            row[f'{block_name}_{metric_name}'] = value
                overall = payload.get('overall', {})
                if isinstance(overall, dict):
                    row['accuracy'] = overall.get('accuracy')
                    row['tvd_mean'] = overall.get('tvd_mean')
                    row['jsd_mean'] = overall.get('jsd_mean')
                    row['kl_mean'] = overall.get('kl_mean')
                    row['soft_micro_f1'] = overall.get('soft_micro_f1')
                    row['soft_macro_f1'] = overall.get('soft_macro_f1')
                    row['distance_correlation'] = overall.get('distance_correlation')
            else:
                row.update(payload)
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df['run_label'] = df['safe_model'] + ' | ' + df['head'] + ' | ' + df['loss']
        sort_cols = [col for col in ['result_mode', 'level', 'safe_model', 'head', 'loss'] if col in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def format_hms_from_seconds(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f'{hours:02d}:{mins:02d}:{secs:02d}'


def collect_wall_clock_summary(run_root: Path, result_roots: dict[str, Path]) -> pd.DataFrame:
    checkpoint_rows = []
    for checkpoint_path in run_root.glob('**/checkpoint-*/trainer_state.json'):
        level = checkpoint_path.parts[-5] if len(checkpoint_path.parts) >= 5 else 'unknown'
        checkpoint_rows.append({'level': level, 'path': checkpoint_path, 'mtime': checkpoint_path.stat().st_mtime})

    eval_rows = []
    for result_mode, result_root in result_roots.items():
        if not result_root.exists():
            continue
        for eval_path in result_root.glob('**/test/evaluation.json'):
            meta = parse_new_result_path(eval_path)
            eval_rows.append({'level': meta['level'], 'result_mode': result_mode, 'path': eval_path, 'mtime': eval_path.stat().st_mtime})

    rows = []
    if checkpoint_rows and eval_rows:
        first_checkpoint = min(checkpoint_rows, key=lambda x: x['mtime'])
        last_eval = max(eval_rows, key=lambda x: x['mtime'])
        rows.append({
            'scope': 'all',
            'completed_runs': len(eval_rows),
            'wall_clock_hms': format_hms_from_seconds(max(0.0, last_eval['mtime'] - first_checkpoint['mtime'])),
            'wall_clock_seconds': max(0.0, last_eval['mtime'] - first_checkpoint['mtime']),
            'first_observed_checkpoint': str(first_checkpoint['path']),
            'last_observed_test_eval': str(last_eval['path']),
        })

        for level in sorted({row['level'] for row in eval_rows}):
            level_checkpoints = [row for row in checkpoint_rows if row['level'] == level]
            level_evals = [row for row in eval_rows if row['level'] == level]
            if not level_checkpoints or not level_evals:
                continue
            level_first = min(level_checkpoints, key=lambda x: x['mtime'])
            level_last = max(level_evals, key=lambda x: x['mtime'])
            rows.append({
                'scope': level,
                'completed_runs': len(level_evals),
                'wall_clock_hms': format_hms_from_seconds(max(0.0, level_last['mtime'] - level_first['mtime'])),
                'wall_clock_seconds': max(0.0, level_last['mtime'] - level_first['mtime']),
                'first_observed_checkpoint': str(level_first['path']),
                'last_observed_test_eval': str(level_last['path']),
            })
    return pd.DataFrame(rows)


screen_df, screen_histories = collect_screening_runs(RUN_ROOT)
single_train_df, single_histories = collect_screening_runs(SINGLE_RUN_ROOT)
if not single_train_df.empty:
    screen_df = pd.concat([screen_df, single_train_df], ignore_index=True)
screen_histories.update(single_histories)
result_roots = {'screen': RESULT_ROOT, 'single': SINGLE_RUN_ROOT}
test_df = collect_test_results(result_roots)
# Backward-compatible names used by older notebook cells/scripts.
discogem_results = test_df.copy()
discogem_screen = screen_df.copy()
single_test_df = test_df[test_df['result_mode'] == 'single'].copy() if not test_df.empty else pd.DataFrame()
multilevel_test_df = test_df[test_df['result_mode'] == 'multilevel'].copy() if not test_df.empty else pd.DataFrame()
wall_clock_df = collect_wall_clock_summary(ROOT / 'outputs/discogem', result_roots)
print('training runs =', len(screen_df))
print('test result files =', len(test_df))
print('single-level test result files =', len(single_test_df))
print('multilevel test result files =', len(multilevel_test_df))
print('wall clock rows =', len(wall_clock_df))
""".replace("__RUN_ROOT__", repr(run_root)).replace("__RESULT_ROOT__", repr(result_root)).replace("__MULTILEVEL_RESULT_ROOT__", repr(multilevel_result_root)).replace("__DISCOGEM_PATH__", repr(discogem_path)).replace("__DISCOGEM_LANGUAGE__", repr(discogem_language))
        ),
        code_cell(
            """if screen_df.empty:
    print('No DiscoGeM runs found under', RUN_ROOT, 'or', SINGLE_RUN_ROOT)
else:
    display(screen_df[['result_mode', 'level', 'safe_model', 'head', 'loss', 'best_eval_tvd', 'best_eval_kl_divergence', 'best_eval_accuracy', 'best_epoch', 'best_eval_samples_per_second']])
"""
        ),
        code_cell(
            """if test_df.empty:
    print('No test evaluation JSON found under', RESULT_ROOT, 'or', SINGLE_RUN_ROOT)
else:
    preferred_cols = ['result_mode', 'level', 'safe_model', 'head', 'loss', 'accuracy', 'tvd_mean', 'jsd_mean', 'kl_mean', 'soft_micro_f1', 'soft_macro_f1', 'distance_correlation', 'overall_accuracy', 'overall_tvd_mean', 'overall_jsd_mean', 'overall_kl_mean', 'overall_soft_micro_f1', 'overall_soft_macro_f1', 'overall_distance_correlation']
    display(test_df[[col for col in preferred_cols if col in test_df.columns]])
"""
        ),
        code_cell(
            """if not single_test_df.empty:
    single_summary = (
        single_test_df.groupby(['level', 'head', 'loss'], as_index=False)
        .agg(
            accuracy=('accuracy', 'mean'),
            tvd_mean=('tvd_mean', 'mean'),
            jsd_mean=('jsd_mean', 'mean'),
            kl_mean=('kl_mean', 'mean'),
            soft_micro_f1=('soft_micro_f1', 'mean'),
            soft_macro_f1=('soft_macro_f1', 'mean'),
            distance_correlation=('distance_correlation', 'mean'),
            runs=('safe_model', 'count'),
        )
        .sort_values(['level', 'tvd_mean', 'kl_mean', 'accuracy'], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )
    display(single_summary)

if not multilevel_test_df.empty:
    multilevel_summary = (
        multilevel_test_df.groupby(['head', 'loss'], as_index=False)
        .agg(
            overall_accuracy=('accuracy', 'mean'),
            overall_tvd_mean=('tvd_mean', 'mean'),
            overall_jsd_mean=('jsd_mean', 'mean'),
            overall_kl_mean=('kl_mean', 'mean'),
            overall_soft_micro_f1=('soft_micro_f1', 'mean'),
            overall_soft_macro_f1=('soft_macro_f1', 'mean'),
            overall_distance_correlation=('distance_correlation', 'mean'),
            level1_accuracy=('level1_accuracy', 'mean'),
            level2_accuracy=('level2_accuracy', 'mean'),
            level3_accuracy=('level3_accuracy', 'mean'),
            runs=('safe_model', 'count'),
        )
        .sort_values(['overall_tvd_mean', 'overall_kl_mean', 'overall_accuracy'], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    display(multilevel_summary)
"""
        ),
        code_cell(
            """comparison_rows = []

metric_names = ['accuracy', 'tvd_mean', 'jsd_mean', 'kl_mean', 'soft_micro_f1', 'soft_macro_f1', 'distance_correlation']

if not single_test_df.empty:
    single_rows = single_test_df.copy()
    single_rows = single_rows[['safe_model', 'head', 'loss', 'level', *[c for c in metric_names if c in single_rows.columns]]].copy()
    single_rows['mode'] = 'single'
    comparison_rows.append(single_rows)

if not multilevel_test_df.empty:
    for level in ['level1', 'level2', 'level3']:
        rename_map = {
            f'{level}_accuracy': 'accuracy',
            f'{level}_tvd_mean': 'tvd_mean',
            f'{level}_jsd_mean': 'jsd_mean',
            f'{level}_kl_mean': 'kl_mean',
            f'{level}_soft_micro_f1': 'soft_micro_f1',
            f'{level}_soft_macro_f1': 'soft_macro_f1',
            f'{level}_distance_correlation': 'distance_correlation',
        }
        available = [src for src in rename_map if src in multilevel_test_df.columns]
        if not available:
            continue
        level_rows = multilevel_test_df[['safe_model', 'head', 'loss', *available]].copy()
        level_rows = level_rows.rename(columns={src: dst for src, dst in rename_map.items() if src in available})
        level_rows['level'] = level
        level_rows['mode'] = 'multilevel'
        comparison_rows.append(level_rows[['safe_model', 'head', 'loss', 'level', 'mode', *[c for c in metric_names if c in level_rows.columns]]])

if comparison_rows:
    comparison_df = pd.concat(comparison_rows, ignore_index=True)
    comparison_df['level'] = pd.Categorical(comparison_df['level'], categories=['level1', 'level2', 'level3'], ordered=True)
    display(comparison_df.sort_values(['level', 'mode', 'head', 'loss', 'safe_model']).reset_index(drop=True))

    tvd_rank_cols = ['level', 'mode', 'safe_model', 'head', 'loss']
    tvd_rank_cols += [col for col in ['accuracy', 'tvd_mean', 'jsd_mean', 'kl_mean', 'soft_micro_f1', 'soft_macro_f1', 'distance_correlation'] if col in comparison_df.columns]
    tvd_ranked_df = comparison_df[tvd_rank_cols].copy()
    tvd_ranked_df = tvd_ranked_df.sort_values(['level', 'tvd_mean', 'mode', 'head', 'loss', 'safe_model'], ascending=[True, True, True, True, True, True]).reset_index(drop=True)
    display(tvd_ranked_df)

    plot_metrics = [metric for metric in ['accuracy', 'tvd_mean', 'kl_mean'] if metric in comparison_df.columns]
    if plot_metrics:
        fig, axes = plt.subplots(len(plot_metrics), 1, figsize=(14, 4 * len(plot_metrics)), squeeze=False)
        for ax, metric in zip(axes.flat, plot_metrics):
            sns.boxplot(data=comparison_df, x='level', y=metric, hue='mode', ax=ax)
            sns.stripplot(data=comparison_df, x='level', y=metric, hue='mode', dodge=True, alpha=0.25, size=2.5, ax=ax)
            ax.set_title(f'Single vs multilevel per concrete level: {metric}')
            ax.set_xlabel('level')
            ax.set_ylabel(metric)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:2], labels[:2], title='mode', loc='best')
        plt.tight_layout()
        plt.show()
"""
        ),
        code_cell(
            """if not single_test_df.empty:
    single_summary = (
        single_test_df.groupby(['level', 'head', 'loss'], as_index=False)
        .agg(
            accuracy=('accuracy', 'mean'),
            tvd_mean=('tvd_mean', 'mean'),
            jsd_mean=('jsd_mean', 'mean'),
            kl_mean=('kl_mean', 'mean'),
            soft_micro_f1=('soft_micro_f1', 'mean'),
            soft_macro_f1=('soft_macro_f1', 'mean'),
            distance_correlation=('distance_correlation', 'mean'),
            runs=('safe_model', 'count'),
        )
        .sort_values(['level', 'tvd_mean', 'kl_mean', 'accuracy'], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )
    display(single_summary)

if not multilevel_test_df.empty:
    multilevel_summary = (
        multilevel_test_df.groupby(['head', 'loss'], as_index=False)
        .agg(
            overall_accuracy=('accuracy', 'mean'),
            overall_tvd_mean=('tvd_mean', 'mean'),
            overall_jsd_mean=('jsd_mean', 'mean'),
            overall_kl_mean=('kl_mean', 'mean'),
            overall_soft_micro_f1=('soft_micro_f1', 'mean'),
            overall_soft_macro_f1=('soft_macro_f1', 'mean'),
            overall_distance_correlation=('distance_correlation', 'mean'),
            level1_accuracy=('level1_accuracy', 'mean'),
            level2_accuracy=('level2_accuracy', 'mean'),
            level3_accuracy=('level3_accuracy', 'mean'),
            runs=('safe_model', 'count'),
        )
        .sort_values(['overall_tvd_mean', 'overall_kl_mean', 'overall_accuracy'], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    display(multilevel_summary)
"""
        ),
        code_cell(
            """if not single_test_df.empty:
    best_per_level = single_test_df.sort_values(['level', 'tvd_mean', 'kl_mean', 'accuracy'], ascending=[True, True, True, False]).groupby('level').head(10)
    levels = list(best_per_level['level'].drop_duplicates())
    fig, axes = plt.subplots(len(levels), 1, figsize=(14, 4 * len(levels)), squeeze=False)
    for ax, level in zip(axes.flat, levels):
        level_df = best_per_level[best_per_level['level'] == level].copy()
        sns.barplot(data=level_df, x='tvd_mean', y='run_label', hue='run_label', dodge=False, legend=False, ax=ax)
        ax.set_title(f'Top single-level test TVD runs: {level}')
        ax.set_xlabel('tvd_mean')
        ax.set_ylabel('run_label')
    plt.tight_layout()
    plt.show()

if not multilevel_test_df.empty:
    best_multi = multilevel_test_df.sort_values(['overall_tvd_mean', 'overall_kl_mean', 'overall_accuracy'], ascending=[True, True, False]).groupby(['head', 'loss']).head(10)
    fig, ax = plt.subplots(figsize=(14, max(4, 0.35 * len(best_multi))))
    sns.barplot(data=best_multi, x='overall_tvd_mean', y='run_label', hue='head', dodge=False, ax=ax)
    ax.set_title('Top multilevel test TVD runs (overall)')
    ax.set_xlabel('overall_tvd_mean')
    ax.set_ylabel('run_label')
    plt.tight_layout()
    plt.show()
"""
        ),
        code_cell(
            """if wall_clock_df.empty:
    print('No wall-clock summary could be reconstructed from current outputs.')
else:
    print('Wall-clock time reconstructed from the earliest observed checkpoint to the latest observed test eval.')
    display(wall_clock_df)
"""
        ),
        code_cell(
            """def load_discogem_20_en(path: Path) -> pd.DataFrame:
    import tarfile

    with tarfile.open(path, 'r:gz') as tf:
        member = 'DiscoGeM2.0_annotation/DiscoGeM2.0_items.csv'
        raw = tf.extractfile(member)
        assert raw is not None
        text = raw.read().decode('utf-8-sig').splitlines()
    return pd.DataFrame(list(csv.DictReader(text, delimiter='	')))


dataset_df = load_discogem_20_en(DISCOGEM_PATH)
dataset_df = dataset_df[(dataset_df['arg1_context_en'].fillna('') != '') & (dataset_df['arg2_context_en'].fillna('') != '')].copy()
dataset_df['split_norm'] = dataset_df['split'].astype(str).str.strip().str.lower().replace({'valid': 'dev', 'validation': 'dev'})
dataset_df['label_level3'] = dataset_df['WAWA_en'].fillna('').astype(str).str.strip()
dataset_df['label_level2'] = dataset_df['label_level3'].replace({
    'precedence': 'asynchronous', 'succession': 'asynchronous',
    'reason': 'cause', 'result': 'cause',
    'arg1-as-goal': 'purpose', 'arg2-as-goal': 'purpose',
    'arg1-as-cond': 'condition', 'arg1-as-negcond': 'condition', 'arg2-as-cond': 'condition', 'arg2-as-negcond': 'condition',
    'arg1-as-denier': 'concession', 'arg2-as-denier': 'concession',
    'arg1-as-instance': 'instantiation', 'arg2-as-instance': 'instantiation',
    'arg1-as-detail': 'level-of-detail', 'arg2-as-detail': 'level-of-detail',
    'arg1-as-excpt': 'exception', 'arg2-as-excpt': 'exception',
    'arg1-as-manner': 'manner', 'arg2-as-manner': 'manner',
    'arg1-as-subst': 'substitution', 'arg2-as-subst': 'substitution',
})
dataset_df['label_level1'] = dataset_df['label_level2'].replace({
    'synchronous': 'temporal', 'asynchronous': 'temporal',
    'cause': 'contingency', 'purpose': 'contingency', 'condition': 'contingency',
    'concession': 'comparison', 'contrast': 'comparison', 'similarity': 'comparison',
    'equivalence': 'expansion', 'instantiation': 'expansion', 'level-of-detail': 'expansion', 'conjunction': 'expansion', 'disjunction': 'expansion', 'exception': 'expansion', 'manner': 'expansion', 'substitution': 'expansion',
})
split_size_df = dataset_df.groupby('split_norm', as_index=False).size().rename(columns={'size': 'n_examples'})
level_summaries = []
for level_col in ['label_level1', 'label_level2', 'label_level3']:
    level_summaries.append({
        'level': level_col.removeprefix('label_'),
        'n_label_types': int(dataset_df[level_col].nunique()),
        'label_types': ', '.join(sorted(dataset_df[level_col].dropna().unique())),
    })
label_summary_df = pd.DataFrame(level_summaries)
print('Dataset size by split')
display(split_size_df.sort_values('split_norm').reset_index(drop=True))
print('Label summary by level')
display(label_summary_df)
"""
        ),
        code_cell(
            """# Align single-level and multilevel runs by concrete label level.
comparison_metrics = ['accuracy', 'tvd_mean', 'jsd_mean', 'kl_mean', 'soft_micro_f1', 'soft_macro_f1', 'distance_correlation']
comparison_rows = []
for _, row in single_test_df.iterrows():
    item = {
        'comparison_level': row.get('level'),
        'source_mode': 'single',
        'safe_model': row.get('safe_model'),
        'head': row.get('head'),
        'loss': row.get('loss'),
        'seed': row.get('seed'),
    }
    for metric in comparison_metrics:
        item[metric] = row.get(metric)
    comparison_rows.append(item)

for _, row in multilevel_test_df.iterrows():
    for level_name in ['level1', 'level2', 'level3', 'overall']:
        prefix = '' if level_name == 'overall' else level_name + '_'
        item = {
            'comparison_level': level_name,
            'source_mode': 'multilevel',
            'safe_model': row.get('safe_model'),
            'head': row.get('head'),
            'loss': row.get('loss'),
            'seed': row.get('seed'),
        }
        for metric in comparison_metrics:
            item[metric] = row.get(prefix + metric)
        comparison_rows.append(item)

level_comparison_df = pd.DataFrame(comparison_rows)
if not level_comparison_df.empty:
    level_comparison_df = level_comparison_df.sort_values(
        ['comparison_level', 'safe_model', 'head', 'loss', 'source_mode', 'seed']
    ).reset_index(drop=True)
    level_comparison_summary = (
        level_comparison_df.groupby(
            ['comparison_level', 'source_mode', 'safe_model', 'head', 'loss'], as_index=False
        )[comparison_metrics].mean()
        .sort_values(['comparison_level', 'tvd_mean', 'kl_mean', 'accuracy'], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )
    print('Level-aligned comparison: single-level vs multilevel')
    display(level_comparison_summary)
"""
        ),
        code_cell(
            """# Export the current dashboard tables so Run All refreshes the HTML result page.
html_path = ROOT / 'notebooks' / 'discogem_results.html'
html_css = '''<style>body{font-family:Arial,sans-serif;margin:24px;color:#222}h1,h2{margin-top:28px}.table-wrap{overflow-x:auto;margin:12px 0 28px}table{border-collapse:collapse;font-size:12px;white-space:nowrap}th,td{border:1px solid #ccc;padding:5px 7px}th{background:#f2f2f2;position:sticky;top:0}tr:nth-child(even){background:#fafafa}.note{background:#eef6ff;border-left:4px solid #4285f4;padding:10px}</style>'''
html_parts = [
    '<!doctype html><html><head><meta charset=\"utf-8\"><title>DiscoGeM results</title>',
    html_css,
    '</head><body><h1>DiscoGeM results</h1>',
    '<div class=\"note\">Generated from the current outputs/discogem layout. Includes all test runs, aggregate comparisons, per-level multilevel metrics, and TVD-ranked results.</div>',
]
if discogem_results.empty:
    html_parts.append('<p>No test evaluation results found.</p>')
else:
    html_parts.append(f'<h2>All test runs ({len(discogem_results)})</h2><div class=\"table-wrap\">{discogem_results.to_html(index=False)}</div>')
    if not level_comparison_df.empty:
        html_parts.append(f'<h2>Level-aligned detailed comparison (single vs multilevel)</h2><div class=\"table-wrap\">{level_comparison_df.to_html(index=False)}</div>')
        html_parts.append(f'<h2>Level-aligned mean comparison by model / loss</h2><div class=\"table-wrap\">{level_comparison_summary.to_html(index=False)}</div>')
html_parts.append('</body></html>')
html_path.write_text(''.join(html_parts), encoding='utf-8')
print('Exported HTML:', html_path, 'with', len(discogem_results), 'test results')
"""
        ),
    ]
    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.12'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate DiscoGeM dashboard notebook')
    parser.add_argument('--run-root', default='outputs/discogem/screen', help='Screen run directory to analyze.')
    parser.add_argument('--result-root', default='outputs/discogem/screen', help='Screen result root; results are nested under each run.')
    parser.add_argument('--multilevel-result-root', default='outputs/discogem/screen/multilevel', help='Screen multilevel result root.')
    parser.add_argument('--discogem-path', default='data/external/DiscoGeM/DiscoGeM 2.0/DiscoGeM2.0_annotation.tgz', help='Path to DiscoGeM 2.0 annotations archive.')
    parser.add_argument('--discogem-language', default='en', help='Language slice represented in the dashboard.')
    parser.add_argument('--output-notebook', default='notebooks/discogem_experiment_dashboard.ipynb', help='Notebook path to write.')
    args = parser.parse_args()

    output_path = Path(args.output_notebook)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook(args.run_root, args.result_root, args.multilevel_result_root, args.discogem_path, args.discogem_language)
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote notebook to {output_path}')


if __name__ == '__main__':
    main()
