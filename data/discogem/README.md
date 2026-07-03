# DiscoGeM Data Guide

This document describes the DiscoGeM data layout used by HLV Toolkits after preprocessing.

## Canonical Processed File

Run:

```bash
bash scripts/preprocess.sh discogem
```

This produces the canonical merged file:

```text
data/processed/discogem.jsonl
```

The file contains one JSONL record per example with multi-level labels attached.

## Processed Record Format

Each line looks like this:

```json
{
  "id": "...",
  "task": "discogem",
  "split": "train",
  "source": "discogem",
  "meta": {},
  "premise": "...",
  "hypothesis": "...",
  "hard_labels": {
    "level1": 0,
    "level2": 0,
    "level3": 17
  },
  "human_dists": {
    "level1": [ ... ],
    "level2": [ ... ],
    "level3": [ ... ]
  },
  "_schema": "DiscoGeMMultiLevelSample"
}
```

Meaning:

- `hard_labels[levelX]` is the integer class id for that level.
- `human_dists[levelX]` is the normalized human label distribution for that level.
- `level3` is the most fine-grained label space.
- `level2` and `level1` are deterministic rollups from `level3`.
- In the raw archive, the hard label comes from `WAWA_en` and the soft distribution comes from `WAWA_dist_en`.

## Label Levels

### Level 1

Level 1 is the coarsest label space.

| id | label |
|---:|---|
| 0 | temporal |
| 1 | contingency |
| 2 | comparison |
| 3 | expansion |
| 4 | norel |

### Level 2

Level 2 is the mid-level label space.

| id | label |
|---:|---|
| 0 | asynchronous |
| 1 | cause |
| 2 | concession |
| 3 | condition |
| 4 | conjunction |
| 5 | contrast |
| 6 | disjunction |
| 7 | equivalence |
| 8 | exception |
| 9 | instantiation |
| 10 | level-of-detail |
| 11 | manner |
| 12 | norel |
| 13 | purpose |
| 14 | similarity |
| 15 | substitution |
| 16 | synchronous |

### Level 3

Level 3 is the finest label space used by the raw DiscoGeM annotations and the soft distributions.

| id | label |
|---:|---|
| 0 | asynchronous |
| 1 | cause |
| 2 | concession |
| 3 | condition |
| 4 | conjunction |
| 5 | contrast |
| 6 | disjunction |
| 7 | equivalence |
| 8 | exception |
| 9 | instantiation |
| 10 | level-of-detail |
| 11 | manner |
| 12 | norel |
| 13 | purpose |
| 14 | similarity |
| 15 | substitution |
| 16 | synchronous |
| 17 | precedence |
| 18 | succession |
| 19 | reason |
| 20 | result |
| 21 | arg1-as-cond |
| 22 | arg2-as-cond |
| 23 | arg1-as-negcond |
| 24 | arg2-as-negcond |
| 25 | arg1-as-goal |
| 26 | arg2-as-goal |
| 27 | arg1-as-manner |
| 28 | arg2-as-manner |
| 29 | arg1-as-detail |
| 30 | arg2-as-detail |
| 31 | arg1-as-instance |
| 32 | arg2-as-instance |
| 33 | arg1-as-subst |
| 34 | arg2-as-subst |
| 35 | arg1-as-excpt |
| 36 | arg2-as-excpt |
| 37 | arg1-as-denier |
| 38 | arg2-as-denier |

## Mapping From Level 3 To Level 2

The following Level 3 labels collapse into Level 2 labels:

- `precedence`, `succession` -> `asynchronous`
- `reason`, `result` -> `cause`
- `arg1-as-goal`, `arg2-as-goal` -> `purpose`
- `arg1-as-cond`, `arg2-as-cond`, `arg1-as-negcond`, `arg2-as-negcond` -> `condition`
- `arg1-as-denier`, `arg2-as-denier` -> `concession`
- `arg1-as-instance`, `arg2-as-instance` -> `instantiation`
- `arg1-as-detail`, `arg2-as-detail` -> `level-of-detail`
- `arg1-as-excpt`, `arg2-as-excpt` -> `exception`
- `arg1-as-manner`, `arg2-as-manner` -> `manner`
- `arg1-as-subst`, `arg2-as-subst` -> `substitution`

All other Level 3 labels keep the same name at Level 2.

## Mapping From Level 2 To Level 1

Level 2 labels roll up to Level 1 as follows:

- `temporal`: `asynchronous`, `synchronous`
- `contingency`: `cause`, `purpose`, `condition`
- `comparison`: `concession`, `contrast`, `similarity`
- `expansion`: `equivalence`, `instantiation`, `level-of-detail`, `conjunction`, `disjunction`, `exception`, `manner`, `substitution`
- `norel`: `norel`

## How To Use It

Train from the processed file with:

```bash
uv run python -m hlv_toolkits.scripts.train \
  --data_source processed \
  --processed_task discogem \
  --processed_data_dir data/processed/discogem.jsonl
```

Choose the target label level with the CLI flag used by your experiment script.

- `level1`: 5-way classification
- `level2`: 17-way classification
- `level3`: 39-way classification
- `all`: multi-level sample object containing all levels
