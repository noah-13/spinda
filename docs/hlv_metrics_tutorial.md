# HLV Metrics Tutorial (Intuition First)

This tutorial helps users of this toolkit build intuition for HLV metrics with concrete toy data.

Goal:
- Understand when each metric is **minimum**.
- Understand when each metric is **maximum** (or very large).
- Understand what “small” vs “large” means in practice.

All examples below call the real implementations in `hlv_toolkits.eval.metrics`.

## 1. Quick Metric Map

| Metric | Better direction | Typical range in this toolkit | Minimum means | Maximum / very large means |
|---|---|---|---|---|
| `tvd` | lower is better | `[0, 1]` | exact match | fully disjoint mass |
| `jsd` (with `base=2`) | lower is better | `[0, 1]` | exact match | maximally different distributions |
| `kl(human || pred)` | lower is better | `[0, +inf)` theoretical; clipped in code | exact match | model puts tiny/zero mass where human mass is high |
| `cross_entropy(human, pred)` | lower is better | `[H(human), +inf)` | equals human entropy when pred==human | very harsh for confident wrong predictions |
| `euclidean_distance` | lower is better | `[0, sqrt(2)]` for 3-way simplex | exact match | opposite one-hot corners |
| `soft_micro_f1` | higher is better | `[0, 1]` | no overlap | exact overlap |
| `soft_macro_f1` | higher is better | `[0, 1]` | no overlap per class | exact overlap |
| `distance_correlation` | higher is better | `[0, 1]` | no global dependency | very strong global structural dependency |

Note: for `jsd`, tiny floating-point error can appear (e.g., `2e-9` instead of exact `0`).

## 2. Mathematical Relationships

For a normalized categorical human distribution `q` and model distribution `p`:

- `TVD(q, p) = 1/2 ||q - p||_1`. In this setting it is also DistCE, and
  `L1(q, p) = 2 * TVD(q, p)`.
- Categorical soft accuracy is exactly `1 - TVD(q, p)`. Reporting it alongside
  TVD gives the same ranking with the direction reversed.
- `cross_entropy(q, p) = H(q) + KL(q || p)`. For a fixed evaluation set,
  cross-entropy and KL differ only by the human-label entropy term.
- With base-2 logarithms, `PO-JSD(q, p) = 1 - JSD(q, p)`. JSD is a divergence;
  Jensen-Shannon distance is `sqrt(JSD)`.

These are mathematical relationships, not empirical findings. They explain why
equivalent transforms should not be co-primary metrics. Observed correlations
between distinct metrics are dataset-dependent; see the
[evaluation metric guide](evaluation_metrics.md#correlation-evidence) for the
paper's correlation analysis and its reporting implications.

## 3. Controlled Toy Data

We use one human matrix and 5 prediction variants:
- `identical`: exactly equal to human.
- `small_shift`: slightly perturbed from human.
- `confident_wrong`: often places mass on the opposite class.
- `uniform`: predicts `[1/3, 1/3, 1/3]` for every sample.
- `row_permuted`: uses real human rows but assigned to the wrong instances.

```python
import numpy as np
from hlv_toolkits.eval.metrics import (
    compute_tvd,
    compute_jsd,
    compute_kl,
    compute_cross_entropy,
    compute_euclidean_distance,
    compute_soft_micro_f1,
    compute_soft_macro_f1,
    compute_distance_correlation,
)

human = np.array([
    [0.90, 0.08, 0.02],
    [0.70, 0.20, 0.10],
    [0.40, 0.40, 0.20],
    [0.34, 0.33, 0.33],
    [0.10, 0.20, 0.70],
    [0.02, 0.08, 0.90],
], dtype=float)

pred_identical = human.copy()
pred_small_shift = np.array([
    [0.85, 0.11, 0.04],
    [0.64, 0.24, 0.12],
    [0.36, 0.44, 0.20],
    [0.30, 0.36, 0.34],
    [0.14, 0.22, 0.64],
    [0.05, 0.11, 0.84],
], dtype=float)
pred_confident_wrong = np.array([
    [0.02, 0.08, 0.90],
    [0.10, 0.20, 0.70],
    [0.20, 0.40, 0.40],
    [0.33, 0.33, 0.34],
    [0.70, 0.20, 0.10],
    [0.90, 0.08, 0.02],
], dtype=float)
pred_uniform = np.full_like(human, 1/3)
pred_row_permuted = human[[5, 4, 3, 2, 1, 0]]

cases = {
    "identical": pred_identical,
    "small_shift": pred_small_shift,
    "confident_wrong": pred_confident_wrong,
    "uniform": pred_uniform,
    "row_permuted": pred_row_permuted,
}

for name, pred in cases.items():
    print("\n", name)
    print("tvd", compute_tvd(pred, human).mean())
    print("jsd", compute_jsd(pred, human, base=2).mean())
    print("kl", compute_kl(human, pred).mean())
    print("ce", compute_cross_entropy(human, pred).mean())
    print("l2", compute_euclidean_distance(pred, human).mean())
    print("soft_micro_f1", compute_soft_micro_f1(pred, human))
    print("soft_macro_f1", compute_soft_macro_f1(pred, human))
    print("distance_correlation", compute_distance_correlation(pred, human))
```

## 4. Example Output (From This Repo)

| Case | `tvd` | `jsd` | `kl` | `l2` | `soft_micro_f1` | `distance_correlation` |
|---|---:|---:|---:|---:|---:|---:|
| identical | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| small_shift | 0.0517 | 0.0574 | 0.0094 | 0.0654 | 0.9483 | 0.9989 |
| confident_wrong | 0.5283 | 0.5346 | 1.5290 | 0.7472 | 0.4717 | 1.0000 |
| uniform | 0.3344 | 0.3080 | 0.3474 | 0.4119 | 0.6656 | 0.0000 |
| row_permuted | 0.5367 | 0.5379 | 1.5205 | 0.7508 | 0.4633 | 0.9894 |

How to read this quickly:
- `identical` gives best values for pointwise distance/overlap metrics.
- `small_shift` is slightly worse than `identical`, as expected.
- `confident_wrong` and `row_permuted` are much worse on pointwise metrics.
- `uniform` is mediocre on pointwise metrics, but has `distance_correlation ~= 0`.

## 5. Exact Extremes (Single Sample)

Use one-hot opposite predictions to see maxima clearly:

```python
import numpy as np
from hlv_toolkits.eval.metrics import (
    compute_tvd,
    compute_jsd,
    compute_kl,
    compute_cross_entropy,
    compute_euclidean_distance,
    compute_soft_micro_f1,
)

human = np.array([[1.0, 0.0, 0.0]])
pred  = np.array([[0.0, 1.0, 0.0]])

print("tvd", compute_tvd(pred, human)[0])                    # 1.0 (max)
print("jsd", compute_jsd(pred, human, base=2)[0])            # ~1.0 (max)
print("kl", compute_kl(human, pred)[0])                      # ~27.63 with epsilon=1e-12
print("cross_entropy", compute_cross_entropy(human, pred)[0])# ~27.63
print("l2", compute_euclidean_distance(pred, human)[0])      # sqrt(2) (max on 3-way simplex)
print("soft_micro_f1", compute_soft_micro_f1(pred, human))   # 0.0 (min)
```

## 6. “Large vs Small” Interpretation Rules

- `tvd/jsd/l2`:
  - `0` means exact match.
  - Closer to the upper bound means stronger per-sample mismatch.
- `kl/cross_entropy`:
  - Near `0` (for KL) means good match.
  - Big values usually indicate confident wrong probabilities.
- `soft_micro_f1/soft_macro_f1`:
  - Near `1` means strong overlap.
  - Near `0` means almost no overlap.
- `distance_correlation`:
  - Measures global geometry agreement, not row-by-row closeness.
  - Can stay high even when per-sample errors are large, if structural dependency is preserved.

## 7. Practical Recommendation

For paper-aligned reporting, choose a compact set based on the annotation
structure rather than reporting every available score:

- Categorical HLV: report mean `tvd` as the representative distributional
  discrepancy, with `kl` and `entropy_correlation` as complementary metrics.
  Add `--analysis` when disagreement-stratified and instance-level errors are
  relevant.
- Multi-label HLV: report conventional macro-F1 together with soft macro-F1,
  multilabel PO-JSD, and multilabel entropy correlation. Soft micro-F1 remains
  available for development selection, but it is strongly correlated with soft
  macro-F1 in the paper analyses.

`jsd`, `l2`, `cross_entropy`, and `distance_correlation` remain useful for
targeted analyses or comparison with prior work, but they are not part of the
paper's compact primary reporting sets.

For the empirical correlation figures and the complete metric-selection
rationale, see the [evaluation metric guide](evaluation_metrics.md). For the
visual version of the toy examples, see `notebooks/metrics_tutorial.ipynb`.
