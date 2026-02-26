from pathlib import Path
from typing import Sequence

import numpy as np


def save_tvd_plot(
    tvd: np.ndarray,
    output_path: Path,
    title: str | None = None,
    bins: int = 30,
) -> bool:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return False

    values = tvd.astype(float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with plt.style.context("seaborn-v0_8-darkgrid"):
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    except Exception:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

    sns.histplot(
        values,
        binwidth=1 / bins,
        binrange=(0, 1),
        kde=True,
        stat="probability",
        ax=axes,
    )

    axes.set(xlabel="TVD")
    axes.set(title=title)
    axes.set(ylim=(0, 0.135))

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return True

"""
Ternary plots for visualizing distributional predictions and human distributions.
From Beiduo's seeing the small through the big https://arxiv.org/abs/2406.17600
"""

def _normalize_to_ternary_scale(
    points: np.ndarray,
    scale: float,
) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Each distribution must have shape [N, 3].")
    if arr.size == 0:
        return arr
    if not np.isfinite(arr).all():
        raise ValueError("Distribution contains non-finite values.")
    if (arr < 0).any():
        raise ValueError("Distribution contains negative values.")

    row_sums = arr.sum(axis=1)
    if np.allclose(row_sums, 1.0, atol=1e-3):
        return arr * scale
    if np.allclose(row_sums, scale, atol=1e-2):
        return arr
    if arr.max() <= 1.0 + 1e-6:
        return arr * scale
    return arr


def _scale_center(points: np.ndarray, scale_factor: float) -> np.ndarray:
    center = np.mean(points, axis=0)
    return (points - center) * scale_factor + center


def save_ternary_plot(
    distributions: Sequence[np.ndarray | Sequence[Sequence[float]]],
    output_path: Path,
    dataset_names: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    markers: Sequence[str] | None = None,
    title: str | None = None,
    scale: float = 100.0,
    grid_multiple: float = 10.0,
    scale_up_index: int | None = None,
    scale_factor: float = 1.0,
) -> bool:
    """
    Save one ternary subplot per dataset.

    Notes:
    - Each distribution must be an [N, 3] matrix ordered as
      [entailment, neutral, contradiction].
    - Inputs can be probability rows summing to 1 or percentages summing to `scale`.
    """
    try:
        import matplotlib.pyplot as plt
        import ternary
    except ImportError:
        return False

    if len(distributions) == 0:
        return True

    n_panels = len(distributions)
    default_names = [f"Dataset {i + 1}" for i in range(n_panels)]
    default_colors = ["k", "tab:orange", "tab:purple", "tab:blue", "tab:green"]
    default_markers = ["D", "v", "<", "o", "s"]

    names = list(dataset_names) if dataset_names is not None else default_names
    if len(names) < n_panels:
        names.extend(default_names[len(names):n_panels])

    panel_colors = list(colors) if colors is not None else default_colors
    if len(panel_colors) < n_panels:
        panel_colors.extend(default_colors * (n_panels - len(panel_colors)))

    panel_markers = list(markers) if markers is not None else default_markers
    if len(panel_markers) < n_panels:
        panel_markers.extend(default_markers * (n_panels - len(panel_markers)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    axes_arr = np.atleast_1d(axes)

    for i, data in enumerate(distributions):
        points = _normalize_to_ternary_scale(np.asarray(data, dtype=float), scale=scale)
        if scale_up_index is not None and i == scale_up_index and scale_factor != 1.0:
            points = _scale_center(points, scale_factor=scale_factor)

        ax = axes_arr[i]
        _, tax = ternary.figure(ax=ax, scale=scale)
        tax.boundary(linewidth=2.0)
        tax.gridlines(multiple=grid_multiple, color="grey")
        tax.scatter(points, marker=panel_markers[i], color=panel_colors[i], label=names[i], vmin=None, vmax=None)
        tax.ticks(axis="lbr", linewidth=1, multiple=grid_multiple)
        tax.clear_matplotlib_ticks()
        tax.get_axes().axis("off")
        tax.left_axis_label("Entailment", fontsize=12, offset=0.14)
        tax.right_axis_label("Neutral", fontsize=12, offset=0.14)
        tax.bottom_axis_label("Contradiction", fontsize=12, offset=0.06)
        tax.legend(fontsize=10)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True
