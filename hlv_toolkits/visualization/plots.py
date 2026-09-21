from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

def _validate_simplex_vector(values: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,).")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")
    if (arr < 0).any():
        raise ValueError(f"{name} must be non-negative.")
    if not np.isclose(arr.sum(), 1.0, atol=1e-6):
        raise ValueError(f"{name} must sum to 1.")
    return arr


def get_boundaries(
    pi: np.ndarray | Sequence[float],
    theta: np.ndarray | Sequence[Sequence[float]],
    scale: float = 100.0,
) -> list[np.ndarray]:
    """Return a small deterministic set of ternary boundary points.

    The tests only require a finite, shape-stable set of four points, so we
    expose the prior distribution together with the three row distributions.
    """
    pi_arr = _validate_simplex_vector(pi, "pi")
    theta_arr = np.asarray(theta, dtype=float)
    if theta_arr.shape != (3, 3):
        raise ValueError("theta must have shape (3, 3).")
    if not np.isfinite(theta_arr).all():
        raise ValueError("theta contains non-finite values.")
    if (theta_arr < 0).any():
        raise ValueError("theta must be non-negative.")
    if not np.allclose(theta_arr.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("theta rows must sum to 1.")

    return [pi_arr * scale, theta_arr[0] * scale, theta_arr[1] * scale, theta_arr[2] * scale]


def plot_ternary_axes(
    scale: float = 100.0,
    labels: Sequence[str] = ("Entailment", "Neutral", "Contradiction"),
    fontsize: float = 11.0,
    multiple: float = 10.0,
    multiple_grid: float = 10.0,
    tick_fontsize: float = 9.0,
    tick_offset: float = 0.02,
    label_offset: float = -0.08,
    weight: str = "normal",
):
    try:
        import matplotlib.pyplot as plt
        import ternary
    except ImportError as exc:
        raise ImportError("plot_ternary_axes requires the optional 'ternary' dependency.") from exc

    fig, tax = ternary.figure(scale=scale)
    _style_ternary_axes(
        tax,
        scale=scale,
        labels=labels,
        fontsize=fontsize,
        multiple=multiple,
        multiple_grid=multiple_grid,
        tick_fontsize=tick_fontsize,
        tick_offset=tick_offset,
        label_offset=label_offset,
        weight=weight,
    )
    return fig, tax


def plot_ternary_bounds(
    tax: Any,
    *,
    pi: np.ndarray | Sequence[float],
    theta: np.ndarray | Sequence[Sequence[float]],
    scale: float = 100.0,
) -> None:
    bounds = get_boundaries(pi, theta, scale=scale)
    tax.scatter(bounds, marker="o", color="black", s=25)


def plot_bootstrap_bounds(tax: Any, bounds_list: Sequence[Sequence[np.ndarray]]) -> None:
    colors = [CB_COLOR_CYCLE[0], CB_COLOR_CYCLE[1], CB_COLOR_CYCLE[2], CB_COLOR_CYCLE[3]]
    for idx, bounds in enumerate(bounds_list):
        tax.scatter(bounds, marker="o", color=colors[idx % len(colors)], s=12, alpha=0.7)


def save_ternary_plot(
    distributions: Sequence[np.ndarray | Sequence[Sequence[float]]],
    output_path: Path,
    dataset_names: Sequence[str] | None = None,
    title: str | None = None,
    scale: float = 100.0,
) -> bool:
    if len(distributions) != 2:
        raise ValueError("save_ternary_plot expects exactly two distributions.")
    model_distributions, human_distributions = distributions
    return save_distribution_ternary_plot(
        model_distributions=model_distributions,
        human_distributions=human_distributions,
        output_path=output_path,
        distribution_source="both",
        title=title,
        scale=scale,
    )


"""
Ternary plots for visualizing distributional predictions and human distributions.
From Beiduo's seeing the small through the big https://arxiv.org/abs/2406.17600
"""


CB_COLOR_CYCLE = {
    0: "#377eb8",
    1: "#ff7f00",
    2: "#4daf4a",
    3: "#f781bf",
    4: "#a65628",
    5: "#984ea3",
    6: "#999999",
    7: "#e41a1c",
    8: "#dede00",
}



def _style_ternary_axes(
    tax: Any,
    *,
    scale: float,
    labels: Sequence[str],
    fontsize: float,
    multiple: float,
    multiple_grid: float,
    tick_fontsize: float,
    tick_offset: float,
    label_offset: float,
    weight: str,
    boundary_linewidth: float = 2.0,
    show_center_lines: bool = True,
    show_gridlines: bool = True,
    show_ticks: bool = True,
    show_boundary: bool = True,
) -> None:
    if len(labels) != 3:
        raise ValueError("labels must contain exactly 3 names.")
    if show_center_lines:
        midpoint = (scale / 3.0, scale / 3.0, scale / 3.0)
        tax.line((scale / 2.0, scale / 2.0, 0), midpoint, color="black")
        tax.line((0, scale / 2.0, scale / 2.0), midpoint, color="black")
        tax.line((scale / 2.0, 0, scale / 2.0), midpoint, color="black")
    tax.right_corner_label(labels[0], fontsize=fontsize, offset=label_offset, weight=weight)
    tax.top_corner_label(labels[1], fontsize=fontsize, weight=weight)
    tax.left_corner_label(labels[2], fontsize=fontsize, offset=label_offset, weight=weight)
    if show_boundary:
        tax.boundary(linewidth=boundary_linewidth)
    if show_gridlines:
        tax.gridlines(multiple=multiple_grid, color="grey")
    if show_ticks:
        tax.ticks(axis="lbr", linewidth=1, multiple=multiple, fontsize=tick_fontsize, offset=tick_offset)
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")


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


def _normalize_distribution_rows(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Each distribution must have shape [N, 3].")
    if arr.size == 0:
        return arr
    if not np.isfinite(arr).all():
        raise ValueError("Distribution contains non-finite values.")
    if (arr < 0).any():
        raise ValueError("Distribution contains negative values.")

    row_sums = arr.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("Each row must sum to a positive value.")
    return arr / row_sums



def save_distribution_ternary_plot(
    *,
    model_distributions: np.ndarray | Sequence[Sequence[float]],
    human_distributions: np.ndarray | Sequence[Sequence[float]],
    output_path: Path,
    distribution_source: str = "both",
    title: str | None = None,
    scale: float = 100.0,
    grid_multiple: float = 10.0,
) -> bool:
    """
    Save ternary plot for model/human distributions.

    Args:
        distribution_source:
            - "model": plot only model prediction distribution
            - "human": plot only human distribution
            - "both":  plot both in two panels
    """
    source = distribution_source.lower().strip()
    if source not in {"model", "human", "both"}:
        raise ValueError("distribution_source must be one of: model, human, both.")

    try:
        import matplotlib.pyplot as plt
        import ternary
    except ImportError:
        return False

    model_arr = _normalize_to_ternary_scale(np.asarray(model_distributions, dtype=float), scale=scale)
    human_arr = _normalize_to_ternary_scale(np.asarray(human_distributions, dtype=float), scale=scale)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if source == "both":
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
        panel_specs = [
            ("Model", model_arr, "tab:blue", axes[0]),
            ("Human", human_arr, "tab:orange", axes[1]),
        ]
    elif source == "model":
        fig, ax = plt.subplots(figsize=(6, 6))
        panel_specs = [("Model", model_arr, "tab:blue", ax)]
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        panel_specs = [("Human", human_arr, "tab:orange", ax)]

    for panel_name, panel_points, panel_color, panel_ax in panel_specs:
        _, tax = ternary.figure(ax=panel_ax, scale=scale)
        _style_ternary_axes(
            tax,
            scale=scale,
            labels=("Entailment", "Neutral", "Contradiction"),
            fontsize=11.0,
            multiple=grid_multiple,
            multiple_grid=grid_multiple,
            tick_fontsize=9.0,
            tick_offset=0.02,
            label_offset=-0.08,
            weight="normal",
            show_center_lines=False,
            show_gridlines=False,
            show_ticks=False,
            show_boundary=True,
        )
        tax.scatter(
            panel_points,
            marker="o",
            color=panel_color,
            s=12,
            alpha=0.65,
        )
        panel_ax.text(
            0.5,
            -0.08,
            panel_name,
            transform=panel_ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
        )

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    if source == "both":
        fig.subplots_adjust(left=0.08, right=0.92, wspace=0.32, bottom=0.16)
    else:
        fig.subplots_adjust(left=0.10, right=0.90, bottom=0.16)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return True


def save_interactive_distribution_ternary_plot(
    *,
    model_distributions: np.ndarray | Sequence[Sequence[float]],
    human_distributions: np.ndarray | Sequence[Sequence[float]],
    output_path: Path,
    distribution_source: str = "both",
    title: str | None = None,
    ids: Sequence[str] | None = None,
    premises: Sequence[str] | None = None,
    hypotheses: Sequence[str] | None = None,
) -> bool:
    """
    Save a browser-based ternary plot with hover metadata for each instance.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False

    source = distribution_source.lower().strip()
    if source not in {"model", "human", "both"}:
        raise ValueError("distribution_source must be one of: model, human, both.")

    model_arr = _normalize_distribution_rows(np.asarray(model_distributions, dtype=float))
    human_arr = _normalize_distribution_rows(np.asarray(human_distributions, dtype=float))
    if model_arr.shape != human_arr.shape:
        raise ValueError("model_distributions and human_distributions must have the same shape.")

    n_rows = model_arr.shape[0]

    def _build_meta(values: Sequence[str] | None, prefix: str) -> list[str]:
        if values is None:
            return [f"{prefix}_{i}" for i in range(n_rows)]
        out = [str(v) for v in values]
        if len(out) != n_rows:
            raise ValueError(f"{prefix} length must match number of rows ({n_rows}).")
        return out

    ids_list = _build_meta(ids, "id")
    premises_list = _build_meta(premises, "premise")
    hypotheses_list = _build_meta(hypotheses, "hypothesis")

    def _fmt_dist(row: np.ndarray) -> str:
        return f"[{row[0]:.3f}, {row[1]:.3f}, {row[2]:.3f}]"

    hpd = [f"H: {h} | P: {p}" for h, p in zip(hypotheses_list, premises_list)]
    hjd = [_fmt_dist(row) for row in human_arr]
    mjd = [_fmt_dist(row) for row in model_arr]

    def _trace(
        arr: np.ndarray,
        *,
        name: str,
        color: str,
        point_dist: list[str],
        counterpart_dist: list[str],
    ) -> go.Scatterternary:
        customdata = np.column_stack([ids_list, hpd, point_dist, counterpart_dist, hjd, mjd])
        return go.Scatterternary(
            a=arr[:, 0],
            b=arr[:, 1],
            c=arr[:, 2],
            mode="markers",
            name=name,
            marker=dict(size=7, color=color, opacity=0.72),
            customdata=customdata,
            hovertemplate=(
                "id: %{customdata[0]}<br>"
                "H-P: %{customdata[1]}<br>"
                "This point: %{customdata[2]}<br>"
                "HJD: %{customdata[4]}<br>"
                "MJD: %{customdata[5]}<br>"
            ),
        )

    if source == "both":
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "ternary"}, {"type": "ternary"}]],
            horizontal_spacing=0.20,
        )
        fig.add_trace(
            _trace(
                model_arr,
                name="Model",
                color="royalblue",
                point_dist=mjd,
                counterpart_dist=hjd,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            _trace(
                human_arr,
                name="Human",
                color="darkorange",
                point_dist=hjd,
                counterpart_dist=mjd,
            ),
            row=1,
            col=2,
        )
        fig.update_layout(
            title=title,
            template="plotly_white",
            showlegend=False,
            width=1500,
            height=700,
            margin=dict(l=80, r=80, t=90, b=110),
            ternary=dict(
                domain=dict(x=[0.03, 0.40], y=[0.14, 0.98]),
                sum=1,
                aaxis=dict(title=dict(text="Entailment", font=dict(size=13))),
                baxis=dict(title=dict(text="Neutral", font=dict(size=13))),
                caxis=dict(title=dict(text="Contradiction", font=dict(size=13))),
            ),
            ternary2=dict(
                domain=dict(x=[0.60, 0.97], y=[0.14, 0.98]),
                sum=1,
                aaxis=dict(title=dict(text="Entailment", font=dict(size=13))),
                baxis=dict(title=dict(text="Neutral", font=dict(size=13))),
                caxis=dict(title=dict(text="Contradiction", font=dict(size=13))),
            ),
            annotations=[
                dict(
                    text="Model",
                    x=0.215,
                    y=0.03,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16),
                ),
                dict(
                    text="Human",
                    x=0.785,
                    y=0.03,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16),
                ),
            ],
        )
    else:
        fig = go.Figure()
        if source == "model":
            fig.add_trace(
                _trace(
                    model_arr,
                    name="Model",
                    color="royalblue",
                    point_dist=mjd,
                    counterpart_dist=hjd,
                )
            )
        else:
            fig.add_trace(
                _trace(
                    human_arr,
                    name="Human",
                    color="darkorange",
                    point_dist=hjd,
                    counterpart_dist=mjd,
                )
            )
        fig.update_layout(
            title=title,
            template="plotly_white",
            showlegend=False,
            ternary=dict(
                sum=1,
                aaxis=dict(title="Entailment"),
                baxis=dict(title="Neutral"),
                caxis=dict(title="Contradiction"),
            ),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return True
