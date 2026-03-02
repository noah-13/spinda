import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from nli_toolkits.visualization import (
    get_boundaries,
    plot_bootstrap_bounds,
    plot_ternary_axes,
    plot_ternary_bounds,
    save_distribution_ternary_plot,
    save_distce_plot,
    save_interactive_distribution_ternary_plot,
    save_ternary_plot,
    save_tvd_plot,
)


def _valid_pi_theta() -> tuple[np.ndarray, np.ndarray]:
    pi = np.array([0.34, 0.33, 0.33], dtype=float)
    theta = np.array(
        [
            [0.75, 0.15, 0.10],
            [0.10, 0.80, 0.10],
            [0.15, 0.10, 0.75],
        ],
        dtype=float,
    )
    return pi, theta


def test_plot_ternary_axes_runs():
    pytest.importorskip("ternary")
    fig, _ = plot_ternary_axes()
    fig.clf()


def test_get_boundaries_valid_shape_and_finite():
    pi, theta = _valid_pi_theta()
    points = get_boundaries(pi, theta, scale=100.0)
    assert len(points) == 4
    for point in points:
        assert point.shape == (3,)
        assert np.isfinite(point).all()


@pytest.mark.parametrize(
    "pi,theta,error_msg",
    [
        (np.array([0.5, 0.5]), np.eye(3), "shape"),
        (np.array([0.5, -0.1, 0.6]), np.eye(3), "non-negative"),
        (np.array([0.5, 0.2, 0.2]), np.eye(3), "sum to 1"),
    ],
)
def test_get_boundaries_invalid_inputs_raise(pi, theta, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        get_boundaries(pi, theta)


def test_plot_ternary_bounds_and_bootstrap_runs():
    pytest.importorskip("ternary")
    pi, theta = _valid_pi_theta()
    fig, tax = plot_ternary_axes(scale=100.0)
    plot_ternary_bounds(tax, pi=pi, theta=theta, scale=100.0)

    bounds_list = [
        get_boundaries(pi, theta, scale=100.0),
        get_boundaries(np.array([0.35, 0.33, 0.32]), theta, scale=100.0),
        get_boundaries(np.array([0.33, 0.34, 0.33]), theta, scale=100.0),
    ]
    plot_bootstrap_bounds(tax, bounds_list)
    fig.clf()


def test_save_ternary_plot_and_distce_alias(tmp_path):
    pytest.importorskip("ternary")
    output = tmp_path / "ternary.png"
    pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2]], dtype=float)
    human = np.array([[0.6, 0.3, 0.1], [0.2, 0.6, 0.2]], dtype=float)

    ok = save_ternary_plot([pred, human], output_path=output, dataset_names=["Model", "Human"])
    assert ok is True
    assert output.exists()

    tvd = np.array([0.1, 0.2], dtype=float)
    distce_out = tmp_path / "distce.png"
    assert save_distce_plot is save_tvd_plot
    assert save_distce_plot(tvd, distce_out) is True
    assert distce_out.exists()


@pytest.mark.parametrize("source", ["model", "human", "both"])
def test_save_distribution_ternary_plot_sources(tmp_path, source):
    pytest.importorskip("ternary")
    output = tmp_path / f"ternary_{source}.png"
    pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2]], dtype=float)
    human = np.array([[0.6, 0.3, 0.1], [0.2, 0.6, 0.2]], dtype=float)

    ok = save_distribution_ternary_plot(
        model_distributions=pred,
        human_distributions=human,
        output_path=output,
        distribution_source=source,
    )
    assert ok is True
    assert output.exists()


def test_save_distribution_ternary_plot_invalid_source_raises(tmp_path):
    output = tmp_path / "ternary_invalid.png"
    pred = np.array([[0.7, 0.2, 0.1]], dtype=float)
    human = np.array([[0.6, 0.3, 0.1]], dtype=float)

    with pytest.raises(ValueError, match="distribution_source"):
        save_distribution_ternary_plot(
            model_distributions=pred,
            human_distributions=human,
            output_path=output,
            distribution_source="invalid",
        )


@pytest.mark.parametrize("source", ["model", "human", "both"])
def test_save_interactive_distribution_ternary_plot_sources(tmp_path, source):
    pytest.importorskip("plotly")
    output = tmp_path / f"ternary_{source}.html"
    pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2]], dtype=float)
    human = np.array([[0.6, 0.3, 0.1], [0.2, 0.6, 0.2]], dtype=float)

    ok = save_interactive_distribution_ternary_plot(
        model_distributions=pred,
        human_distributions=human,
        output_path=output,
        distribution_source=source,
        ids=["a1", "a2"],
        premises=["premise one", "premise two"],
        hypotheses=["hyp one", "hyp two"],
    )
    assert ok is True
    assert output.exists()
    html = output.read_text(encoding="utf-8")
    assert "H-P:" in html
    assert "HJD:" in html


def test_save_interactive_distribution_ternary_plot_invalid_source_raises(tmp_path):
    pytest.importorskip("plotly")
    output = tmp_path / "ternary_invalid.html"
    pred = np.array([[0.7, 0.2, 0.1]], dtype=float)
    human = np.array([[0.6, 0.3, 0.1]], dtype=float)

    with pytest.raises(ValueError, match="distribution_source"):
        save_interactive_distribution_ternary_plot(
            model_distributions=pred,
            human_distributions=human,
            output_path=output,
            distribution_source="invalid",
        )
