import numpy as np
import pytest

from spinda.eval.disagreement import (
    assign_disagreement_strata,
    compute_instance_distribution_errors,
    disagreement_stratified_evaluation,
    instance_error_records,
    instance_error_summary,
)


def test_disagreement_strata_and_instance_error_summary():
    human = np.array([[1.0, 0.0], [0.75, 0.25], [0.5, 0.5]])
    prediction = np.array([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])
    result = disagreement_stratified_evaluation(
        prediction, human, np.array([0, 0, 1]), np.array([0, 0, 0])
    )

    assert [result["groups"][name]["n"] for name in ("low", "medium", "high")] == [1, 1, 1]
    assert result["groups"]["low"]["metrics"]["tvd"] == pytest.approx(0.1)
    assert result["groups"]["high"]["metrics"]["tvd"] == pytest.approx(0.3)
    summary = instance_error_summary(compute_instance_distribution_errors(prediction, human)["tvd"], metric="tvd")
    assert summary["n"] == 3
    assert summary["p90"] > summary["median"]


def test_disagreement_supports_explicit_boundaries_and_instance_records():
    human = np.array([[1.0, 0.0], [0.75, 0.25], [0.5, 0.5]])
    prediction = np.array([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])
    entropy = np.array([0.0, 0.5, 1.0])

    strata, metadata = assign_disagreement_strata(
        entropy, num_groups=4, boundaries=[0.2, 0.6, 0.8]
    )
    assert list(strata) == ["group_1", "group_2", "group_4"]
    assert metadata["method"] == "explicit_boundaries"
    assert metadata["group_names"] == ["group_1", "group_2", "group_3", "group_4"]

    rows = instance_error_records(
        ["a", "b", "c"], prediction, human, np.array([0, 0, 1]), np.array([0, 0, 0]),
        boundaries=[0.2, 0.6],
    )
    assert rows[0]["id"] == "a"
    assert rows[2]["is_correct"] is False
    assert rows[2]["tvd"] == pytest.approx(0.3)
