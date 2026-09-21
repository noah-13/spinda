import json

import pytest

from hlv_toolkits.data.schemas import (
    MultilevelSample,
    PredictionRecord,
    SingleTextMultilevelSample,
    SingleTextMultilabelDistributionSample,
    TextPairDistributionSample,
)
from hlv_toolkits.eval import DEFAULT_CATEGORICAL_DISTRIBUTION_METRICS, Evaluator
from hlv_toolkits.scripts.evaluate import _build_multilevel_eval_inputs, load_ground_truth


def test_distribution_evaluation_reports_only_standard_metrics_by_default():
    ground_truth = [
        TextPairDistributionSample(id="a", task="text_pair_label_distribution", label=0, human_dist=[0.7, 0.2, 0.1]),
        TextPairDistributionSample(id="b", task="text_pair_label_distribution", label=1, human_dist=[0.1, 0.8, 0.1]),
    ]
    predictions = [
        PredictionRecord(id="a", task="text_pair_label_distribution", outputs={"pred": 0, "probs": [0.6, 0.3, 0.1]}),
        PredictionRecord(id="b", task="text_pair_label_distribution", outputs={"pred": 1, "probs": [0.2, 0.7, 0.1]}),
    ]

    metrics = Evaluator().evaluate(predictions, ground_truth).metrics

    assert tuple(metrics) == DEFAULT_CATEGORICAL_DISTRIBUTION_METRICS


def test_distribution_evaluation_supports_opt_in_derived_metric_names():
    ground_truth = [
        TextPairDistributionSample(
            id="a", task="text_pair_label_distribution", label=0,
            human_dist=[0.7, 0.2, 0.1],
        ),
    ]
    predictions = [
        PredictionRecord(
            id="a", task="text_pair_label_distribution",
            outputs={"pred": 0, "probs": [0.6, 0.3, 0.1]},
        ),
    ]

    metrics = Evaluator(
        distribution_metrics=("tvd", "l1", "jsd", "pojsd", "soft_accuracy", "soft_micro_f1")
    ).evaluate(predictions, ground_truth).metrics

    assert metrics["l1"] == pytest.approx(2 * metrics["tvd"])
    assert metrics["pojsd"] == pytest.approx(1 - metrics["jsd"])
    assert metrics["soft_accuracy"] == pytest.approx(metrics["soft_micro_f1"])
    assert metrics["soft_accuracy"] == pytest.approx(1 - metrics["tvd"])


def test_external_ground_truth_supports_optional_metadata(tmp_path):
    ground_truth_path = tmp_path / "ground_truth.json"
    ground_truth_path.write_text(
        json.dumps([
            {
                "id": "a",
                "label": 1,
                "human_dist": [0.1, 0.8, 0.1],
                "producer_note": "ignored",
            }
        ]),
        encoding="utf-8",
    )

    ground_truth = load_ground_truth(ground_truth_path)

    assert len(ground_truth) == 1
    assert ground_truth[0].id == "a"
    assert ground_truth[0].label == 1
    assert ground_truth[0].human_dist == [0.1, 0.8, 0.1]


def test_multilabel_evaluation_and_prediction_coverage_validation():
    ground_truth = [SingleTextMultilabelDistributionSample(id="a", task="mfrc", labels=[1, 0], human_probs=[0.75, 0.25])]
    predictions = [PredictionRecord(id="a", task="mfrc", outputs={"pred": [1, 0], "probs": [0.75, 0.25]})]
    metrics = Evaluator().evaluate(predictions, ground_truth).metrics
    assert metrics["accuracy"] == 1.0
    assert metrics["micro_f1"] == 1.0
    assert metrics["macro_f1"] == 0.5
    assert "tvd" not in metrics
    assert metrics["soft_micro_f1"] == 1.0
    assert metrics["soft_macro_f1"] == 1.0
    assert metrics["multilabel_pojsd"] == 1.0
    assert metrics["multilabel_entropy_correlation"] == 0.0

    with pytest.raises(ValueError, match="duplicate IDs"):
        Evaluator().evaluate(predictions + predictions, ground_truth)


def test_multilevel_evaluation_inputs_support_text_pair_and_single_text_samples():
    predictions = [
        PredictionRecord(
            id="sample-1",
            task="multilevel",
            outputs={
                "dimensions": {
                    "level1": {"probs": [0.25, 0.75], "pred": 1},
                    "level2": {"probs": [0.8, 0.2], "pred": 0},
                    "level3": {"probs": [0.4, 0.6], "pred": 1},
                }
            },
        )
    ]
    human_dists = {"level1": [0.25, 0.75], "level2": [0.8, 0.2], "level3": [0.4, 0.6]}

    for sample in (
        MultilevelSample(id="sample-1", task="multilevel", hard_labels={"level1": 1, "level2": 0, "level3": 1}, human_dists=human_dists),
        SingleTextMultilevelSample(id="sample-1", task="multilevel", hard_labels={"level1": 1, "level2": 0, "level3": 1}, human_dists=human_dists),
    ):
        level_predictions, level_ground_truth = _build_multilevel_eval_inputs(
            predictions, [sample], "level1"
        )
        result = Evaluator().evaluate(level_predictions, level_ground_truth)

        assert result.metrics["accuracy"] == 1.0
        assert result.metrics["tvd"] == 0.0
