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
from hlv_toolkits.scripts.evaluate import (
    _build_multilevel_eval_inputs,
    _filter_metrics,
    _infer_prediction_kind,
    _load_merged_json_configs,
    load_human_labels,
)


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


def test_human_labels_test_file_supports_annotation_votes(tmp_path):
    ground_truth_path = tmp_path / "test.json"
    ground_truth_path.write_text(
        json.dumps([{
            "id": "a", "text_a": "premise", "text_b": "hypothesis",
            "annotation_labels": [0, 1, 1],
        }]),
        encoding="utf-8",
    )

    ground_truth = load_human_labels(ground_truth_path, "categorical")

    assert len(ground_truth) == 1
    assert ground_truth[0].id == "a"
    assert ground_truth[0].label == 1
    assert ground_truth[0].human_dist == [pytest.approx(1 / 3), pytest.approx(2 / 3)]


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


def test_evaluation_infers_prediction_contract_and_filters_metrics(tmp_path):
    categorical = [PredictionRecord(id="a", task="ignored", outputs={"pred": 1, "probs": [0.2, 0.8]})]
    multilabel = [PredictionRecord(id="a", task="ignored", outputs={"pred": [1, 0], "probs": [0.7, 0.2]})]
    multilevel = [PredictionRecord(id="a", task="ignored", outputs={"dimensions": {"sentiment": {"pred": 1, "probs": [0.1, 0.9]}}})]

    assert _infer_prediction_kind(categorical) == "categorical"
    assert _infer_prediction_kind(multilabel) == "multilabel"
    assert _infer_prediction_kind(multilevel) == "multilevel"
    assert _filter_metrics({"accuracy": 1.0, "tvd": 0.2}, ["tvd"]) == {"tvd": 0.2}

    defaults = tmp_path / "defaults.json"
    run = tmp_path / "run.json"
    defaults.write_text(json.dumps({"metrics": ["accuracy"], "ternary_source": "model"}), encoding="utf-8")
    run.write_text(json.dumps({"predictions": "predictions.json", "human_labels": "test.json", "metrics": ["tvd"]}), encoding="utf-8")
    assert _load_merged_json_configs([str(defaults), str(run)]) == {
        "metrics": ["tvd"], "ternary_source": "model",
        "predictions": "predictions.json", "human_labels": "test.json",
    }
