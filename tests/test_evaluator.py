import json

from hlv_toolkits.data.schemas import PredictionRecord, TextPairDistributionSample
from hlv_toolkits.eval import Evaluator
from hlv_toolkits.scripts.evaluate import load_ground_truth


def test_distribution_evaluation_reports_all_supported_metrics():
    ground_truth = [
        TextPairDistributionSample(id="a", task="text_pair_label_distribution", label=0, human_dist=[0.7, 0.2, 0.1]),
        TextPairDistributionSample(id="b", task="text_pair_label_distribution", label=1, human_dist=[0.1, 0.8, 0.1]),
    ]
    predictions = [
        PredictionRecord(id="a", task="text_pair_label_distribution", outputs={"pred": 0, "probs": [0.6, 0.3, 0.1]}),
        PredictionRecord(id="b", task="text_pair_label_distribution", outputs={"pred": 1, "probs": [0.2, 0.7, 0.1]}),
    ]

    metrics = Evaluator().evaluate(predictions, ground_truth).metrics

    assert set(metrics) == {
        "accuracy",
        "tvd",
        "jsd",
        "pojsd",
        "kl",
        "soft_micro_f1",
        "soft_macro_f1",
        "distance_correlation",
        "l2",
        "ce",
    }


def test_external_ground_truth_supports_optional_metadata(tmp_path):
    ground_truth_path = tmp_path / "ground_truth.jsonl"
    ground_truth_path.write_text(
        json.dumps(
            {
                "id": "a",
                "label": 1,
                "human_dist": [0.1, 0.8, 0.1],
                "producer_note": "ignored",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    ground_truth = load_ground_truth(ground_truth_path)

    assert len(ground_truth) == 1
    assert ground_truth[0].id == "a"
    assert ground_truth[0].label == 1
    assert ground_truth[0].human_dist == [0.1, 0.8, 0.1]
