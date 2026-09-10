import json

import pytest

from hlv_toolkits.data.readers.multilevel_reader import TextPairMultilevelJSONLReader
from hlv_toolkits.data.readers.single_text_multilevel_reader import SingleTextMultilevelJSONLReader


def test_multilevel_reader_uses_manifest_level_labels(tmp_path):
    labels = {
        "level1": ["a", "b"],
        "level2": ["c", "d", "e"],
        "level3": ["f", "g"],
    }
    (tmp_path / "dataset.json").write_text(
        json.dumps({"format": "text_pair_multilevel_label_distribution", "level_labels": labels}),
        encoding="utf-8",
    )
    (tmp_path / "train.jsonl").write_text(
        json.dumps(
            {
                "id": "sample-1",
                "text_a": "Premise",
                "text_b": "Hypothesis",
                "human_dists": {
                    "level1": [0.2, 0.8],
                    "level2": [0.1, 0.7, 0.2],
                    "level3": [1.0, 0.0],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    reader = TextPairMultilevelJSONLReader(data_path=str(tmp_path))

    assert reader.level_labels == labels
    assert reader.load_train()[0].hard_labels == {"level1": 1, "level2": 1, "level3": 0}


def test_multilevel_reader_rejects_distribution_with_wrong_manifest_width(tmp_path):
    (tmp_path / "dataset.json").write_text(
        json.dumps(
            {
                "format": "text_pair_multilevel_label_distribution",
                "level_labels": {"level1": ["a", "b"], "level2": ["c", "d"], "level3": ["e", "f"]},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "train.jsonl").write_text(
        json.dumps(
            {"id": "sample-1", "human_dists": {"level1": [1.0], "level2": [1.0, 0.0], "level3": [1.0, 0.0]}}
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="level1 distribution"):
        TextPairMultilevelJSONLReader(data_path=str(tmp_path)).load_train()


def test_single_text_multilevel_reader_uses_text_only_schema(tmp_path):
    labels = {"level1": ["a", "b"], "level2": ["c", "d"], "level3": ["e", "f"]}
    (tmp_path / "dataset.json").write_text(
        json.dumps({"format": "single_text_multilevel_label_distribution", "level_labels": labels}),
        encoding="utf-8",
    )
    (tmp_path / "train.jsonl").write_text(
        json.dumps(
            {
                "id": "sample-1",
                "text": "A single text.",
                "human_dists": {"level1": [1.0, 0.0], "level2": [0.0, 1.0], "level3": [1.0, 0.0]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sample = SingleTextMultilevelJSONLReader(data_path=str(tmp_path)).load_train()[0]

    assert sample.text == "A single text."
    assert not hasattr(sample, "text_a")
    assert not hasattr(sample, "text_b")
    assert sample.hard_labels == {"level1": 0, "level2": 1, "level3": 0}
