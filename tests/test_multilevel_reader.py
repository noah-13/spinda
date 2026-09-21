import json

import pytest

from hlv_toolkits.data.readers.multilevel_reader import TextPairMultilevelJSONReader
from hlv_toolkits.data.readers.single_text_multilevel_reader import SingleTextMultilevelJSONReader


def _write_records(path, records):
    path.write_text(json.dumps(records), encoding="utf-8")


def test_multilevel_reader_uses_manifest_level_labels(tmp_path):
    labels = {"level1": ["a", "b"], "level2": ["c", "d", "e"], "level3": ["f", "g"]}
    (tmp_path / "dataset.json").write_text(
        json.dumps({"format": "text_pair_multidimensional_label_distribution", "level_labels": labels}),
        encoding="utf-8",
    )
    _write_records(tmp_path / "train.json", [{
        "id": "sample-1", "text_a": "Premise", "text_b": "Hypothesis",
        "annotation_labels": {"level1": [0, 1], "level2": [1, 1], "level3": [0, 0]},
    }])

    reader = TextPairMultilevelJSONReader(data_path=str(tmp_path))

    assert reader.level_labels == labels
    assert reader.load_train()[0].hard_labels == {"level1": 1, "level2": 1, "level3": 0}


def test_multilevel_reader_rejects_distribution_with_wrong_manifest_width(tmp_path):
    (tmp_path / "dataset.json").write_text(
        json.dumps({
            "format": "text_pair_multidimensional_label_distribution",
            "level_labels": {"level1": ["a", "b"], "level2": ["c", "d"], "level3": ["e", "f"]},
        }),
        encoding="utf-8",
    )
    _write_records(tmp_path / "train.json", [{
        "id": "sample-1", "annotation_labels": {"level1": [1.0], "level2": [0, 0], "level3": [0, 0]},
    }])

    with pytest.raises(ValueError, match="level1 annotation_labels"):
        TextPairMultilevelJSONReader(data_path=str(tmp_path)).load_train()


def test_single_text_multilevel_reader_uses_text_only_schema(tmp_path):
    labels = {"level1": ["a", "b"], "level2": ["c", "d"], "level3": ["e", "f"]}
    (tmp_path / "dataset.json").write_text(
        json.dumps({"format": "single_text_multidimensional_label_distribution", "level_labels": labels}),
        encoding="utf-8",
    )
    _write_records(tmp_path / "train.json", [{
        "id": "sample-1", "text": "A single text.",
        "annotation_labels": {"level1": [0, 0], "level2": [1, 1], "level3": [0, 0]},
    }])

    sample = SingleTextMultilevelJSONReader(data_path=str(tmp_path)).load_train()[0]

    assert sample.text == "A single text."
    assert not hasattr(sample, "text_a")
    assert not hasattr(sample, "text_b")
    assert sample.hard_labels == {"level1": 0, "level2": 1, "level3": 0}
