import json

from hlv_toolkits.data.readers.single_text_multilevel_reader import SingleTextMultilevelJSONLReader


def test_single_text_multilevel_reader_uses_text_field_and_manifest(tmp_path):
    labels = {
        "level1": ["a", "b"],
        "level2": ["c", "d", "e"],
        "level3": ["f", "g"],
    }
    (tmp_path / "dataset.json").write_text(
        json.dumps({"format": "single_text_multidimensional_label_distribution", "level_labels": labels}),
        encoding="utf-8",
    )
    (tmp_path / "train.jsonl").write_text(
        json.dumps(
            {
                "id": "sample-1",
                "text": "One input only.",
                "annotation_labels": {
                    "level1": [0, 1],
                    "level2": [1, 1],
                    "level3": [0, 0],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sample = SingleTextMultilevelJSONLReader(data_path=str(tmp_path)).load_train()[0]

    assert sample.text == "One input only."
    assert not hasattr(sample, "text_a")
    assert not hasattr(sample, "text_b")
    assert sample.hard_labels == {"level1": 1, "level2": 1, "level3": 0}


def test_single_text_multilevel_reader_requires_text(tmp_path):
    (tmp_path / "dataset.json").write_text(
        json.dumps({"format": "single_text_multidimensional_label_distribution", "level_labels": {"level1": ["a", "b"], "level2": ["c", "d"], "level3": ["e", "f"]}}),
        encoding="utf-8",
    )
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"id": "sample-1", "annotation_labels": {"level1": [0, 0], "level2": [0, 0], "level3": [0, 0]}}) + "\n",
        encoding="utf-8",
    )

    try:
        SingleTextMultilevelJSONLReader(data_path=str(tmp_path)).load_train()
    except ValueError as exc:
        assert "text" in str(exc)
    else:
        raise AssertionError("Reader should reject a record without text.")
