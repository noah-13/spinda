import json

from hlv_toolkits.scripts.train import DATA_FORMAT_SPECS, _load_json_config


def test_format_determines_model_head_type():
    assert DATA_FORMAT_SPECS == {
        "text_pair_label_distribution": ("text_pair", "classification"),
        "single_text_label_distribution": ("single_text", "classification"),
        "single_text_multilabel_annotation_distribution": ("single_text_multilabel", "multilabel_classification"),
        "text_pair_multilevel_label_distribution": ("multilevel", "multilevel_classification"),
        "single_text_multilevel_label_distribution": ("multilevel", "multilevel_classification"),
    }


def test_training_config_preserves_multilevel_manifest_labels(tmp_path):
    manifest_path = tmp_path / "dataset.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "single_text_multilevel_label_distribution",
                "label_mode": "soft",
                "level_labels": {"level1": ["a", "b"], "level2": ["c", "d"], "level3": ["e", "f"]},
                "train_path": "train.jsonl",
                "dev_path": "dev.jsonl",
            }
        ),
        encoding="utf-8",
    )

    config = _load_json_config(
        str(manifest_path),
        {"data_format", "label_mode", "level_labels", "train_path", "dev_path"},
    )

    assert config == {
        "data_format": "single_text_multilevel_label_distribution",
        "label_mode": "soft",
        "level_labels": {"level1": ["a", "b"], "level2": ["c", "d"], "level3": ["e", "f"]},
        "train_path": "train.jsonl",
        "dev_path": "dev.jsonl",
    }
