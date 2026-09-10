import json

import pytest

from hlv_toolkits.scripts.prepare_multipico_annotation_labels import LABELS, prepare_multipico


def _item(split: str, labels: dict[str, str]) -> dict:
    return {
        "text": {"post": "The original post", "reply": "The reply"},
        "annotations": labels,
        "split": split,
        "lang": "en",
        "other_info": {"source": "reddit", "language_variety": "us"},
    }


def test_prepare_multipico_preserves_official_splits_and_votes(tmp_path):
    splits = {
        "train": {"1": _item("train", {"Ann0": "0", "Ann1": "1", "Ann2": "1"})},
        "dev": {"2": _item("dev", {"Ann0": "0", "Ann1": "1"})},
        "test": {"3": _item("test", {"Ann0": "1"})},
    }
    output = tmp_path / "processed"
    assert prepare_multipico(splits, output) == {"train": 1, "dev": 1, "test": 1}
    manifest = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    train = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))
    assert manifest["labels"] == LABELS
    assert "soft_label_metric_for_best_model" not in manifest
    assert train["id"] == "multipico:train:1"
    assert train["annotation_labels"] == [0, 1, 1]
    assert train["meta"]["source_id"] == "1"


def test_prepare_multipico_rejects_split_or_label_errors(tmp_path):
    splits = {split: {"1": _item(split, {"Ann0": "0"})} for split in ("train", "dev", "test")}
    splits["dev"]["1"]["split"] = "train"
    with pytest.raises(ValueError, match="expected 'dev'"):
        prepare_multipico(splits, tmp_path / "bad-split")
    splits["dev"]["1"]["split"] = "dev"
    splits["test"]["1"]["annotations"] = {"Ann0": "2"}
    with pytest.raises(ValueError, match="invalid label"):
        prepare_multipico(splits, tmp_path / "bad-label")
