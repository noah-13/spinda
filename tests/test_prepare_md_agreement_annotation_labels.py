import json

import pytest

from hlv_toolkits.scripts.prepare_md_agreement_annotation_labels import prepare_md_agreement


def _item(text: str, split: str, annotations: str = "0,0,0,1,1") -> dict:
    return {
        "text": text,
        "annotations": annotations,
        "split": split,
        "other_info": {"domain": "BLM"},
    }


def test_prepare_md_agreement_preserves_votes_and_splits(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for split in ("train", "dev", "test"):
        (raw / f"MD-Agreement_{split}.json").write_text(
            json.dumps({"1": _item(f"{split} tweet", split)}), encoding="utf-8"
        )

    output = tmp_path / "processed"
    assert prepare_md_agreement(raw, output) == {"train": 1, "dev": 1, "test": 1}
    manifest = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    row = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))

    assert manifest["format"] == "single_text_label_distribution"
    assert manifest["label_mode"] == "soft"
    assert manifest["labels"] == ["not_offensive", "offensive"]
    assert manifest["train_path"] == str(output / "train.jsonl")
    assert manifest["dev_path"] == str(output / "dev.jsonl")
    assert set(manifest) == {"format", "label_mode", "labels", "train_path", "dev_path"}
    assert row == {
        "id": "md_agreement:train:1",
        "text": "train tweet",
        "annotation_labels": [0, 0, 0, 1, 1],
        "meta": {"domain": "BLM", "source_id": "1"},
    }


def test_prepare_md_agreement_rejects_non_binary_or_wrong_vote_count(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for split in ("train", "dev", "test"):
        annotations = "0,0,2" if split == "train" else "0,0,0,1,1"
        (raw / f"MD-Agreement_{split}.json").write_text(
            json.dumps({"1": _item("tweet", split, annotations)}), encoding="utf-8"
        )

    with pytest.raises(ValueError, match="exactly five binary"):
        prepare_md_agreement(raw, tmp_path / "processed")


def test_md_agreement_output_loads_as_single_text_soft_dataset(tmp_path):
    from hlv_toolkits.data import SingleTextClassificationJSONLReader

    raw = tmp_path / "raw"
    raw.mkdir()
    for split in ("train", "dev", "test"):
        (raw / f"MD-Agreement_{split}.json").write_text(
            json.dumps({"1": _item("tweet", split)}), encoding="utf-8"
        )
    output = tmp_path / "processed"
    prepare_md_agreement(raw, output)

    sample = SingleTextClassificationJSONLReader(str(output)).load_train()[0]
    assert sample.text == "tweet"
    assert sample.label == 0
    assert sample.human_dist == [0.6, 0.4]
    assert sample.annotation_labels == [0, 0, 0, 1, 1]


def test_md_agreement_soft_to_hard_retains_distribution_for_dev_metrics(tmp_path):
    from hlv_toolkits.data import SingleTextClassificationJSONLReader

    raw = tmp_path / "raw"
    raw.mkdir()
    for split in ("train", "dev", "test"):
        (raw / f"MD-Agreement_{split}.json").write_text(
            json.dumps({"1": _item("tweet", split)}), encoding="utf-8"
        )
    output = tmp_path / "processed"
    prepare_md_agreement(raw, output)

    sample = SingleTextClassificationJSONLReader(
        data_format="single_text_label_distribution",
        train_path=str(output / "train.jsonl"),
        dev_path=str(output / "dev.jsonl"),
        labels=["not_offensive", "offensive"],
        label_mode="soft_to_hard",
    ).load_train()[0]
    assert sample.label == 0
    assert sample.human_dist == [0.6, 0.4]
