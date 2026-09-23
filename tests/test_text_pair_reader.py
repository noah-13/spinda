import json
from spinda.models.trainer import TrainingConfig
from pathlib import Path

import pytest

pytestmark = pytest.mark.skip(reason="Superseded by the annotation_labels-only public contract.")
from spinda.data import TextPairClassificationJSONReader
from spinda.data.schemas import TextPairClassificationSample


def write_dataset(path: Path, rows: list[dict]) -> None:
    path.mkdir()
    (path / "dataset.json").write_text(json.dumps({
        "format": "text_pair_label_distribution", "label_mode": "hard", "labels": ["no", "yes"]
    }), encoding="utf-8")
    for split in ("train", "dev"):
        split_rows = rows if split == "train" else rows[:1]
        (path / f"{split}.json").write_text("\n".join(json.dumps(row) for row in split_rows) + "\n", encoding="utf-8")


def test_text_pair_reader_loads_fixed_contract(tmp_path: Path):
    data_dir = tmp_path / "pairs"
    write_dataset(data_dir, [{"id": "a", "text_a": "left", "text_b": "right", "label": 1, "meta": {"source_id": 3}}])
    reader = TextPairClassificationJSONReader(str(data_dir))
    samples = reader.load_train()
    assert reader.labels == ["no", "yes"]
    assert isinstance(samples[0], TextPairClassificationSample)
    assert samples[0].text_a == "left"
    assert samples[0].label == 1


def test_text_pair_reader_rejects_out_of_range_label(tmp_path: Path):
    data_dir = tmp_path / "pairs"
    write_dataset(data_dir, [{"id": "a", "text_a": "left", "text_b": "right", "label": 2}])
    with pytest.raises(ValueError, match="label outside"):
        TextPairClassificationJSONReader(str(data_dir)).load_train()


def test_text_pair_soft_reader_loads_distribution(tmp_path: Path):
    data_dir = tmp_path / "soft_pairs"
    data_dir.mkdir()
    (data_dir / "dataset.json").write_text(json.dumps({
        "format": "text_pair_label_distribution", "label_mode": "soft", "labels": ["no", "yes"]
    }), encoding="utf-8")
    row = {"id": "a", "text_a": "left", "text_b": "right", "label_distribution": [0.25, 0.75]}
    for split in ("train", "dev"):
        (data_dir / f"{split}.json").write_text(json.dumps(row) + "\n", encoding="utf-8")
    reader = TextPairClassificationJSONReader(str(data_dir))
    sample = reader.load_train()[0]
    assert reader.use_soft_labels is True
    assert sample.label == 1
    assert sample.human_dist == [0.25, 0.75]


def test_text_pair_soft_reader_rejects_non_normalized_distribution(tmp_path: Path):
    data_dir = tmp_path / "bad_soft_pairs"
    data_dir.mkdir()
    (data_dir / "dataset.json").write_text(json.dumps({
        "format": "text_pair_label_distribution", "label_mode": "soft", "labels": ["no", "yes"]
    }), encoding="utf-8")
    row = {"id": "a", "text_a": "left", "text_b": "right", "label_distribution": [0.2, 0.7]}
    for split in ("train", "dev"):
        (data_dir / f"{split}.json").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sum to 1"):
        TextPairClassificationJSONReader(str(data_dir)).load_train()


def test_text_pair_soft_to_hard_uses_argmax(tmp_path: Path):
    data_dir = tmp_path / "soft_to_hard_pairs"
    data_dir.mkdir()
    (data_dir / "dataset.json").write_text(json.dumps({
        "format": "text_pair_label_distribution", "label_mode": "soft_to_hard", "labels": ["no", "yes"]
    }), encoding="utf-8")
    row = {"id": "a", "text_a": "left", "text_b": "right", "label_distribution": [0.25, 0.75]}
    for split in ("train", "dev"):
        (data_dir / f"{split}.json").write_text(json.dumps(row) + "\n", encoding="utf-8")

    reader = TextPairClassificationJSONReader(str(data_dir))
    sample = reader.load_train()[0]
    assert reader.use_soft_labels is False
    assert sample.label == 1
    assert not hasattr(sample, "human_dist")


def test_text_pair_soft_to_hard_rejects_hard_source(tmp_path: Path):
    data_dir = tmp_path / "bad_soft_to_hard_pairs"
    write_dataset(data_dir, [{"id": "a", "text_a": "left", "text_b": "right", "label": 1}])
    manifest_path = data_dir / "dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["label_mode"] = "soft_to_hard"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="requires source rows with label_distribution"):
        TextPairClassificationJSONReader(str(data_dir))


def test_text_pair_explicit_mode_rejects_data_type_mismatch(tmp_path: Path):
    data_dir = tmp_path / "mode_mismatch_pairs"
    data_dir.mkdir()
    (data_dir / "dataset.json").write_text(json.dumps({
        "format": "text_pair_label_distribution", "label_mode": "hard", "labels": ["no", "yes"]
    }), encoding="utf-8")
    row = {"id": "a", "text_a": "left", "text_b": "right", "label_distribution": [0.25, 0.75]}
    for split in ("train", "dev"):
        (data_dir / f"{split}.json").write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        TextPairClassificationJSONReader(str(data_dir))


def test_text_pair_soft_reader_rejects_distribution_length_not_matching_labels(tmp_path: Path):
    data_dir = tmp_path / "wrong_distribution_size"
    data_dir.mkdir()
    (data_dir / "dataset.json").write_text(json.dumps({
        "format": "text_pair_label_distribution", "label_mode": "soft", "labels": ["a", "b", "c"]
    }), encoding="utf-8")
    row = {"id": "a", "text_a": "left", "text_b": "right", "label_distribution": [0.25, 0.75]}
    for split in ("train", "dev"):
        (data_dir / f"{split}.json").write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="with 3 values"):
        TextPairClassificationJSONReader(str(data_dir)).load_train()


def test_text_pair_missing_mode_and_labels_are_inferred_with_warnings(tmp_path: Path):
    data_dir = tmp_path / "inferred_pairs"
    data_dir.mkdir()
    (data_dir / "dataset.json").write_text(json.dumps({"format": "text_pair_label_distribution"}), encoding="utf-8")
    row = {"id": "a", "text_a": "left", "text_b": "right", "label_distribution": [0.25, 0.75]}
    for split in ("train", "dev"):
        (data_dir / f"{split}.json").write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.warns(UserWarning):
        reader = TextPairClassificationJSONReader(str(data_dir))
    assert reader.label_mode == "soft"
    assert reader.labels == ["label0", "label1"]


def test_text_pair_reader_supports_direct_train_path_without_dev(tmp_path: Path):
    train_path = tmp_path / "train.json"
    row = {"id": "a", "text_a": "left", "text_b": "right", "label_distribution": [0.2, 0.8]}
    train_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    reader = TextPairClassificationJSONReader(
        data_format="text_pair_label_distribution",
        train_path=str(train_path),
        labels=["no", "yes"],
        label_mode="soft",
    )
    assert reader.has_dev is False
    assert reader.load_train()[0].human_dist == [0.2, 0.8]
    with pytest.raises(FileNotFoundError, match="No dev_path"):
        reader.load_dev()


def test_text_pair_direct_training_requires_format(tmp_path: Path):
    train_path = tmp_path / "train.json"
    train_path.write_text('{"id":"a","text_a":"left","text_b":"right","label":0}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="format is required"):
        TextPairClassificationJSONReader(train_path=str(train_path))


def test_training_config_disables_evaluation_and_best_model_without_dev(tmp_path: Path):
    training_args = TrainingConfig(output_dir=str(tmp_path / "outputs"), has_eval=False).to_training_args()
    assert training_args.eval_strategy == "no"
    assert training_args.load_best_model_at_end is False
