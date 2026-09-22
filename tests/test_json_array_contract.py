import json

import pytest
from pathlib import Path
from types import SimpleNamespace

import torch

from hlv_toolkits.data import (
    MultilevelSample,
    SingleTextClassificationJSONReader,
    SingleTextMultilabelJSONReader,
    TextPairClassificationJSONReader,
)
from hlv_toolkits.data.readers.multilevel_reader import TextPairMultilevelJSONReader
from hlv_toolkits.data.readers.single_text_multilevel_reader import SingleTextMultilevelJSONReader
from hlv_toolkits.eval import Evaluator
from hlv_toolkits.scripts.evaluate import _build_multilevel_eval_inputs, load_predictions
from hlv_toolkits.scripts.predict import _build_multilevel_outputs, predict_batch


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_json_arrays_preserve_annotations_for_all_dataset_contracts(tmp_path):
    single = tmp_path / "single"
    single.mkdir()
    _write(single / "dataset.json", {"format": "single_text_label_distribution", "label_mode": "soft", "labels": ["no", "yes"]})
    _write(single / "train.json", [{"id": "s", "text": "text", "annotation_labels": [0, 1, 1]}])
    assert SingleTextClassificationJSONReader(str(single)).load_train()[0].annotation_labels == [0, 1, 1]

    pair = tmp_path / "pair"
    pair.mkdir()
    _write(pair / "dataset.json", {"format": "text_pair_label_distribution", "label_mode": "soft", "labels": ["no", "yes"]})
    _write(pair / "train.json", [{"id": "p", "text_a": "a", "text_b": "b", "annotation_labels": [0, 1, 1]}])
    assert TextPairClassificationJSONReader(str(pair)).load_train()[0].annotation_labels == [0, 1, 1]

    multilabel = tmp_path / "multilabel"
    multilabel.mkdir()
    _write(multilabel / "dataset.json", {"format": "single_text_multilabel_annotation_distribution", "labels": ["a", "b"]})
    _write(multilabel / "train.json", [{"id": "m", "text": "text", "annotation_label_sets": [[0], [0, 1]]}])
    assert SingleTextMultilabelJSONReader(str(multilabel)).load_train()[0].annotation_label_sets == [[0], [0, 1]]

    dimensions = {"sentiment": ["negative", "positive"], "topic": ["a", "b", "c"]}
    pair_dimensions = tmp_path / "pair_dimensions"
    pair_dimensions.mkdir()
    _write(pair_dimensions / "dataset.json", {"format": "text_pair_multidimensional_label_distribution", "level_labels": dimensions})
    _write(pair_dimensions / "train.json", [{"id": "d", "text_a": "a", "text_b": "b", "annotation_labels": {"sentiment": [0, 1], "topic": [2, 2]}}])
    pair_sample = TextPairMultilevelJSONReader(str(pair_dimensions)).load_train()[0]
    assert pair_sample.annotation_labels == {"sentiment": [0, 1], "topic": [2, 2]}

    text_dimensions = tmp_path / "text_dimensions"
    text_dimensions.mkdir()
    _write(text_dimensions / "dataset.json", {"format": "single_text_multidimensional_label_distribution", "level_labels": dimensions})
    _write(text_dimensions / "train.json", [{"id": "sd", "text": "text", "annotation_labels": {"sentiment": [1, 1], "topic": [1, 2]}}])
    assert SingleTextMultilevelJSONReader(str(text_dimensions)).load_train()[0].annotation_labels["topic"] == [1, 2]


class _Tokenizer:
    model_max_length = 8

    def __call__(self, text_a, text_b=None, **kwargs):
        return {"input_ids": torch.ones((len(text_a), 2), dtype=torch.long)}


class _Model(torch.nn.Module):
    def __init__(self, logits, problem_type=None):
        super().__init__()
        self.config = SimpleNamespace(problem_type=problem_type)
        self.logits = torch.tensor(logits, dtype=torch.float)

    def forward(self, **kwargs):
        return SimpleNamespace(logits=self.logits)


def test_prediction_json_schemas_and_dimension_evaluation(tmp_path):
    categorical = predict_batch(_Model([[0.0, 1.0]]), _Tokenizer(), ["x"], None, device="cpu")
    multilabel = predict_batch(_Model([[2.0, -2.0]], "multi_label_classification"), _Tokenizer(), ["x"], None, device="cpu")
    dimensions = _build_multilevel_outputs({"sentiment": torch.tensor([[0.1, 0.9]]), "topic": torch.tensor([[0.2, 0.3, 0.5]])})
    assert set(categorical[0]) == {"probs", "pred"}
    assert multilabel[0]["pred"] == [1, 0]
    assert set(dimensions[0]) == {"dimensions"}



def test_prediction_input_file_infers_text_pair_shape(tmp_path):
    input_path = tmp_path / "external_examples.json"
    _write(
        input_path,
        [{
            "id": "external-1",
            "text_a": "A dog is running.",
            "text_b": "An animal is moving.",
            "annotation_labels": [0, 1],
        }],
    )

    from hlv_toolkits.scripts.predict import _load_inputs

    input_shape, samples = _load_inputs(input_path)

    assert input_shape == "text_pair"
    assert samples == [{
        "id": "external-1",
        "text_a": "A dog is running.",
        "text_b": "An animal is moving.",
    }]


def test_prediction_configs_merge_left_to_right(tmp_path):
    defaults = tmp_path / "defaults.json"
    run_config = tmp_path / "run.json"
    _write(defaults, {"batch_size": 16, "device": "cpu"})
    _write(
        run_config,
        {
            "model_path": "outputs/model/final_model",
            "input_file": "external.json",
            "batch_size": 64,
        },
    )

    from hlv_toolkits.scripts.predict import _load_merged_json_configs

    assert _load_merged_json_configs([str(defaults), str(run_config)]) == {
        "device": "cpu",
        "model_path": "outputs/model/final_model",
        "input_file": "external.json",
        "batch_size": 64,
    }


def test_prediction_config_rejects_training_metadata(tmp_path):
    config_path = tmp_path / "training.json"
    _write(config_path, {"format": "text_pair_label_distribution", "labels": ["no", "yes"]})

    from hlv_toolkits.scripts.predict import _load_json_config

    with pytest.raises(ValueError, match="Unknown prediction config keys.*format.*labels"):
        _load_json_config(str(config_path))
