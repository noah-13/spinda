import json
import os

from spinda.models import trainer as trainer_module
from spinda.models.trainer import TrainingConfig
from spinda.scripts.train import DATA_FORMAT_SPECS, _load_json_config


def test_format_determines_model_head_type():
    assert DATA_FORMAT_SPECS == {
        "text_pair_label_distribution": ("text_pair", "classification"),
        "single_text_label_distribution": ("single_text", "classification"),
        "single_text_multilabel_annotation_distribution": ("single_text_multilabel", "multilabel_classification"),
        "text_pair_multidimensional_label_distribution": ("multilevel", "multilevel_classification"),
        "single_text_multidimensional_label_distribution": ("multilevel", "multilevel_classification"),
    }


def test_training_config_preserves_multilevel_manifest_labels(tmp_path):
    manifest_path = tmp_path / "dataset.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "single_text_multidimensional_label_distribution",
                "label_mode": "soft",
                "level_labels": {"level1": ["a", "b"], "level2": ["c", "d"], "level3": ["e", "f"]},
                "train_path": "train.json",
                "dev_path": "dev.json",
            }
        ),
        encoding="utf-8",
    )

    config = _load_json_config(
        str(manifest_path),
        {"data_format", "label_mode", "level_labels", "train_path", "dev_path"},
    )

    assert config == {
        "data_format": "single_text_multidimensional_label_distribution",
        "label_mode": "soft",
        "level_labels": {"level1": ["a", "b"], "level2": ["c", "d"], "level3": ["e", "f"]},
        "train_path": "train.json",
        "dev_path": "dev.json",
    }


def test_multilabel_metric_for_best_model_checkpoint_selection_direction(tmp_path):
    output_dir = tmp_path / "run"
    for metric_name in ("soft_micro_f1", "soft_macro_f1", "multilabel_pojsd", "multilabel_entropy_correlation"):
        args = TrainingConfig(
            output_dir=str(output_dir),
            head_type="multilabel_classification",
            use_soft_labels=True,
            multilabel_metric_for_best_model=metric_name,
        ).to_training_args()
        assert args.metric_for_best_model == metric_name
        assert args.greater_is_better is True


def test_explicit_cuda_device_uses_one_gpu_without_changing_process_environment(monkeypatch, tmp_path):
    observed = {}

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            observed["accelerate_device"] = os.environ.get("ACCELERATE_TORCH_DEVICE")

    monkeypatch.setattr(trainer_module, "TrainingArguments", FakeTrainingArguments)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("ACCELERATE_TORCH_DEVICE", raising=False)

    args = TrainingConfig(output_dir=str(tmp_path / "run"), device="cuda:1").to_training_args()

    assert observed["accelerate_device"] == "cuda:1"
    assert args._n_gpu == 1
    assert "ACCELERATE_TORCH_DEVICE" not in os.environ
