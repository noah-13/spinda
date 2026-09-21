import torch
from torch import nn
from transformers import TrainingArguments

from hlv_toolkits.models.trainer import SoftLabelTrainer


class _RejectLabelsModel(nn.Module):
    """Minimal classifier that fails if trainer-owned labels reach the model."""

    def forward(self, input_ids, **kwargs):
        assert "labels" not in kwargs
        return type("Output", (), {"logits": torch.zeros((input_ids.shape[0], 3))})()


def test_soft_label_trainer_does_not_forward_distribution_as_model_labels(tmp_path):
    trainer = SoftLabelTrainer(
        model=_RejectLabelsModel(),
        use_soft_labels=True,
        args=TrainingArguments(output_dir=str(tmp_path)),
    )
    inputs = {
        "input_ids": torch.ones((2, 4), dtype=torch.long),
        "labels": torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]),
    }

    loss = trainer.compute_loss(trainer.model, inputs)

    assert torch.isfinite(loss)
