from types import SimpleNamespace

import torch
import torch.nn.functional as F

from hlv_toolkits.models.trainer import SoftLabelTrainer


class _Model:
    def __call__(self, **kwargs):
        return SimpleNamespace(logits=torch.tensor([[0.4, -0.7]], requires_grad=True))


def _loss(strategy, labels):
    trainer = SimpleNamespace(head_type="multilabel_classification", soft_label_loss=strategy)
    return SoftLabelTrainer.compute_loss(trainer, _Model(), {"labels": torch.tensor([labels])})


def test_multilabel_ce_and_rel_use_bce():
    expected = F.binary_cross_entropy_with_logits(torch.tensor([[0.4, -0.7]]), torch.tensor([[0.75, 0.25]]))
    assert torch.allclose(_loss("ce", [0.75, 0.25]), expected)
    assert torch.allclose(_loss("rel", [0.75, 0.25]), expected)


def test_multilabel_mse_uses_sigmoid_probabilities():
    expected = F.mse_loss(torch.sigmoid(torch.tensor([[0.4, -0.7]])), torch.tensor([[0.75, 0.25]]))
    assert torch.allclose(_loss("mse", [0.75, 0.25]), expected)


def test_multilabel_jsd_is_finite_for_probability_targets():
    loss = _loss("jsd", [0.75, 0.25])
    assert torch.isfinite(loss)
    assert loss.item() >= 0
