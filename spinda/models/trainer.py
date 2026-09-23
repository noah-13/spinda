from __future__ import annotations

import inspect
import json
import os
import time
from importlib.util import find_spec
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset  # type: ignore
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer_utils import EvalPrediction
from transformers.utils import ModelOutput

from spinda.eval.metrics import (
    compute_multilabel_entropy_correlation,
    compute_multilabel_pojsd,
    compute_soft_macro_f1,
    compute_soft_micro_f1,
)
from spinda.data.schemas import (
    MultilevelSample,
    TextPairClassificationSample,
    TextPairDistributionSample,
    SingleTextClassificationSample,
    SingleTextDistributionSample,
    SingleTextMultilabelDistributionSample,
    SingleTextMultilevelSample,
)


def _build_training_arguments(device: str, **kwargs) -> TrainingArguments:
    """Create TrainingArguments while honoring an explicit single CUDA device."""
    is_single_process_cuda = (
        device.startswith("cuda")
        and os.environ.get("LOCAL_RANK") in {None, "", "-1"}
    )
    if not is_single_process_cuda:
        return TrainingArguments(**kwargs)

    # Transformers otherwise selects cuda:0 and treats every visible GPU as
    # available for DataParallel. Accelerate reads this during construction.
    previous_accelerate_device = os.environ.get("ACCELERATE_TORCH_DEVICE")
    os.environ["ACCELERATE_TORCH_DEVICE"] = device
    try:
        training_args = TrainingArguments(**kwargs)
    finally:
        if previous_accelerate_device is None:
            os.environ.pop("ACCELERATE_TORCH_DEVICE", None)
        else:
            os.environ["ACCELERATE_TORCH_DEVICE"] = previous_accelerate_device

    training_args._n_gpu = 1
    return training_args


def load_tokenizer_with_fallback(model_name_or_path: str):
    model_ref = str(model_name_or_path).lower()
    if "deberta" in model_ref:
        if find_spec("sentencepiece") is None:
            raise ImportError(
                "Loading DeBERTa tokenizers requires the sentencepiece package. "
                "Install it with `uv add sentencepiece` or `uv pip install sentencepiece`, "
                "then rerun training."
            )

        from transformers import DebertaV2Tokenizer

        print("Loading DeBERTa tokenizer with the slow SentencePiece tokenizer.")
        return DebertaV2Tokenizer.from_pretrained(model_name_or_path)

    try:
        return AutoTokenizer.from_pretrained(model_name_or_path)
    except AttributeError as exc:
        # Some checkpoints hit a transformers fast-tokenizer conversion bug.
        if "NoneType" not in str(exc) or "endswith" not in str(exc):
            raise
        print(
            "Fast tokenizer loading failed with a known transformers conversion error; "
            "retrying with use_fast=False."
        )
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)


def resolve_max_length(tokenizer, config, requested_max_length: int) -> int:
    """Resolve 0/negative max_length to the model's usable sequence limit."""
    if requested_max_length > 0:
        return requested_max_length

    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and tokenizer_limit < 10**9:
        return tokenizer_limit

    config_limit = getattr(config, "max_position_embeddings", None)
    if isinstance(config_limit, int) and config_limit > 0:
        # RoBERTa-style configs include two reserved positions.
        return config_limit - 2 if config_limit > 512 else config_limit

    return 512


MULTILEVEL_LEVEL_ORDER: tuple[str, ...] = ()
MultilevelInputSample = Union[MultilevelSample, SingleTextMultilevelSample]


LABEL_INPUT_PREFIXES = ("labels", "hard_labels", "soft_labels")


def _print_dev_metric_plot(trainer: Trainer, metric_name: str) -> None:
    """Print a small dependency-free terminal plot of the dev metric."""
    metric_key = f"eval_{metric_name}"
    points = [
        (float(row["epoch"]), float(row[metric_key]))
        for row in trainer.state.log_history
        if row.get("epoch") is not None and row.get(metric_key) is not None
    ]
    if not points:
        print(f"No dev values found for {metric_key}; skipping plot.")
        return

    width, height = 60, 18
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    y_min, y_max = min(ys), max(ys)
    if y_max == y_min:
        y_min -= 0.5
        y_max += 0.5
    else:
        margin = (y_max - y_min) * 0.08
        y_min -= margin
        y_max += margin

    def project(epoch: float, value: float) -> tuple[int, int]:
        x = 0 if xs[-1] == xs[0] else round((epoch - xs[0]) / (xs[-1] - xs[0]) * (width - 1))
        y = round((y_max - value) / (y_max - y_min) * (height - 1))
        return max(0, min(width - 1, x)), max(0, min(height - 1, y))

    canvas = [[" " for _ in range(width)] for _ in range(height)]
    previous = None
    for epoch, value in points:
        x, y = project(epoch, value)
        if previous is not None:
            px, py = previous
            step = 1 if x >= px else -1
            for ix in range(px, x + step, step):
                iy = round(py + (y - py) * ((ix - px) / (x - px))) if x != px else y
                canvas[max(0, min(height - 1, iy))][ix] = "▌"
        canvas[y][x] = "█"
        previous = (x, y)

    print(f"\nDev scores ({metric_name}) over epochs (x)")
    print("┌" + "─" * width + "┐")
    for row_idx, row in enumerate(canvas):
        label = f" {y_max:.3g}" if row_idx == 0 else (f" {y_min:.3g}" if row_idx == height - 1 else "")
        print("│" + "".join(row) + "│" + label)
    print("└" + "─" * width + "┘")
    print(f"epoch {xs[0]:g}" + " " * max(1, width - 16) + f"{xs[-1]:g}")


def _strip_label_inputs(kwargs):
    return {
        key: value
        for key, value in kwargs.items()
        if key not in {"labels", "hard_labels", "soft_labels"}
        and not key.startswith("hard_labels_")
        and not key.startswith("soft_labels_")
    }


@dataclass
class TrainingConfig:
    """Configuration for HLV model training."""

    # Model configuration
    model_name_or_path: str = "roberta-base"  # or "bert-base-uncased"
    num_labels: int = 0
    label_names: Optional[List[str]] = None

    # Training hyperparameters
    learning_rate: float = 2e-5
    train_batch_size: int = 32
    eval_batch_size: int = 64
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1

    # Head and objective settings
    head_type: str = "classification"  # choices: classification, multilevel_classification, multilabel_classification

    # Soft-label training settings
    use_soft_labels: bool = False
    # ``soft_to_hard`` trains on argmax labels, but retains human distributions
    # on evaluation so model selection can still use distributional metrics.
    use_soft_eval_metrics: bool = False
    soft_label_loss: str = "ce"  # choices: ce, mse, jsd, rel
    soft_label_metric_for_best_model: str = "tvd"  # choices: accuracy, tvd, kl_divergence, soft_{micro,macro}_f1, multilabel_{pojsd,entropy_correlation}
    multilabel_metric_for_best_model: str = "soft_micro_f1"  # checkpoint metric for every multilabel_classification task
    multilevel_label_sizes: Optional[Dict[str, int]] = None

    # Data paths
    output_dir: str = "./outputs"
    resume_from_checkpoint: Optional[str] = None
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    has_eval: bool = True
    save_total_limit: int = 1

    # Tokenization settings
    max_length: int = 0  # 0 means auto-resolve from tokenizer/model config.

    # Other settings
    seed: int = 42
    fp16: bool = False
    device: str = "auto"
    dataloader_num_workers: int = 2

    # Wandb settings
    use_wandb: bool = False
    wandb_project: str = "hlv"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_job_type: Optional[str] = None
    wandb_run_name: Optional[str] = None

    def to_training_args(self) -> TrainingArguments:
        valid_head_types = {"classification", "multilevel_classification", "multilabel_classification"}
        if self.head_type not in valid_head_types:
            raise ValueError(
                "head_type must be one of: classification, multilevel_classification, multilabel_classification"
            )
        if self.soft_label_loss not in {"ce", "mse", "jsd", "rel"}:
            raise ValueError("label_training_strategy must be one of: ce, mse, jsd, rel")
        multilabel_only_metrics = {"multilabel_pojsd", "multilabel_entropy_correlation"}
        if self.multilabel_metric_for_best_model not in {"accuracy", "soft_micro_f1", "soft_macro_f1", *multilabel_only_metrics}:
            raise ValueError("Unsupported multilabel_metric_for_best_model")
        if self.soft_label_metric_for_best_model in multilabel_only_metrics and self.head_type != "multilabel_classification":
            raise ValueError("Multilabel checkpoint metrics require head_type=multilabel_classification")
        if self.label_names is not None and len(self.label_names) != self.num_labels:
            raise ValueError("label_names must have exactly num_labels entries")
        report_to = []
        if self.use_wandb:
            try:
                import wandb  # noqa: F401
            except ImportError:
                raise ImportError(
                    "wandb is required when --use_wandb is enabled. "
                    "Install it with: uv pip install wandb "
                    "or: uv sync --extra dev"
                )
            report_to.append("wandb")
            if self.wandb_run_name:
                os.environ["WANDB_RUN_NAME"] = self.wandb_run_name
                os.environ["WANDB_NAME"] = self.wandb_run_name
            if self.wandb_entity:
                os.environ["WANDB_ENTITY"] = self.wandb_entity
            if self.wandb_group:
                os.environ["WANDB_RUN_GROUP"] = self.wandb_group
            if self.wandb_job_type:
                os.environ["WANDB_JOB_TYPE"] = self.wandb_job_type
            os.environ["WANDB_PROJECT"] = self.wandb_project

        metric_for_best_model = "accuracy"
        greater_is_better = True
        if self.use_soft_labels or self.use_soft_eval_metrics:
            metric_for_best_model = (
                self.multilabel_metric_for_best_model
                if self.head_type == "multilabel_classification"
                else self.soft_label_metric_for_best_model
            )
            greater_is_better = metric_for_best_model in {
                "accuracy", "soft_micro_f1", "soft_macro_f1",
                "multilabel_pojsd", "multilabel_entropy_correlation",
            }

        label_names = ["soft_labels"] if self.use_soft_eval_metrics else ["labels"]
        if self.head_type == "multilevel_classification":
            rel_training = self.use_soft_labels and self.soft_label_loss == "rel"
            # ReL trains on individual hard annotation votes, but validation
            # must expose aggregate soft distributions to compute TVD/KL.
            # ``label_names`` controls labels collected for evaluation; hard
            # columns remain in the batch for loss calculation.
            prefix = "soft_labels" if rel_training or self.use_soft_eval_metrics else (
                "hard_labels" if not self.use_soft_labels else "soft_labels"
            )
            label_names = [f"{prefix}_{level}" for level in MULTILEVEL_LEVEL_ORDER]

        return _build_training_arguments(
            self.device,
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            eval_strategy=self.eval_strategy if self.has_eval else "no",
            save_strategy=self.save_strategy,
            save_total_limit=self.save_total_limit,
            seed=self.seed,
            fp16=self.fp16,
            use_cpu=self.device == "cpu",
            dataloader_num_workers=self.dataloader_num_workers,
            dataloader_pin_memory=self.device != "cpu",
            load_best_model_at_end=self.has_eval,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            logging_steps=100,
            save_steps=500,
            report_to=report_to,
            label_names=label_names,
        )


@dataclass
class MultiLevelSequenceClassifierOutput(ModelOutput):
    logits: Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class MultiLevelClassificationModel(nn.Module):
    """Backbone + one softmax head per dataset-defined label level."""

    META_FILENAME = "multilevel_classification_meta.json"

    def __init__(self, backbone: nn.Module, hidden_size: int, level_num_labels: dict[str, int]) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.1)
        if not level_num_labels or any(size < 2 for size in level_num_labels.values()):
            raise ValueError("level_num_labels must define at least two labels for every dimension.")
        self.level_num_labels = dict(level_num_labels)
        self.classifiers = nn.ModuleDict(
            {
                level: nn.Linear(hidden_size, self.level_num_labels[level])
                for level in self.level_num_labels
            }
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        level_num_labels: Optional[dict[str, int]] = None,
    ) -> "MultiLevelClassificationModel":
        config = AutoConfig.from_pretrained(model_name_or_path)
        backbone = AutoModel.from_pretrained(model_name_or_path, config=config)
        hidden_size = int(config.hidden_size)
        meta_path = Path(model_name_or_path) / cls.META_FILENAME
        state_path = Path(model_name_or_path) / "multilevel_classification_heads.pt"
        if level_num_labels is None:
            if not meta_path.is_file():
                raise ValueError("level_num_labels is required unless loading a saved multilevel model.")
            with open(meta_path, encoding="utf-8") as f:
                level_num_labels = json.load(f).get("level_num_labels")
        model = cls(backbone=backbone, hidden_size=hidden_size, level_num_labels=level_num_labels)
        if meta_path.exists() and state_path.exists():
            model.load_state_dict(torch.load(state_path, map_location="cpu"))
        return model

    def save_pretrained(self, save_dir: Union[str, Path]) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(save_path)
        torch.save(self.state_dict(), save_path / "multilevel_classification_heads.pt")
        with open(save_path / self.META_FILENAME, "w", encoding="utf-8") as f:
            json.dump({"head_type": "multilevel_classification", "level_num_labels": self.level_num_labels}, f)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        backbone_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **_strip_label_inputs(kwargs),
        }
        if token_type_ids is not None and "token_type_ids" in inspect.signature(self.backbone.forward).parameters:
            backbone_kwargs["token_type_ids"] = token_type_ids
        outputs = self.backbone(**backbone_kwargs)
        pooled = outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None else outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits = tuple(self.classifiers[level](pooled) for level in self.level_num_labels)
        return MultiLevelSequenceClassifierOutput(
            logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )


class ThroughputLoggingCallback(TrainerCallback):
    def __init__(self, train_batch_size: int, gradient_accumulation_steps: int) -> None:
        self.examples_per_step = max(1, train_batch_size) * max(1, gradient_accumulation_steps)
        self.last_epoch_time: Optional[float] = None
        self.last_epoch_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_epoch_time = time.time()
        self.last_epoch_step = int(state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.last_epoch_time is None:
            self.last_epoch_time = time.time()
            self.last_epoch_step = int(state.global_step)
            return

        current_step = int(state.global_step)
        step_delta = current_step - self.last_epoch_step
        time_delta = time.time() - self.last_epoch_time
        if step_delta <= 0 or time_delta <= 0:
            return

        world_size = max(1, getattr(args, 'world_size', 1) or 1)
        examples_per_second = (step_delta * self.examples_per_step * world_size) / time_delta
        epoch_label = state.epoch
        if epoch_label is None:
            epoch_text = 'unknown'
        else:
            epoch_text = f'{epoch_label:.2f}'
        print(f"Epoch {epoch_text} throughput: {examples_per_second:.2f} examples/s")

        self.last_epoch_time = time.time()
        self.last_epoch_step = current_step


class SoftLabelTrainer(Trainer):
    """HuggingFace Trainer for classification and multilevel classification."""

    def __init__(
        self,
        *args,
        use_soft_labels: bool = False,
        soft_label_loss: str = "ce",
        head_type: str = "classification",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_soft_labels = use_soft_labels
        self.soft_label_loss = soft_label_loss
        self.head_type = head_type

    def _set_signature_columns_if_needed(self) -> None:
        super()._set_signature_columns_if_needed()
        # Preserve soft reference labels used exclusively by evaluation in
        # soft-to-hard runs; multilevel loss also needs hard label columns.
        label_columns = ["soft_labels"]
        if self.head_type == "multilevel_classification":
            label_columns.extend(
                f"{prefix}_{level}"
                for prefix in ("hard_labels", "soft_labels")
                for level in MULTILEVEL_LEVEL_ORDER
            )
        self._signature_columns = list(set(self._signature_columns + label_columns))

    def _compute_soft_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(logits.dtype)
        labels = labels / labels.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        if self.soft_label_loss == "mse":
            predictions = F.softmax(logits, dim=-1)
            return F.mse_loss(predictions, labels)

        log_probs = F.log_softmax(logits, dim=-1)
        if self.soft_label_loss == "jsd":
            predictions = log_probs.exp()
            mixture = 0.5 * (predictions + labels)
            log_mixture = mixture.clamp(min=1e-8).log()
            kl_labels = (labels * (labels.clamp(min=1e-8).log() - log_mixture)).sum(dim=-1)
            kl_predictions = (predictions * (log_probs - log_mixture)).sum(dim=-1)
            return 0.5 * (kl_labels + kl_predictions).mean()
        return -(labels * log_probs).sum(dim=-1).mean()

    def _labels_for_level(self, inputs, level: str, prefix: str) -> torch.Tensor:
        labels = inputs[f"{prefix}_{level}"]
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        return labels

    def _level_logits_from_output(self, logits):
        if isinstance(logits, tuple):
            return {level: logits[idx] for idx, level in enumerate(MULTILEVEL_LEVEL_ORDER)}
        if isinstance(logits, list):
            return {level: logits[idx] for idx, level in enumerate(MULTILEVEL_LEVEL_ORDER)}
        raise TypeError(f"Unsupported multilevel logits type: {type(logits)!r}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Hugging Face sequence-classification models interpret ``labels`` as
        # hard class indices. Distributional soft labels have shape
        # ``[batch_size, num_labels]``; passing them through would make the
        # base model flatten them while computing its own loss (e.g. 17 -> 51
        # for three NLI labels). This trainer owns every loss calculation, so
        # never forward any of its label columns to the model.
        outputs = model(**_strip_label_inputs(inputs))
        logits = outputs.logits

        if self.head_type == "multilabel_classification":
            targets = inputs["labels"].to(logits.dtype)
            if self.soft_label_loss == "mse":
                loss = F.mse_loss(torch.sigmoid(logits), targets)
            elif self.soft_label_loss == "jsd":
                probs = torch.sigmoid(logits)
                target_dist = torch.stack((targets, 1 - targets), dim=-1)
                pred_dist = torch.stack((probs, 1 - probs), dim=-1)
                mixture = 0.5 * (target_dist + pred_dist)
                loss = 0.5 * ((target_dist * (target_dist.clamp_min(1e-8).log() - mixture.clamp_min(1e-8).log())).sum(dim=-1) + (pred_dist * (pred_dist.clamp_min(1e-8).log() - mixture.clamp_min(1e-8).log())).sum(dim=-1)).mean()
            else:
                loss = F.binary_cross_entropy_with_logits(logits, targets)
        elif self.head_type == "multilevel_classification":
            rel_training = self.use_soft_labels and self.soft_label_loss == "rel"
            prefix = "hard_labels" if rel_training or not self.use_soft_labels else "soft_labels"
            level_logits = self._level_logits_from_output(logits)
            losses = []
            for level in MULTILEVEL_LEVEL_ORDER:
                level_logits_t = level_logits[level]
                level_labels = self._labels_for_level(inputs, level, prefix)
                if self.use_soft_labels and not rel_training:
                    losses.append(self._compute_soft_loss(level_logits_t, level_labels))
                else:
                    losses.append(F.cross_entropy(level_logits_t, level_labels.long()))
            loss = sum(losses)
        else:
            labels = inputs["labels"]
            if self.use_soft_labels:
                loss = F.cross_entropy(logits, labels.long()) if self.soft_label_loss == "rel" and labels.ndim == 1 else self._compute_soft_loss(logits, labels)
            else:
                loss = F.cross_entropy(logits, labels.long())

        return (loss, outputs) if return_outputs else loss


class HLVTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.tokenizer = None
        self.model = None

    def initialize_model(self) -> None:
        model_name = self.config.model_name_or_path
        print(f"Initializing model: {model_name}", flush=True)
        print("Loading tokenizer...", flush=True)
        self.tokenizer = load_tokenizer_with_fallback(model_name)
        print("Loading pretrained model weights...", flush=True)

        if self.config.head_type == "multilevel_classification":
            level_sizes = self.config.multilevel_label_sizes
            if level_sizes is None:
                raise ValueError("multilevel_label_sizes must come from the dataset manifest.")
            global MULTILEVEL_LEVEL_ORDER
            MULTILEVEL_LEVEL_ORDER = tuple(level_sizes)
            self.model = MultiLevelClassificationModel.from_pretrained(
                model_name,
                level_num_labels=level_sizes,
            )
        else:
            if self.config.num_labels < 2:
                raise ValueError("num_labels must come from the dataset manifest and be at least 2.")
            model_kwargs = {"num_labels": self.config.num_labels}
            if self.config.head_type == "multilabel_classification":
                model_kwargs["problem_type"] = "multi_label_classification"
            if self.config.label_names is not None:
                model_kwargs["id2label"] = {i: label for i, label in enumerate(self.config.label_names)}
                model_kwargs["label2id"] = {label: i for i, label in enumerate(self.config.label_names)}
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
        print("Model initialization complete.", flush=True)

    def prepare_dataset(
        self,
        train_samples: List[Union[TextPairClassificationSample, TextPairDistributionSample, SingleTextClassificationSample, SingleTextDistributionSample, SingleTextMultilabelDistributionSample, MultilevelInputSample]],
        eval_samples: Optional[List[Union[TextPairClassificationSample, TextPairDistributionSample, SingleTextClassificationSample, SingleTextDistributionSample, SingleTextMultilabelDistributionSample, MultilevelInputSample]]] = None,
    ) -> Tuple[Dataset, Optional[Dataset]]:
        if self.tokenizer is None:
            raise ValueError("Must call initialize_model() first")
        model_config = getattr(self.model, "config", None)
        if not self.config.use_soft_labels and self.config.soft_label_loss != "ce":
            raise ValueError("Hard or soft_to_hard data only supports label_training_strategy='ce'.")
        if self.config.soft_label_loss == "rel" and not self.config.use_soft_labels:
            raise ValueError("ReL requires soft data with multiple annotation_labels per example.")
        effective_max_length = resolve_max_length(self.tokenizer, model_config, self.config.max_length)

        def tokenize_function(examples: dict) -> dict:
            tokenizer_kwargs = {
                "padding": False,
                "truncation": True,
                "max_length": effective_max_length,
            }

            return self.tokenizer(
                examples["text_a"],
                examples["text_b"],
                **tokenizer_kwargs,
            )

        def extract_soft_labels(samples: List[Union[TextPairClassificationSample, TextPairDistributionSample]]) -> List[List[float]]:
            labels: List[List[float]] = []
            for s in samples:
                if not isinstance(s, (TextPairDistributionSample, SingleTextDistributionSample)) or not s.human_dist:
                    raise ValueError("Soft-label training requires samples with a non-empty human_dist.")
                if len(s.human_dist) != self.config.num_labels:
                    raise ValueError(
                        f"Expected human_dist length {self.config.num_labels}, got {len(s.human_dist)} for sample {s.id}."
                    )
                labels.append([float(x) for x in s.human_dist])
            return labels

        def extract_multilevel_labels(samples: List[MultilevelInputSample], use_soft: bool) -> dict[str, List]:
            label_columns: dict[str, List] = {}
            prefix = "soft_labels" if use_soft else "hard_labels"
            for level in MULTILEVEL_LEVEL_ORDER:
                key = f"{prefix}_{level}"
                values: List = []
                for sample in samples:
                    if use_soft:
                        dist = sample.human_dists.get(level)
                        if not dist:
                            raise ValueError(f"Missing {level} human_dist for sample {sample.id}.")
                        if self.config.multilevel_label_sizes is None:
                            raise ValueError("multilevel_label_sizes must come from the dataset manifest.")
                        expected = self.config.multilevel_label_sizes[level]
                        if len(dist) != expected:
                            raise ValueError(
                                f"Expected {level} human_dist length {expected}, got {len(dist)} for sample {sample.id}."
                            )
                        values.append([float(x) for x in dist])
                    else:
                        if level not in sample.hard_labels:
                            raise ValueError(f"Missing {level} hard label for sample {sample.id}.")
                        values.append(int(sample.hard_labels[level]))
                label_columns[key] = values
            return label_columns

        def extract_labels(samples) -> List[Union[int, List[float]]]:
            if self.config.head_type == "multilabel_classification":
                result = []
                for sample in samples:
                    if not isinstance(sample, SingleTextMultilabelDistributionSample):
                        raise ValueError("Multilabel heads require SingleTextMultilabelDistributionSample inputs.")
                    result.append(list(sample.human_probs) if self.config.use_soft_labels else list(sample.labels))
                return result
            if not self.config.use_soft_labels:
                return [s.label for s in samples]
            return extract_soft_labels(samples)

        def build_dataset(samples: List[Union[TextPairClassificationSample, TextPairDistributionSample, SingleTextClassificationSample, SingleTextDistributionSample, SingleTextMultilabelDistributionSample, MultilevelInputSample]]) -> Dataset:
            data = {
                "text_a": [s.text_a if isinstance(s, (TextPairClassificationSample, MultilevelSample)) else s.text for s in samples],
                "text_b": [s.text_b if isinstance(s, (TextPairClassificationSample, MultilevelSample)) else "" for s in samples],
            }
            if self.config.head_type == "multilevel_classification":
                if not samples or not isinstance(samples[0], (MultilevelSample, SingleTextMultilevelSample)):
                    raise ValueError("Multilevel heads require multilevel sample inputs.")
                multilevel_samples = [s for s in samples if isinstance(s, (MultilevelSample, SingleTextMultilevelSample))]
                data.update(extract_multilevel_labels(multilevel_samples, self.config.use_soft_labels))
                rel_training = self.config.use_soft_labels and self.config.soft_label_loss == "rel"
                if self.config.use_soft_eval_metrics or rel_training:
                    # Keep both columns: compute_loss reads hard labels, while
                    # label_names makes the soft labels available to metrics.
                    # ReL starts with soft columns above, so add the hard
                    # labels required by its training/evaluation loss.
                    data.update(extract_multilevel_labels(multilevel_samples, not self.config.use_soft_labels))
            else:
                data["labels"] = extract_labels(samples)  # type: ignore[arg-type]
                if self.config.use_soft_eval_metrics:
                    if self.config.head_type == "multilabel_classification":
                        # Multi-label samples store independent annotation
                        # probabilities in ``human_probs``, rather than the
                        # categorical ``human_dist`` used by single-label data.
                        data["soft_labels"] = [
                            list(sample.human_probs)
                            for sample in samples
                            if isinstance(sample, SingleTextMultilabelDistributionSample)
                        ]
                    else:
                        data["soft_labels"] = extract_soft_labels(samples)  # type: ignore[arg-type]
            return Dataset.from_dict(data).map(tokenize_function, batched=True)

        def build_rel_dataset(samples: List[Union[TextPairClassificationSample, TextPairDistributionSample]]) -> Dataset:
            text_a: List[str] = []
            text_b: List[str] = []
            labels: List[int] = []
            for sample in samples:
                if not isinstance(sample, (TextPairDistributionSample, SingleTextDistributionSample)) or not sample.annotation_labels:
                    raise ValueError("ReL requires soft text-pair samples with non-empty annotation_labels.")
                for label in sample.annotation_labels:
                    text_a.append(sample.text_a if isinstance(sample, TextPairDistributionSample) else sample.text)
                    text_b.append(sample.text_b if isinstance(sample, TextPairDistributionSample) else "")
                    labels.append(int(label))
            if not labels:
                raise ValueError("ReL requires at least one annotation label.")
            return Dataset.from_dict({"text_a": text_a, "text_b": text_b, "labels": labels}).map(
                tokenize_function, batched=True
            )

        def build_multilabel_rel_dataset(samples: List[SingleTextMultilabelDistributionSample]) -> Dataset:
            text_a: List[str] = []
            labels: List[List[int]] = []
            for sample in samples:
                for annotation in sample.annotation_label_sets:
                    target = [0] * self.config.num_labels
                    for label in annotation:
                        target[label] = 1
                    text_a.append(sample.text)
                    labels.append(target)
            if not labels:
                raise ValueError("ReL requires at least one multilabel annotation set.")
            return Dataset.from_dict({"text_a": text_a, "text_b": [""] * len(text_a), "labels": labels}).map(tokenize_function, batched=True)

        def build_multilevel_rel_dataset(samples: List[MultilevelInputSample]) -> Dataset:
            text_a: List[str] = []
            text_b: List[str] = []
            label_columns = {f"hard_labels_{level}": [] for level in MULTILEVEL_LEVEL_ORDER}
            for sample in samples:
                votes = sample.annotation_labels
                if not isinstance(votes, dict) or not votes:
                    raise ValueError(f"ReL requires annotation_labels for multidimensional sample {sample.id}.")
                level_votes = []
                for level in MULTILEVEL_LEVEL_ORDER:
                    labels = votes.get(level)
                    if not isinstance(labels, list) or not labels:
                        raise ValueError(f"ReL requires non-empty {level} multidimensional annotation_labels for multilevel sample {sample.id}.")
                    level_votes.append(labels)
                vote_count = len(level_votes[0])
                if any(len(labels) != vote_count for labels in level_votes[1:]):
                    raise ValueError(f"Multilevel multidimensional annotation_labels must have the same count at every level for sample {sample.id}.")
                for vote_index in range(vote_count):
                    text_a.append(sample.text if isinstance(sample, SingleTextMultilevelSample) else sample.text_a)
                    text_b.append("" if isinstance(sample, SingleTextMultilevelSample) else sample.text_b)
                    for level, labels in zip(MULTILEVEL_LEVEL_ORDER, level_votes):
                        label_columns[f"hard_labels_{level}"].append(int(labels[vote_index]))
            if not text_a:
                raise ValueError("ReL requires at least one multilevel annotation vote.")
            return Dataset.from_dict({"text_a": text_a, "text_b": text_b, **label_columns}).map(
                tokenize_function, batched=True
            )

        if self.config.soft_label_loss == "rel":
            if self.config.head_type == "multilevel_classification":
                train_dataset = build_multilevel_rel_dataset([sample for sample in train_samples if isinstance(sample, (MultilevelSample, SingleTextMultilevelSample))])
            elif self.config.head_type == "multilabel_classification":
                train_dataset = build_multilabel_rel_dataset([sample for sample in train_samples if isinstance(sample, SingleTextMultilabelDistributionSample)])
            else:
                train_dataset = build_rel_dataset(train_samples)
        else:
            train_dataset = build_dataset(train_samples)

        eval_dataset = None
        if eval_samples:
            eval_dataset = build_dataset(eval_samples)

        return train_dataset, eval_dataset

    def train(
        self,
        train_samples: List[Union[TextPairClassificationSample, TextPairDistributionSample, SingleTextClassificationSample, SingleTextDistributionSample, SingleTextMultilabelDistributionSample, MultilevelInputSample]],
        eval_samples: Optional[List[Union[TextPairClassificationSample, TextPairDistributionSample, SingleTextClassificationSample, SingleTextDistributionSample, SingleTextMultilabelDistributionSample, MultilevelInputSample]]] = None,
    ) -> None:
        if self.model is None or self.tokenizer is None:
            self.initialize_model()

        print("Preparing tokenized datasets...", flush=True)
        train_dataset, eval_dataset = self.prepare_dataset(train_samples, eval_samples)

        def _prepare_multilevel_predictions(predictions, labels):
            if self.config.head_type == "multilevel_classification":
                level_probs = {
                    level: torch.softmax(torch.tensor(predictions[idx]), dim=-1).cpu().numpy()
                    for idx, level in enumerate(MULTILEVEL_LEVEL_ORDER)
                }
            else:
                raise ValueError(f"Unsupported multilevel head type: {self.config.head_type}")
            level_labels = {level: np.asarray(labels[idx]) for idx, level in enumerate(MULTILEVEL_LEVEL_ORDER)}
            return level_probs, level_labels

        def _metrics_from_level_probs(level_probs: dict[str, np.ndarray], level_labels: dict[str, np.ndarray]) -> dict:
            metrics: dict[str, float] = {}
            accuracies: List[float] = []
            tvrs: List[float] = []
            kls: List[float] = []

            for level in MULTILEVEL_LEVEL_ORDER:
                probs = np.asarray(level_probs[level], dtype=np.float32)
                labels = np.asarray(level_labels[level])
                pred_labels = probs.argmax(axis=-1)

                if self.config.use_soft_labels or self.config.use_soft_eval_metrics:
                    labels = labels.astype(np.float32)
                    labels = labels / np.clip(labels.sum(axis=-1, keepdims=True), a_min=1e-8, a_max=None)
                    true_labels = labels.argmax(axis=-1)
                    accuracy = float((pred_labels == true_labels).mean())
                    eps = 1e-8
                    kl_div = float(np.mean(np.sum(labels * (np.log(labels + eps) - np.log(probs + eps)), axis=-1)))
                    tvd = float(np.mean(0.5 * np.sum(np.abs(probs - labels), axis=-1)))
                    metrics[f"{level}_accuracy"] = accuracy
                    metrics[f"{level}_kl_divergence"] = kl_div
                    metrics[f"{level}_tvd"] = tvd
                    accuracies.append(accuracy)
                    kls.append(kl_div)
                    tvrs.append(tvd)
                else:
                    accuracy = float((pred_labels == labels).mean())
                    metrics[f"{level}_accuracy"] = accuracy
                    accuracies.append(accuracy)

            metrics["accuracy"] = float(np.mean(accuracies)) if accuracies else 0.0
            if self.config.use_soft_labels or self.config.use_soft_eval_metrics:
                metrics["kl_divergence"] = float(np.mean(kls)) if kls else 0.0
                metrics["tvd"] = float(np.mean(tvrs)) if tvrs else 0.0
            return metrics

        def compute_metrics(eval_pred: EvalPrediction) -> dict:
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids

            if self.config.head_type == "multilabel_classification":
                pred_probs = torch.sigmoid(torch.tensor(predictions)).cpu().numpy()
                targets = np.asarray(labels, dtype=np.float32)
                pred_hard, target_hard = pred_probs >= 0.5, targets >= 0.5
                denom = pred_hard.sum() + target_hard.sum()
                micro_f1 = float(2 * np.logical_and(pred_hard, target_hard).sum() / denom) if denom else 1.0
                macro_denominators = pred_hard.sum(axis=0) + target_hard.sum(axis=0)
                macro_f1 = float(np.mean(np.divide(
                    2 * np.logical_and(pred_hard, target_hard).sum(axis=0),
                    macro_denominators,
                    out=np.zeros_like(macro_denominators, dtype=float),
                    where=macro_denominators != 0,
                )))
                accuracy = float(np.all(pred_hard == target_hard, axis=-1).mean())
                return {
                    "accuracy": accuracy, "micro_f1": micro_f1, "macro_f1": macro_f1,
                    "soft_micro_f1": compute_soft_micro_f1(pred_probs, targets),
                    "soft_macro_f1": compute_soft_macro_f1(pred_probs, targets),
                    "multilabel_pojsd": compute_multilabel_pojsd(pred_probs, targets),
                    "multilabel_entropy_correlation": compute_multilabel_entropy_correlation(pred_probs, targets),
                }
            if self.config.head_type == "multilevel_classification":
                level_probs, level_labels = _prepare_multilevel_predictions(predictions, labels)
                return _metrics_from_level_probs(level_probs, level_labels)

            pred_probs = torch.softmax(torch.tensor(predictions), dim=-1).cpu().numpy()

            pred_labels = pred_probs.argmax(axis=-1)

            if self.config.use_soft_labels or self.config.use_soft_eval_metrics:
                labels = np.asarray(labels, dtype=np.float32)
                labels = labels / np.clip(labels.sum(axis=-1, keepdims=True), a_min=1e-8, a_max=None)
                true_labels = labels.argmax(axis=-1)
                accuracy = float((pred_labels == true_labels).mean())

                eps = 1e-8
                kl_div = float(np.mean(np.sum(labels * (np.log(labels + eps) - np.log(pred_probs + eps)), axis=-1)))
                tvd = float(np.mean(0.5 * np.sum(np.abs(pred_probs - labels), axis=-1)))
                metrics = {
                    "accuracy": accuracy, "tvd": tvd, "kl_divergence": kl_div,
                }
                # Soft F1 is a derived, opt-in metric for categorical label
                # distributions. Still emit the selected one so Trainer can
                # use it for checkpoint selection.
                if self.config.soft_label_metric_for_best_model == "soft_micro_f1":
                    metrics["soft_micro_f1"] = compute_soft_micro_f1(pred_probs, labels)
                elif self.config.soft_label_metric_for_best_model == "soft_macro_f1":
                    metrics["soft_macro_f1"] = compute_soft_macro_f1(pred_probs, labels)
                return metrics

            labels = np.asarray(labels)
            accuracy = float((pred_labels == labels).mean())
            return {"accuracy": accuracy}

        print("Creating training loop...", flush=True)
        training_args = self.config.to_training_args()
        trainer = SoftLabelTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[
                ThroughputLoggingCallback(
                    train_batch_size=self.config.train_batch_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                )
            ],
            use_soft_labels=self.config.use_soft_labels,
            soft_label_loss=self.config.soft_label_loss,
            head_type=self.config.head_type,
        )

        print("Starting training...")
        trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)
        if eval_dataset is not None:
            _print_dev_metric_plot(trainer, self.config.soft_label_metric_for_best_model)

        output_path = Path(self.config.output_dir) / "final_model"
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"Model saved to {output_path}")

    def save_model(self, path: str) -> None:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not initialized. Call train() or initialize_model() first.")

        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(target)
        self.tokenizer.save_pretrained(target)
        print(f"Model saved to {target}")

    def load_model(self, path: str) -> None:
        model_dir = Path(path)
        self.tokenizer = load_tokenizer_with_fallback(model_dir)

        multilevel_class_meta = model_dir / MultiLevelClassificationModel.META_FILENAME
        if multilevel_class_meta.exists():
            with open(multilevel_class_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            level_num_labels = meta.get("level_num_labels")
            if level_num_labels is None:
                raise ValueError("Saved multilevel model metadata is missing level_num_labels.")
            if not isinstance(level_num_labels, dict) or not level_num_labels:
                raise ValueError("Saved multilevel model metadata must define a non-empty level_num_labels mapping.")
            global MULTILEVEL_LEVEL_ORDER
            MULTILEVEL_LEVEL_ORDER = tuple(level_num_labels)
            self.model = MultiLevelClassificationModel.from_pretrained(
                str(model_dir),
                level_num_labels={level: int(size) for level, size in level_num_labels.items()},
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        print(f"Model loaded from {model_dir}")
