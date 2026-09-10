#!/usr/bin/env python3
"""
Training script for HLV classification models.

Example usage:
    python -m hlv_toolkits.scripts.train \
        --model roberta-base \
        --output_dir ./outputs/roberta_snli \
        --num_epochs 3
"""

import argparse
from pathlib import Path
import json
import warnings
from typing import Any

import torch

from hlv_toolkits.data import (
    TextPairMultilevelJSONLReader,
    TextPairClassificationJSONLReader,
    SingleTextClassificationJSONLReader,
    SingleTextMultilabelJSONLReader,
    SingleTextMultilevelJSONLReader,
)
from hlv_toolkits.models import HLVTrainer, TrainingConfig


DATA_FORMAT_SPECS = {
    "text_pair_label_distribution": ("text_pair", "classification"),
    "single_text_label_distribution": ("single_text", "classification"),
    "single_text_multilabel_annotation_distribution": ("single_text_multilabel", "multilabel_classification"),
    "text_pair_multilevel_label_distribution": ("multilevel", "multilevel_classification"),
    "single_text_multilevel_label_distribution": ("multilevel", "multilevel_classification"),
}


def _load_json_config(config_path: str, valid_keys: set[str]) -> dict[str, Any]:
    path = Path(config_path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Training config not found: {path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in training config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Training config {path} must be a JSON object.")

    # JSON uses the public field name `format`; argparse stores it as
    # `data_format` to avoid shadowing Python's built-in format function.
    if "format" in config:
        if "data_format" in config:
            raise ValueError("Use only 'format', not both 'format' and 'data_format', in a training config.")
        config["data_format"] = config.pop("format")
    if "label_training_strategy" in config:
        if "soft_label_loss" in config:
            raise ValueError("Use only 'label_training_strategy', not both label_training_strategy and soft_label_loss.")
        config["soft_label_loss"] = config.pop("label_training_strategy")
    unknown = sorted(set(config) - valid_keys)
    if unknown:
        raise ValueError(f"Unknown training config keys in {path}: {', '.join(unknown)}")
    return config


def _load_merged_json_configs(config_paths: list[str], valid_keys: set[str]) -> dict[str, Any]:
    """Load JSON configuration layers from lowest to highest priority."""
    merged: dict[str, Any] = {}
    for config_path in config_paths:
        merged.update(_load_json_config(config_path, valid_keys))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BERT/RoBERTa on HLV classification tasks")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        action="extend",
        default=[],
        metavar="PATH",
        help="One or more JSON configuration files. Files are merged left to right; explicit CLI values override all JSON values.",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="roberta-base",
        help="Pre-trained model to fine-tune",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto, cpu, cuda, or a specific GPU such as cuda:1",
    )

    # Data configuration
    parser.add_argument(
        "--data_source",
        type=str,
        default="text_pair",
        choices=["text_pair", "single_text", "single_text_multilabel", "multilevel"],
        help="Training data source; text_pair and multilevel use JSONL dataset contracts",
    )
    parser.add_argument(
        "--format",
        dest="data_format",
        type=str,
        default=None,
        choices=["text_pair_label_distribution", "single_text_label_distribution", "single_text_multilabel_annotation_distribution", "text_pair_multilevel_label_distribution", "single_text_multilevel_label_distribution"],
        help="Required format for direct JSONL training data.",
    )
    parser.add_argument("--train_path", type=str, default=None, help="Required JSONL training file for direct text-pair training.")
    parser.add_argument("--dev_path", type=str, default=None, help="Optional JSONL development file; omit to train without evaluation.")
    parser.add_argument("--labels", type=str, nargs="+", default=None, help="Optional label names in numeric-index order.")
    parser.add_argument(
        "--level_labels",
        type=json.loads,
        default=None,
        help="Optional level-to-label-list mapping; when provided, it must match the multilevel dataset manifest.",
    )
    parser.add_argument(
        "--label_mode",
        dest="label_mode",
        type=str,
        default=None,
        choices=["hard", "soft", "soft_to_hard"],
        help="Expected data label mode; soft_to_hard converts annotation vote counts, resolving ties with a reproducible hash-based random choice.",
    )
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (used if warmup_steps is None)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=0,
        help="Maximum sequence length for tokenization (set to 0 to use the model limit)",
    )

    # Soft-label training settings
    parser.add_argument(
        "--use_soft_labels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Train with distribution targets (requires data with human label distributions)",
    )
    parser.add_argument(
        "--label_training_strategy",
        dest="soft_label_loss",
        type=str,
        default="ce",
        choices=["ce", "mse", "jsd", "rel"],
    )
    parser.add_argument(
        "--soft_label_metric_for_best_model",
        type=str,
        default="tvd",
        choices=["kl_divergence", "tvd", "accuracy"],
        help="Metric used to select best checkpoint when --use_soft_labels is enabled",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume a seed run from a Hugging Face checkpoint directory.",
    )

    # Other settings
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="List of random seeds for training (default: 42)",
    )
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False, help="Use mixed precision training")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help="Number of DataLoader worker processes (use 0 to disable multiprocessing)",
    )

    # Wandb settings
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=False, help="Enable Weights & Biases logging")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="hlv",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb entity/team name",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Wandb run group, e.g. dataset name",
    )
    parser.add_argument(
        "--wandb_job_type",
        type=str,
        default=None,
        help="Wandb job type, e.g. screening/train",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name (default: auto-generated)",
    )

    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", type=str, nargs="+", action="extend", default=[])
    bootstrap_args, _ = bootstrap_parser.parse_known_args()
    valid_config_keys = {action.dest for action in parser._actions} - {"config", "help"}
    config_values = _load_merged_json_configs(bootstrap_args.config, valid_config_keys)
    args = parser.parse_args(namespace=argparse.Namespace(**config_values))
    direct_dataset_options = (args.data_format, args.train_path, args.dev_path, args.labels, args.level_labels, args.label_mode)
    if any(value is not None for value in direct_dataset_options):
        if args.data_source not in {"text_pair", "single_text", "single_text_multilabel", "multilevel"}:
            raise ValueError("format/train_path/dev_path/labels/level_labels/label_mode are only valid for direct JSONL training.")
    if args.data_format not in DATA_FORMAT_SPECS:
        raise ValueError("format is required and must identify a supported dataset contract.")
    args.data_source, args.head_type = DATA_FORMAT_SPECS[args.data_format]
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but CUDA is not available.")
        if device == "cuda":
            device = "cuda:0"
        try:
            torch.cuda.set_device(torch.device(device))
        except (RuntimeError, ValueError, IndexError) as exc:
            raise ValueError(f"Invalid CUDA device '{device}'. Available GPU count: {torch.cuda.device_count()}") from exc
    elif device != "cpu":
        raise ValueError("--device must be one of auto, cpu, cuda, or cuda:<index>")
    print(f"Using device: {device}")

    # Load data
    print(f"Loading dataset from source: {args.data_source}")
    num_labels = 0
    label_names = None
    multilevel_head = args.data_source == "multilevel"
    multilevel_label_sizes = None

    if args.data_source == "text_pair":
        if args.data_format != "text_pair_label_distribution":
            raise ValueError("--format text_pair_label_distribution is required for text-pair training.")
        if not args.train_path:
            raise ValueError("--train_path is required for text-pair training.")
        reader = TextPairClassificationJSONLReader(
            data_format=args.data_format,
            train_path=args.train_path,
            dev_path=args.dev_path,
            labels=args.labels,
            label_mode=args.label_mode,
        )
        num_labels = len(reader.labels)
        label_names = reader.labels
        if args.use_soft_labels is not None and args.use_soft_labels != reader.use_soft_labels:
            warnings.warn("--use_soft_labels is overridden by the resolved dataset label_mode.", UserWarning, stacklevel=2)
        args.use_soft_labels = reader.use_soft_labels
        train_samples = reader.load_train()
        if reader.has_dev:
            eval_samples = reader.load_dev()
        else:
            eval_samples = None
            warnings.warn("No dev_path was provided: evaluation is disabled and final_model will be the last epoch model.", UserWarning, stacklevel=2)
    elif args.data_source == "single_text":
        if args.data_format != "single_text_label_distribution" or not args.train_path:
            raise ValueError("--format single_text_label_distribution and --train_path are required for single-text training.")
        reader = SingleTextClassificationJSONLReader(data_format=args.data_format, train_path=args.train_path, dev_path=args.dev_path, labels=args.labels, label_mode=args.label_mode)
        num_labels, label_names, args.use_soft_labels = len(reader.labels), reader.labels, reader.use_soft_labels
        train_samples = reader.load_train()
        eval_samples = reader.load_dev() if reader.has_dev else None
    elif args.data_source == "single_text_multilabel":
        if args.data_format != "single_text_multilabel_annotation_distribution" or not args.train_path:
            raise ValueError("--format single_text_multilabel_annotation_distribution and --train_path are required for MFRC training.")
        if args.label_mode not in {"soft", "soft_to_hard"}:
            raise ValueError("MFRC requires --label_mode soft (probability targets) or soft_to_hard (per-label majority vote).")
        if args.label_mode == "soft_to_hard" and args.soft_label_loss != "ce":
            raise ValueError("MFRC soft_to_hard training supports only --label_training_strategy ce.")
        reader = SingleTextMultilabelJSONLReader(data_format=args.data_format, train_path=args.train_path, dev_path=args.dev_path, labels=args.labels)
        num_labels, label_names = len(reader.labels), reader.labels
        args.use_soft_labels = args.label_mode == "soft"
        train_samples = reader.load_train()
        eval_samples = reader.load_dev() if reader.has_dev else None
    elif args.data_source == "multilevel":
        if args.data_format not in {"text_pair_multilevel_label_distribution", "single_text_multilevel_label_distribution"}:
            raise ValueError("--format text_pair_multilevel_label_distribution or single_text_multilevel_label_distribution is required for multilevel JSONL training.")
        if not args.train_path:
            raise ValueError("--train_path is required for multilevel JSONL training.")
        if not multilevel_head:
            raise ValueError("A multilevel label-distribution format requires a multilevel head.")
        if args.label_mode not in {None, "soft", "hard", "soft_to_hard"}:
            raise ValueError("Multilevel label-distribution formats support label_mode soft, hard, or soft_to_hard.")
        args.use_soft_labels = args.label_mode == "soft"
        reader_cls = SingleTextMultilevelJSONLReader if args.data_format == "single_text_multilevel_label_distribution" else TextPairMultilevelJSONLReader
        reader = reader_cls(data_path=args.train_path)
        if args.level_labels is not None:
            configured_labels = reader._validate_level_labels(args.level_labels)
            if configured_labels != reader.level_labels:
                raise ValueError("level_labels must exactly match the dataset manifest.")
        multilevel_label_sizes = tuple(len(reader.level_labels[level]) for level in reader.LEVEL_ORDER)
        train_samples = reader.load_train()
        if args.dev_path:
            eval_reader = reader_cls(
                data_path=args.dev_path,
                level_labels=reader.level_labels,
            )
            eval_samples = eval_reader.load_dev()
        else:
            eval_samples = None
            warnings.warn("No dev_path was provided: evaluation is disabled and final_model will be the last epoch model.", UserWarning, stacklevel=2)
    else:
        raise ValueError(f"Unknown direct data source: {args.data_source}")

    args.use_soft_labels = bool(args.use_soft_labels)

    if not args.use_soft_labels and args.soft_label_loss != "ce":
        raise ValueError("Hard and soft_to_hard data only supports --label_training_strategy ce.")
    if args.soft_label_loss == "rel" and not args.use_soft_labels:
        raise ValueError("ReL requires soft data with multiple annotations per example.")

    print(f"Loaded {len(train_samples)} training samples")
    print(f"Loaded {0 if eval_samples is None else len(eval_samples)} evaluation samples")
    print(f"Using soft labels: {args.use_soft_labels}")

    # Train with multiple seeds
    seeds = args.seeds
    print(f"\nTraining with {len(seeds)} random seeds: {seeds}")

    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n{'='*60}")
        print(f"Training run {seed_idx}/{len(seeds)} with seed {seed}")
        print(f"{'='*60}")

        # Create output directory for this seed
        seed_output_dir = Path(args.output_dir) / f"seed_{seed}"

        # Create wandb run name if enabled
        wandb_run_name = args.wandb_run_name
        if args.use_wandb and wandb_run_name is None:
            wandb_run_name = f"{args.data_source}__{args.model}__seed_{seed}"

        # Create training config for this seed
        config = TrainingConfig(
            model_name_or_path=args.model,
            num_labels=num_labels,
            label_names=label_names,
            learning_rate=args.learning_rate,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            num_epochs=args.num_epochs,
            warmup_ratio=args.warmup_ratio,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_length=args.max_length,
            has_eval=eval_samples is not None,
            use_soft_labels=args.use_soft_labels,
            use_soft_eval_metrics=(args.data_source in {"text_pair", "single_text", "multilevel", "single_text_multilabel"} and args.label_mode == "soft_to_hard"),
            soft_label_loss=args.soft_label_loss,
            soft_label_metric_for_best_model=args.soft_label_metric_for_best_model,
            head_type=args.head_type,
            multilevel_label_sizes=multilevel_label_sizes,
            output_dir=str(seed_output_dir),
            resume_from_checkpoint=args.resume_from_checkpoint,
            seed=seed,
            fp16=args.fp16,
            device=device,
            dataloader_num_workers=args.dataloader_num_workers,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_group=args.wandb_group,
            wandb_job_type=args.wandb_job_type,
            wandb_run_name=wandb_run_name,
        )

        # Initialize trainer
        trainer = HLVTrainer(config)

        # Train
        trainer.train(train_samples, eval_samples)

        print(f"Training completed for seed {seed}!")

    print(f"\n{'='*60}")
    print(f"All requested seed runs completed! Results saved in: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
