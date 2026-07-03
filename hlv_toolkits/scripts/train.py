#!/usr/bin/env python3
"""
Training script for NLI/discourse models.

Example usage:
    python -m hlv_toolkits.scripts.train \
        --model roberta-base \
        --output_dir ./outputs/roberta_snli \
        --num_epochs 3
"""

import argparse
from pathlib import Path

from hlv_toolkits.data import (
    ChaosNLIReader,
    DiscoGeMReader,
    NLI_NUM_LABELS,
    ProcessedJSONLReader,
    SNLIReader,
    get_discogem_level_num_labels,
    get_discogem_num_labels,
)
from hlv_toolkits.models import NLITrainer, TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BERT/RoBERTa on NLI/discourse tasks")

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="roberta-base",
        help="Pre-trained model to fine-tune",
    )

    # Data configuration
    parser.add_argument(
        "--data_source",
        type=str,
        default="snli",
        choices=["snli", "chaosnli", "discogem", "processed"],
        help="Training data source",
    )
    parser.add_argument(
        "--chaosnli_train_path",
        type=str,
        default="data/external/chaosnli/chaosNLI_snli_train.jsonl",
        help="Path to ChaosNLI train JSONL (default: data/external/chaosnli/chaosNLI_snli_train.jsonl)",
    )
    parser.add_argument(
        "--chaosnli_dev_path",
        type=str,
        default="data/external/chaosnli/chaosNLI_snli_dev.jsonl",
        help="Path to ChaosNLI dev JSONL (default: data/external/chaosnli/chaosNLI_snli_dev.jsonl; set to empty or override to disable evaluation)",
    )
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="",
        help="Directory or file containing canonical JSONL samples",
    )
    parser.add_argument(
        "--processed_task",
        type=str,
        default="nli",
        choices=["nli", "discogem"],
        help="Task type stored in processed_data_dir",
    )
    parser.add_argument(
        "--discogem_path",
        type=str,
        default="",
        help="Path to the DiscoGeM 2.0 annotation archive. If empty, infer the default local path.",
    )
    parser.add_argument(
        "--discogem_version",
        type=str,
        default="auto",
        choices=["auto", "2.0"],
        help="DiscoGeM schema version",
    )
    parser.add_argument(
        "--discogem_label_mode",
        type=str,
        default="soft",
        choices=["soft", "hard"],
        help="Label mode for DiscoGeM training",
    )
    parser.add_argument(
        "--discogem_label_level",
        type=str,
        default="level2",
        choices=["level1", "level2", "level3", "all"],
        help="DiscoGeM label granularity",
    )
    parser.add_argument(
        "--discogem_language",
        type=str,
        default="en",
        choices=["en", "de", "fr", "cs"],
        help="DiscoGeM language slice to use for version 2.0",
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
        action="store_true",
        help="Train with distribution targets (requires data with human label distributions)",
    )
    parser.add_argument(
        "--soft_label_loss",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "kl_div"],
        help="Loss to use when --use_soft_labels is enabled",
    )
    parser.add_argument(
        "--soft_label_metric_for_best_model",
        type=str,
        default="kl_divergence",
        choices=["kl_divergence", "tvd", "accuracy"],
        help="Metric used to select best checkpoint when --use_soft_labels is enabled",
    )

    parser.add_argument(
        "--head_type",
        type=str,
        default="classification",
        choices=["classification", "joint_classification", "per_label_regression", "multilevel_classification", "multilevel_regression"],
        help="Model head type",
    )
    parser.add_argument(
        "--joint_loss_lambda",
        type=float,
        default=0.5,
        help="Lambda in L = lambda * CE_hard + (1-lambda) * KL_soft for joint_classification",
    )
    parser.add_argument(
        "--regression_loss",
        type=str,
        default="mse",
        choices=["mse", "bce"],
        help="Loss for per_label_regression head",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for model checkpoints",
    )

    # Other settings
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="List of random seeds for training (default: 42)",
    )
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")

    # Wandb settings
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
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

    args = parser.parse_args()

    # Load data
    print(f"Loading dataset from source: {args.data_source}")
    num_labels = NLI_NUM_LABELS
    multilevel_head = args.head_type in {"multilevel_classification", "multilevel_regression"}
    multilevel_label_sizes = None
    if multilevel_head and args.data_source not in {"discogem", "processed"}:
        raise ValueError("Multilevel DiscoGeM heads are only supported with DiscoGeM or processed DiscoGeM data.")
    if args.data_source == "processed" and multilevel_head and args.processed_task != "discogem":
        raise ValueError("Multilevel DiscoGeM heads require --processed_task discogem.")

    if args.data_source == "snli":
        if args.use_soft_labels:
            raise ValueError("SNLIReader only provides hard labels. Use --data_source chaosnli/discogem for soft-label training.")
        reader = SNLIReader()
        train_samples = reader.load_train()
        eval_samples = reader.load_dev()
    elif args.data_source == "chaosnli":
        if not args.chaosnli_train_path:
            raise ValueError("--chaosnli_train_path is required when --data_source chaosnli")

        train_reader = ChaosNLIReader(data_path=args.chaosnli_train_path)
        train_samples = train_reader.load_train()

        if args.chaosnli_dev_path:
            eval_reader = ChaosNLIReader(data_path=args.chaosnli_dev_path)
            eval_samples = eval_reader.load_dev()
        else:
            eval_samples = None
            print("No --chaosnli_dev_path provided: training without evaluation set.")
    elif args.data_source == "processed":
        if not args.processed_data_dir:
            raise ValueError("--processed_data_dir is required when --data_source processed")
        if args.processed_task == "discogem":
            if multilevel_head and args.discogem_label_level != "all":
                raise ValueError("Multilevel DiscoGeM heads require --discogem_label_level all.")
            if not multilevel_head and args.discogem_label_level == "all":
                raise ValueError("--discogem_label_level all is only valid for multilevel DiscoGeM heads.")
            num_labels = get_discogem_num_labels(args.discogem_label_level)
            multilevel_label_sizes = None
            if multilevel_head:
                level_sizes = get_discogem_level_num_labels()
                multilevel_label_sizes = tuple(level_sizes[level] for level in ("level1", "level2", "level3"))
            reader = ProcessedJSONLReader(
                data_path=args.processed_data_dir,
                task="discogem",
                label_level=args.discogem_label_level,
                label_mode=args.discogem_label_mode,
                language=args.discogem_language,
            )
        else:
            reader = ProcessedJSONLReader(data_path=args.processed_data_dir, task="nli")
        train_samples = reader.load_train()
        eval_samples = reader.load_dev()
    else:
        # DiscoGeM mode controls soft/hard labels; keep --use_soft_labels for compatibility.
        discogem_soft = args.discogem_label_mode == "soft"
        if args.use_soft_labels != discogem_soft:
            print(
                "Warning: --use_soft_labels overridden by --discogem_label_mode "
                f"({args.discogem_label_mode})."
            )
        args.use_soft_labels = discogem_soft
        multilevel_head = args.head_type in {"multilevel_classification", "multilevel_regression"}
        if multilevel_head and args.discogem_label_level != "all":
            raise ValueError("Multilevel DiscoGeM heads require --discogem_label_level all.")
        if not multilevel_head and args.discogem_label_level == "all":
            raise ValueError("--discogem_label_level all is only valid for multilevel DiscoGeM heads.")
        num_labels = get_discogem_num_labels(args.discogem_label_level)
        multilevel_label_sizes = None
        if multilevel_head:
            level_sizes = get_discogem_level_num_labels()
            multilevel_label_sizes = tuple(level_sizes[level] for level in ("level1", "level2", "level3"))

        reader = DiscoGeMReader(
            data_path=args.discogem_path or None,
            version=args.discogem_version,
            label_mode=args.discogem_label_mode,
            label_level=args.discogem_label_level,
            language=args.discogem_language,
        )
        train_samples = reader.load_train()
        eval_samples = reader.load_dev()

    if args.head_type == "joint_classification" and not args.use_soft_labels:
        raise ValueError("joint_classification requires soft distributions. Enable --use_soft_labels or use soft DiscoGeM mode.")

    print(f"Loaded {len(train_samples)} training samples")
    print(f"Loaded {0 if eval_samples is None else len(eval_samples)} evaluation samples")

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
            dataset_prefix = args.data_source
            if args.data_source == "discogem":
                dataset_prefix = f"discogem_{args.discogem_language}_{args.discogem_label_level}"
            wandb_run_name = f"{dataset_prefix}__{args.model}__seed_{seed}"

        # Create training config for this seed
        config = TrainingConfig(
            model_name_or_path=args.model,
            num_labels=num_labels,
            learning_rate=args.learning_rate,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            num_epochs=args.num_epochs,
            warmup_ratio=args.warmup_ratio,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_length=args.max_length,
            use_soft_labels=args.use_soft_labels,
            soft_label_loss=args.soft_label_loss,
            soft_label_metric_for_best_model=args.soft_label_metric_for_best_model,
            head_type=args.head_type,
            regression_loss=args.regression_loss,
            joint_loss_lambda=args.joint_loss_lambda,
            multilevel_label_sizes=multilevel_label_sizes,
            output_dir=str(seed_output_dir),
            seed=seed,
            fp16=args.fp16,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_group=args.wandb_group,
            wandb_job_type=args.wandb_job_type,
            wandb_run_name=wandb_run_name,
        )

        # Initialize trainer
        trainer = NLITrainer(config)

        # Train
        trainer.train(train_samples, eval_samples)

        print(f"Training completed for seed {seed}!")

    print(f"\n{'='*60}")
    print(f"All training runs completed! Results saved in: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
