#!/usr/bin/env python3
"""
Training script for NLI models.

Example usage:
    python -m nli_toolkits.scripts.train \
        --model roberta-base \
        --output_dir ./outputs/roberta_snli \
        --num_epochs 3
"""

import argparse
from pathlib import Path

from nli_toolkits.data import ChaosNLIReader, SNLIReader
from nli_toolkits.models import NLITrainer, TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BERT/RoBERTa on NLI")

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="roberta-base",
        choices=["roberta-base", "bert-base-uncased"],
        help="Pre-trained model to fine-tune",
    )

    # Data configuration
    parser.add_argument(
        "--data_source",
        type=str,
        default="snli",
        choices=["snli", "chaosnli"],
        help="Training data source",
    )
    parser.add_argument(
        "--chaosnli_train_path",
        type=str,
        default=None,
        help="Path to ChaosNLI train JSONL (required when --data_source chaosnli)",
    )
    parser.add_argument(
        "--chaosnli_dev_path",
        type=str,
        default=None,
        help="Optional path to ChaosNLI dev JSONL; if omitted, no eval set is used",
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
        default=128,
        help="Maximum sequence length for tokenization",
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
        default="nli-toolkits",
        help="Wandb project name",
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
    if args.data_source == "snli":
        if args.use_soft_labels:
            raise ValueError("SNLIReader only provides hard labels. Use --data_source chaosnli for soft-label training.")
        reader = SNLIReader()
        train_samples = reader.load_train()
        eval_samples = reader.load_dev()
    else:
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
            wandb_run_name = f"{args.model}_seed_{seed}"

        # Create training config for this seed
        config = TrainingConfig(
            model_name_or_path=args.model,
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
            output_dir=str(seed_output_dir),
            seed=seed,
            fp16=args.fp16,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
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
