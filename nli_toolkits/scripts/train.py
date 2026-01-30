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

from nli_toolkits.data import SNLIReader
from nli_toolkits.models import NLITrainer, TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BERT/RoBERTa on SNLI")
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="roberta-base",
        choices=["roberta-base", "bert-base-uncased"],
        help="Pre-trained model to fine-tune",
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
    print("Loading SNLI dataset...")
    reader = SNLIReader()
    train_samples = reader.load_train()
    eval_samples = reader.load_dev()
    
    print(f"Loaded {len(train_samples)} training samples")
    print(f"Loaded {len(eval_samples)} evaluation samples")
    
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
