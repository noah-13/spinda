from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from datasets import Dataset  # type: ignore
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction

from nli_toolkits.data.schemas import NLISample, NLI_NUM_LABELS


@dataclass
class TrainingConfig:
    """Configuration for NLI model training."""

    # Model configuration
    model_name_or_path: str = "roberta-base"  # or "bert-base-uncased"
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    train_batch_size: int = 32
    eval_batch_size: int = 64
    num_epochs: int = 3
    warmup_ratio: float = 0.1  # 10% of training steps for warmup
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1  # Effective batch size = train_batch_size * gradient_accumulation_steps
    
    # Data paths
    output_dir: str = "./outputs"
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_total_limit: int = 3
    
    # Tokenization settings
    max_length: int = 128
    
    # Other settings
    seed: int = 42
    fp16: bool = False
    dataloader_num_workers: int = 4
    
    # Wandb settings
    use_wandb: bool = False
    wandb_project: str = "nli-toolkits"
    wandb_run_name: Optional[str] = None
    
    def to_training_args(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        
        # Setup wandb reporting
        report_to = []
        if self.use_wandb:
            # Check if wandb is installed
            try:
                import wandb  # noqa: F401
            except ImportError:
                raise ImportError(
                    "wandb is required when --use_wandb is enabled. "
                    "Install it with: uv pip install wandb "
                    "or: uv sync --extra dev"
                )
            report_to.append("wandb")
            # Set wandb environment variables if run_name is provided
            if self.wandb_run_name:
                os.environ["WANDB_RUN_NAME"] = self.wandb_run_name
            os.environ["WANDB_PROJECT"] = self.wandb_project
        
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            eval_strategy=self.eval_strategy,
            save_strategy=self.save_strategy,
            save_total_limit=self.save_total_limit,
            seed=self.seed,
            fp16=self.fp16,
            dataloader_num_workers=self.dataloader_num_workers,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            logging_steps=100,
            save_steps=500,
            report_to=report_to if report_to else None,
        )


class NLITrainer:
    """
    Trainer for fine-tuning BERT/RoBERTa on NLI tasks.
    
    Handles:
    - Model and tokenizer initialization
    - Dataset preparation (tokenization)
    - Training loop
    - Model saving
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.tokenizer = None
        self.model = None

    def initialize_model(self) -> None:
        """Initialize tokenizer and model."""
        model_name = self.config.model_name_or_path
        
        print(f"Loading tokenizer and model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=NLI_NUM_LABELS,
        )

    def prepare_dataset(
        self,
        train_samples: List[NLISample],
        eval_samples: Optional[List[NLISample]] = None,
    ) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Convert NLISample lists to tokenized datasets.
        
        Args:
            train_samples: Training samples
            eval_samples: Optional evaluation samples
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        if self.tokenizer is None:
            raise ValueError("Must call initialize_model() first")

        def tokenize_function(examples: dict) -> dict:
            """Tokenize premise and hypothesis."""
            return self.tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                padding="longest",
                max_length=self.config.max_length,
            )

        # Convert samples to dict format
        train_dict = {
            "premise": [s.premise for s in train_samples],
            "hypothesis": [s.hypothesis for s in train_samples],
            "label": [s.label for s in train_samples],
        }
        train_dataset = Dataset.from_dict(train_dict)
        train_dataset = train_dataset.map(tokenize_function, batched=True)

        eval_dataset = None
        if eval_samples:
            eval_dict = {
                "premise": [s.premise for s in eval_samples],
                "hypothesis": [s.hypothesis for s in eval_samples],
                "label": [s.label for s in eval_samples],
            }
            eval_dataset = Dataset.from_dict(eval_dict)
            eval_dataset = eval_dataset.map(tokenize_function, batched=True)

        return train_dataset, eval_dataset

    def train(
        self,
        train_samples: List[NLISample],
        eval_samples: Optional[List[NLISample]] = None,
    ) -> None:
        """
        Train the model.
        
        Args:
            train_samples: Training samples
            eval_samples: Optional evaluation samples for validation
        """
        if self.model is None or self.tokenizer is None:
            self.initialize_model()

        # Prepare datasets
        train_dataset, eval_dataset = self.prepare_dataset(train_samples, eval_samples)

        # Compute metrics function
        def compute_metrics(eval_pred: EvalPrediction) -> dict:
            """Compute accuracy."""
            predictions, labels = eval_pred
            predictions = predictions.argmax(axis=-1)
            accuracy = (predictions == labels).mean()
            return {"accuracy": accuracy}

        # Create HuggingFace Trainer
        training_args = self.config.to_training_args()
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        # Train
        print("Starting training...")
        trainer.train()

        # Save final model
        output_path = Path(self.config.output_dir) / "final_model"
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"Model saved to {output_path}")

    def save_model(self, path: str) -> None:
        """Save model and tokenizer to disk."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not initialized. Call train() or initialize_model() first.")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load model and tokenizer from disk."""
        path = Path(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        print(f"Model loaded from {path}")
