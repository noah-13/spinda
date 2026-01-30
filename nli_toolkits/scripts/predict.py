#!/usr/bin/env python3
"""
Prediction script for NLI models.

Example usage:
    python -m nli_toolkits.scripts.predict \
        --model_path ./outputs/roberta_snli/final_model \
        --output_file predictions.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from nli_toolkits.data import ChaosNLIReader, SNLIReader
from nli_toolkits.data.schemas import PredictionRecord


def predict_batch(
    model,
    tokenizer,
    premises: List[str],
    hypotheses: List[str],
    device: str = "cuda",
    max_length: int = 128,
) -> Tuple[List[List[float]], List[int]]:
    """
    Generate predictions for a batch of premise-hypothesis pairs.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        premises: List of premise strings
        hypotheses: List of hypothesis strings
        device: Device to run inference on
        
    Returns:
        Tuple of (probabilities list, predicted labels list)
    """
    model.eval()
    
    # Tokenize
    inputs = tokenizer(
        premises,
        hypotheses,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
    
    # Convert to CPU numpy
    probs_np = probs.cpu().numpy()
    preds_np = preds.cpu().numpy()
    
    return probs_np.tolist(), preds_np.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions with trained NLI model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory",
    )
    
    parser.add_argument(
        "--data_source",
        type=str,
        default="snli",
        choices=["snli", "chaosnli"],
        help="Data source to predict on",
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Data split to predict on",
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default="predictions.jsonl",
        help="Output file path (JSONL format)",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization",
    )

    parser.add_argument(
        "--chaosnli_path",
        type=str,
        default=None,
        help="Path to ChaosNLI JSONL file (required if data_source=chaosnli)",
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(args.device)
    print("Model loaded successfully")
    
    # Load data
    print(f"Loading {args.data_source} {args.split} split...")
    if args.data_source == "snli":
        reader = SNLIReader()
        samples = reader.load_split(args.split)
    elif args.data_source == "chaosnli":
        if args.chaosnli_path is None:
            raise ValueError("--chaosnli_path is required when data_source=chaosnli")
        reader = ChaosNLIReader(data_path=args.chaosnli_path)
        samples = reader.load_split(args.split)
    else:
        raise ValueError(f"Unknown data source: {args.data_source}")
    
    print(f"Loaded {len(samples)} samples")
    
    # Generate predictions
    predictions = []
    batch_size = args.batch_size
    
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i : i + batch_size]
        premises = [s.premise for s in batch_samples]
        hypotheses = [s.hypothesis for s in batch_samples]
        
        probs_list, preds_list = predict_batch(
            model,
            tokenizer,
            premises,
            hypotheses,
            args.device,
            max_length=args.max_length,
        )
        
        for sample, probs, pred in zip(batch_samples, probs_list, preds_list):
            pred_record = PredictionRecord(
                id=sample.id,
                task=sample.task,
                split=sample.split,
                source=sample.source,
                outputs={
                    "probs": probs,
                    "pred": int(pred),
                },
            )
            predictions.append(pred_record)
        
        if (i + batch_size) % (batch_size * 10) == 0:
            print(f"Processed {i + batch_size}/{len(samples)} samples")
    
    print(f"Generated {len(predictions)} predictions")
    
    # Save predictions
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            # Convert to dict for JSON serialization
            pred_dict = {
                "id": pred.id,
                "task": pred.task,
                "split": pred.split,
                "source": pred.source,
                "outputs": pred.outputs,
            }
            f.write(json.dumps(pred_dict) + "\n")
    
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
