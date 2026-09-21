#!/usr/bin/env python3
"""
Prediction script for NLI models.

Example usage:
    python -m hlv_toolkits.scripts.predict \
        --model_path ./outputs/roberta_snli/final_model \
        --output_file predictions.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from hlv_toolkits.data import (
    SingleTextClassificationJSONReader,
    SingleTextMultilabelJSONReader,
    SingleTextMultilevelJSONReader,
    TextPairClassificationJSONReader,
    TextPairMultilevelJSONReader,
)
from hlv_toolkits.data.schemas import PredictionRecord
from hlv_toolkits.data.json_io import write_records
from hlv_toolkits.models.trainer import (
    MULTILEVEL_LEVEL_ORDER,
    MultiLevelClassificationModel,
    HLVTrainer,
    TrainingConfig,
    resolve_max_length,
)


def _build_multilevel_outputs(level_probs: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
    batch_size = next(iter(level_probs.values())).shape[0] if level_probs else 0
    results: List[Dict[str, Any]] = []
    for row_idx in range(batch_size):
        row_levels: Dict[str, Dict[str, Any]] = {}
        for level in level_probs:
            probs = level_probs[level][row_idx]
            pred = int(torch.argmax(probs).item())
            row_levels[level] = {
                "probs": probs.cpu().tolist(),
                "pred": pred,
            }
        results.append({"dimensions": row_levels})
    return results


def predict_batch(
    model,
    tokenizer,
    premises: List[str],
    hypotheses: List[str] | None,
    device: str = "cuda",
    max_length: int = 0,
) -> List[Dict[str, Any]]:
    """Generate predictions for a batch of premise-hypothesis pairs."""
    model.eval()

    model_config = getattr(model, "config", None)
    effective_max_length = resolve_max_length(tokenizer, model_config, max_length)
    tokenizer_kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "max_length": effective_max_length,
    }

    inputs = tokenizer(premises, hypotheses, **tokenizer_kwargs) if hypotheses is not None else tokenizer(premises, **tokenizer_kwargs)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        if isinstance(model, MultiLevelClassificationModel):
            level_probs = {
                level: torch.softmax(logit, dim=-1)
                for level, logit in zip(model.level_num_labels, logits)
            }
            return _build_multilevel_outputs(level_probs)

        multilabel = getattr(model_config, "problem_type", None) == "multi_label_classification"
        probs = torch.sigmoid(logits) if multilabel else torch.softmax(logits, dim=-1)
        preds = (probs >= 0.5).long() if multilabel else torch.argmax(probs, dim=-1)

    probs_np = probs.cpu().numpy()
    preds_np = preds.cpu().numpy()

    return [
        {"probs": probs_np[idx].tolist(), "pred": preds_np[idx].tolist() if preds_np.ndim == 2 else int(preds_np[idx])}
        for idx in range(len(premises))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions with trained HLV model")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory",
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
        default="predictions.json",
        help="Output file path (JSON array format)",
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
        default=0,
        help="Maximum sequence length for tokenization (set to 0 to use the model limit)",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Direct JSON dataset directory containing dataset.json",
    )



    args = parser.parse_args()
    output_path = Path(args.output_file)
    if output_path.suffix != ".json":
        parser.error("--output_file must end in .json; prediction output uses a top-level JSON array.")

    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        args.device = "cuda:0"

    print(f"Using device: {args.device}")
    print(f"Loading model from {args.model_path}...")
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    trainer = HLVTrainer(TrainingConfig(model_name_or_path=str(model_path)))
    trainer.load_model(str(model_path))
    tokenizer = trainer.tokenizer
    model = trainer.model
    if tokenizer is None or model is None:
        raise RuntimeError(f"Failed to load model/tokenizer from {model_path}")

    model.to(args.device)
    print(f"Model loaded successfully ({type(model).__name__})")

    if not args.data_dir:
        raise ValueError("--data_dir is required for prediction.")
    manifest = json.loads((Path(args.data_dir) / "dataset.json").read_text(encoding="utf-8"))
    data_format = manifest.get("format")
    if data_format == "text_pair_label_distribution":
        samples = TextPairClassificationJSONReader(args.data_dir).load_split(args.split)
    elif data_format == "single_text_label_distribution":
        samples = SingleTextClassificationJSONReader(args.data_dir).load_split(args.split)
    elif data_format == "single_text_multilabel_annotation_distribution":
        samples = SingleTextMultilabelJSONReader(args.data_dir).load_split(args.split)
    elif data_format == "text_pair_multidimensional_label_distribution":
        samples = TextPairMultilevelJSONReader(args.data_dir).load_split(args.split)
    elif data_format == "single_text_multidimensional_label_distribution":
        samples = SingleTextMultilevelJSONReader(args.data_dir).load_split(args.split)
    else:
        raise ValueError(f"Unsupported prediction format: {data_format!r}")

    print(f"Loaded {len(samples)} samples")

    predictions = []
    batch_size = args.batch_size

    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i : i + batch_size]
        if hasattr(batch_samples[0], "text_a"):
            premises = [s.text_a for s in batch_samples]
            hypotheses = [s.text_b for s in batch_samples]
        elif hasattr(batch_samples[0], "text"):
            premises = [s.text for s in batch_samples]
            hypotheses = None
        else:
            raise TypeError(f"Unsupported sample type: {type(batch_samples[0]).__name__}")

        batch_outputs = predict_batch(
            model,
            tokenizer,
            premises,
            hypotheses,
            args.device,
            max_length=args.max_length,
        )

        for sample, sample_outputs in zip(batch_samples, batch_outputs):
            pred_record = PredictionRecord(
                id=sample.id,
                task=sample.task,
                split=sample.split,
                source=sample.source,
                outputs=sample_outputs,
            )
            predictions.append(pred_record)

        if (i + batch_size) % (batch_size * 10) == 0:
            print(f"Processed {i + batch_size}/{len(samples)} samples")

    print(f"Generated {len(predictions)} predictions")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_records(output_path, ({
        "id": pred.id, "task": pred.task, "split": pred.split,
        "source": pred.source, "outputs": pred.outputs,
    } for pred in predictions))

    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
