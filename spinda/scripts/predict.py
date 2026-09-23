#!/usr/bin/env python3
"""Generate predictions for an arbitrary JSON input file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from spinda.data.json_io import load_records
from spinda.models.trainer import (
    HLVTrainer,
    MultiLevelClassificationModel,
    TrainingConfig,
    resolve_max_length,
)


PREDICTION_CONFIG_KEYS = {
    "model_path",
    "input_file",
    "output_file",
    "batch_size",
    "device",
    "max_length",
}


def _load_json_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if config_path.suffix != ".json":
        raise ValueError(f"Prediction config must be a JSON file: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Prediction config not found: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in prediction config {config_path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError(f"Prediction config {config_path} must be a JSON object.")
    unknown = sorted(set(config) - PREDICTION_CONFIG_KEYS)
    if unknown:
        unknown_names = ", ".join(unknown)
        raise ValueError(f"Unknown prediction config keys in {config_path}: {unknown_names}")
    return config


def _load_merged_json_configs(paths: list[str]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        merged.update(_load_json_config(path))
    return merged


def _load_inputs(input_path: Path) -> tuple[str, list[dict[str, str]]]:
    """Load unlabeled inference rows and infer their input shape from text fields."""
    if input_path.suffix != ".json":
        raise ValueError("--input_file must be a .json file containing a top-level array.")
    samples: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    input_shape: str | None = None
    for index, record in enumerate(load_records(input_path, kind="prediction input records"), 1):
        if not isinstance(record, dict):
            raise ValueError(f"Input record {index} in {input_path} must be a JSON object.")
        has_pair = "text_a" in record or "text_b" in record
        has_text = "text" in record
        if has_pair and has_text:
            raise ValueError(f"Input record {index} in {input_path} must use either text_a/text_b or text, not both.")
        if has_pair:
            input_shape_for_record = "text_pair"
            required_fields = ("id", "text_a", "text_b")
        elif has_text:
            input_shape_for_record = "single_text"
            required_fields = ("id", "text")
        else:
            raise ValueError(f"Input record {index} in {input_path} must contain text_a/text_b or text.")
        if input_shape is None:
            input_shape = input_shape_for_record
        elif input_shape != input_shape_for_record:
            raise ValueError(f"Input record {index} in {input_path} mixes text-pair and single-text inputs.")
        if not all(field in record for field in required_fields):
            raise ValueError(f"Input record {index} in {input_path} must contain {list(required_fields)}.")
        sample_id = record["id"]
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen_ids:
            raise ValueError(f"Input record {index} in {input_path} has a missing or duplicate id.")
        if any(not isinstance(record[field], str) for field in required_fields if field != "id"):
            raise ValueError(f"Input record {index} in {input_path} text fields must be strings.")
        seen_ids.add(sample_id)
        samples.append({field: record[field] for field in required_fields})
    if not samples:
        raise ValueError(f"Prediction input is empty: {input_path}")
    return input_shape, samples


def _checkpoint_label_names(model: Any) -> list[str] | None:
    """Return categorical names saved in a checkpoint by training, if present."""
    model_config = getattr(model, "config", None)
    mapping = getattr(model_config, "id2label", None)
    num_labels = getattr(model_config, "num_labels", None)
    if not isinstance(mapping, dict) or not isinstance(num_labels, int):
        return None
    names = [mapping.get(index, mapping.get(str(index))) for index in range(num_labels)]
    return names if all(isinstance(name, str) and name for name in names) else None


def _build_multilevel_outputs(level_probs: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
    batch_size = next(iter(level_probs.values())).shape[0] if level_probs else 0
    results: List[Dict[str, Any]] = []
    for row_idx in range(batch_size):
        row_levels: Dict[str, Dict[str, Any]] = {}
        for level in level_probs:
            probs = level_probs[level][row_idx]
            row_levels[level] = {
                "probs": probs.cpu().tolist(),
                "pred": int(torch.argmax(probs).item()),
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
    inputs = tokenizer(
        premises,
        hypotheses,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=effective_max_length,
    ) if hypotheses is not None else tokenizer(
        premises,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=effective_max_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        if isinstance(model, MultiLevelClassificationModel):
            return _build_multilevel_outputs({
                level: torch.softmax(logit, dim=-1)
                for level, logit in zip(model.level_num_labels, logits)
            })
        multilabel = getattr(model_config, "problem_type", None) == "multi_label_classification"
        probs = torch.sigmoid(logits) if multilabel else torch.softmax(logits, dim=-1)
        preds = (probs >= 0.5).long() if multilabel else torch.argmax(probs, dim=-1)

    probs_np = probs.cpu().numpy()
    preds_np = preds.cpu().numpy()
    return [
        {"probs": probs_np[index].tolist(), "pred": preds_np[index].tolist() if preds_np.ndim == 2 else int(preds_np[index])}
        for index in range(len(premises))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions from a checkpoint for an arbitrary JSON input file.")
    parser.add_argument(
        "--config",
        nargs="+",
        action="extend",
        default=[],
        metavar="PATH",
        help="One or more JSON configs merged left to right; explicit CLI values override them.",
    )
    parser.add_argument("--model_path", default=None, help="Path to trained model directory.")
    parser.add_argument("--input_file", default=None, help="JSON array of unlabeled examples to predict.")
    parser.add_argument("--output_file", default="predictions.json", help="Output JSON array path.")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device: auto, cpu, cuda, or cuda:<index>.")
    parser.add_argument("--max_length", type=int, default=0, help="Tokenized maximum length; 0 uses the model limit.")

    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", nargs="+", action="extend", default=[])
    bootstrap_args, _ = bootstrap_parser.parse_known_args()
    config_values = _load_merged_json_configs(bootstrap_args.config)
    args = parser.parse_args(namespace=argparse.Namespace(**config_values))
    if not args.model_path:
        parser.error("--model_path is required (in --config or on the command line).")
    if not args.input_file:
        parser.error("--input_file is required (in --config or on the command line).")

    output_path = Path(args.output_file)
    if output_path.suffix != ".json":
        parser.error("--output_file must end in .json; prediction output uses a top-level JSON array.")
    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        args.device = "cuda:0"

    _input_shape, samples = _load_inputs(Path(args.input_file))
    print(f"Using device: {args.device}")
    print(f"Loading model from {args.model_path}...")
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    trainer = HLVTrainer(TrainingConfig(model_name_or_path=str(model_path)))
    trainer.load_model(str(model_path))
    tokenizer, model = trainer.tokenizer, trainer.model
    if tokenizer is None or model is None:
        raise RuntimeError(f"Failed to load model/tokenizer from {model_path}")
    checkpoint_labels = _checkpoint_label_names(model)
    model.to(args.device)
    print(f"Model loaded successfully ({type(model).__name__})")
    print(f"Loaded {len(samples)} input samples")

    predictions: list[dict[str, Any]] = []
    input_path = Path(args.input_file)
    for start in range(0, len(samples), args.batch_size):
        batch_samples = samples[start : start + args.batch_size]
        if "text_a" in batch_samples[0]:
            premises = [sample["text_a"] for sample in batch_samples]
            hypotheses: list[str] | None = [sample["text_b"] for sample in batch_samples]
        else:
            premises = [sample["text"] for sample in batch_samples]
            hypotheses = None
        batch_outputs = predict_batch(model, tokenizer, premises, hypotheses, args.device, max_length=args.max_length)
        for sample, outputs in zip(batch_samples, batch_outputs):
            predictions.append({
                "id": sample["id"],
                "source": str(input_path),
                "outputs": outputs,
            })
        if (start + args.batch_size) % (args.batch_size * 10) == 0:
            print(f"Processed {start + args.batch_size}/{len(samples)} samples")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"predictions": predictions}
    if checkpoint_labels is not None:
        payload["labels"] = checkpoint_labels
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(predictions)} predictions")
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
