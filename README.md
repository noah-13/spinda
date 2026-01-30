# NLI Toolkits

A toolkit for fine-tuning BERT/RoBERTa models on Natural Language Inference (NLI) tasks and evaluating calibration metrics, implementing methods from ["Stop Measuring Calibration When Humans Disagree"](https://aclanthology.org/2022.emnlp-main.124.pdf) (Baan et al., EMNLP 2022).

## Features

- **Data Loading**: Support for SNLI and ChaosNLI datasets
- **Model Training**: Fine-tune BERT/RoBERTa on SNLI
- **Evaluation Metrics**: 
  - ECE (Expected Calibration Error)
  - EntCE (Human Entropy Calibration Error)
  - RankCS (Human Ranking Calibration Score)
  - DistCE (Human Distribution Calibration Error)

## Installation

```bash
# Install dependencies
pip install -e .
```

Optional (enable plotting in `evaluate.py`):

```bash
pip install -e ".[plot]"
```

## Quick Start

### 1. Train a Model

Train RoBERTa on SNLI:

```bash
python main.py train \
    --model roberta-base \
    --output_dir ./outputs/roberta_snli \
    --num_epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-5
```

Or use BERT:

```bash
python main.py train \
    --model bert-base-uncased \
    --output_dir ./outputs/bert_snli \
    --num_epochs 3
```

### 2. Generate Predictions

Generate predictions on test set:

```bash
python main.py predict \
    --model_path ./outputs/roberta_snli/final_model \
    --split test \
    --output_file predictions.jsonl \
    --batch_size 32
```

### 3. Evaluate Predictions

Evaluate on SNLI test set (single-label evaluation):

```bash
python main.py evaluate \
    --predictions predictions.jsonl \
    --ground_truth_source snli \
    --ground_truth_split test \
    --output_file results.json
```

Evaluate on ChaosNLI (distribution-based evaluation):

```bash
# First, download ChaosNLI from:
# https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip

python main.py evaluate \
    --predictions predictions.jsonl \
    --ground_truth_source chaosnli \
    --chaosnli_path chaosNLI_v1.0/chaosNLI_snli.jsonl \
    --output_file chaosnli_results.json

# By default, evaluation also saves a DistCE distribution plot to outputs/figures/.
# Disable plotting with:
python main.py evaluate \
    --predictions predictions.jsonl \
    --ground_truth_source chaosnli \
    --chaosnli_path ./data/chaosNLI_v1.0/chaosNLI_snli.jsonl \
    --output_file chaosnli_results.json \
    --no-plot
```

## Project Structure

```
nli_toolkits/
├── data/              # Data loading modules
│   ├── schemas.py     # Data schemas (NLISample, etc.)
│   └── readers/       # Dataset readers (SNLI, ChaosNLI)
├── models/            # Model training
│   └── trainer.py     # NLITrainer class
├── eval/              # Evaluation metrics
│   ├── metrics.py     # Metric implementations
│   └── evaluator.py   # Evaluator class
└── scripts/           # CLI scripts
    ├── train.py       # Training script
    ├── predict.py     # Prediction script
    └── evaluate.py    # Evaluation script
```

## Usage Examples

### Programmatic Usage

```python
from nli_toolkits.data import SNLIReader
from nli_toolkits.models import NLITrainer, TrainingConfig
from nli_toolkits.eval import Evaluator

# Load data
reader = SNLIReader()
train_samples = reader.load_train()
eval_samples = reader.load_dev()

# Train model
config = TrainingConfig(
    model_name_or_path="roberta-base",
    num_epochs=3,
    batch_size=16,
)
trainer = NLITrainer(config)
trainer.train(train_samples, eval_samples)

# Evaluate
evaluator = Evaluator()
results = evaluator.evaluate(predictions, ground_truth)
print(results)
```

## Evaluation Metrics

### ECE (Expected Calibration Error)
Measures calibration against majority vote labels. Lower is better.

### EntCE (Human Entropy Calibration Error)
Measures alignment between model uncertainty and human disagreement. 
- Positive values: model is over-confident
- Negative values: model is under-confident

### RankCS (Human Ranking Calibration Score)
Measures whether model's class ranking matches human ranking. Higher is better (range: 0-1).

### DistCE (Human Distribution Calibration Error)
Measures total variation distance between model predictions and human distributions. Lower is better.

## Requirements

- Python >= 3.12
- PyTorch >= 2.10.0
- Transformers >= 4.57.6
- Datasets >= 4.5.0
- NumPy

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{baan2022stop,
  title={Stop Measuring Calibration When Humans Disagree},
  author={Baan, Joris and Aziz, Wilker and Plank, Barbara and Fern{\'a}ndez, Raquel},
  booktitle={Proceedings of EMNLP},
  year={2022}
}
```

## License

[Add your license here]
