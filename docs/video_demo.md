# Video demo: real ChaosNLI data

This is a real, three-class human-label-variation dataset: ChaosNLI MNLI-M
fold 0 (1,295 train / 144 dev / 160 test rows). Its three label probabilities
make it a good visual demo: evaluation writes a ternary PNG, an interactive
ternary HTML view, disagreement-stratified metrics, and two summary plots.

The processed data is already at
`data/datasets/text_pair/chaosnli/mnli_m/0`. The training manifest is merged
into `configs/demo.json` so the native training command stays short.

Run these commands from the repository root:

```bash
uv sync
```

```bash
uv run spinda train --config configs/demo.json
```

```bash
uv run spinda predict --model_path outputs/demo/seed_42/final_model --input_file data/datasets/text_pair/chaosnli/mnli_m/0/test.json --config configs/predict.json --output_file outputs/demo/seed_42/test/predictions.json
```

```bash
uv run spinda evaluate --predictions outputs/demo/seed_42/test/predictions.json --input_file data/datasets/text_pair/chaosnli/mnli_m/0/test.json --human_labels data/datasets/text_pair/chaosnli/mnli_m/0/test.json --analysis --ternary-plot --ternary-plot-dir outputs/demo/seed_42/test/figures --ternary-plot-title "ChaosNLI demo" --output_file outputs/demo/seed_42/test/evaluation.json
```

```bash
uv run spinda analyze --run-dirs outputs/demo --labels BertTiny_JSD --title "ChaosNLI MNLI-M" --output-dir outputs/demo/analysis
```

The third command writes `predictions.json`. Evaluation writes the metrics,
per-instance error CSV, ternary PNG, and interactive ternary HTML under
`outputs/demo/seed_42/test`; analysis writes the TVD-by-disagreement and
instance-TVD violin figures under `outputs/demo/analysis`.

If the processed data is absent, prepare the real source once:

```bash
uv run python -m hlv_toolkits.scripts.prepare_chaosnli_annotation_labels --subsets mnli_m --fold 0
```
