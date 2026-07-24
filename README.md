# HLV Toolkits

HLV Toolkits 用于自然语言推理（NLI）和带有人类意见分布的文本分类实验，覆盖数据准备、预处理、训练、预测、评估和可视化。

## 支持范围

### 数据集

- **SNLI**：通过 Hugging Face `datasets` 加载，使用单一 hard label。
- **ChaosNLI**：使用 100 人标注分布，支持 soft-label 训练和分布评估。
- **DiscoGeM 2.0**：支持英文、德文、法文和捷克文；预处理后同时保留多级 hard labels 和 human distributions。
- **Processed JSONL**：读取仓库统一生成的规范化 JSONL。

内置数据集使用固定目录，不需要在训练、预测或评估命令中提供路径：

```text
data/external/chaosnli/
data/external/DiscoGeM/DiscoGeM 2.0/
data/processed/
```

自定义数据集目前不是默认 CLI 流程的一部分，后续可以单独增加自定义数据入口。

### Encoder 模型

模型通过 Hugging Face 的 `AutoTokenizer`、`AutoModel` 和 `AutoModelForSequenceClassification` 加载。因此，大多数标准 Transformer encoder 都可以使用，例如：

- BERT 和 RoBERTa
- DeBERTa
- XLM-R
- ModernBERT
- InfoXLM
- 其他提供标准 Hugging Face encoder 接口的模型

模型通过 `--model <model-name-or-local-path>` 指定；模型和 tokenizer 不在本地时，Transformers 会自动从 Hugging Face Hub 下载。

当前目标是标准的 encoder-only 文本模型，并不是任意 Transformer：

- decoder-only 模型（如 GPT、Llama）不保证支持；
- encoder-decoder 模型（如 T5、BART）不属于当前训练接口；
- 需要额外实体、图像或特殊输入字段的模型可能无法直接使用；
- 模型需要提供标准的 hidden states，并能接受 premise/hypothesis 文本对。

### 训练方式

训练入口为 `hlv_toolkits.scripts.train`，支持以下 head 和目标：

- `classification`：普通单标签分类，适合 SNLI hard labels；
- `joint_classification`：同时使用 hard label 和 soft distribution；
- `multilevel_classification`：DiscoGeM 多级分类 head；
- `multilevel_regression`：DiscoGeM 多级回归 head。

Soft-label 训练支持：

- `cross_entropy`
- `kl_div`

模型选择指标支持：

- `accuracy`
- `tvd`
- `kl_divergence`

DiscoGeM 还支持：

- `--discogem_label_mode soft|hard`
- `--discogem_label_level level1|level2|level3|all`
- `--discogem_language en|de|fr|cs`

### 设备选择

训练和预测支持：

```text
--device auto
--device cpu
--device cuda
--device cuda:0
--device cuda:1
```

使用 `cuda:1` 可以固定当前进程可见的第 2 张 GPU，不需要设置 `CUDA_VISIBLE_DEVICES`。

## 安装

要求 Python `>=3.12`。

```bash
uv sync
```

可选依赖：

```bash
uv sync --extra dev
uv sync --extra plot
```

## 整体流程

仓库推荐按以下顺序运行：

```text
下载原始数据
    ↓
预处理为规范化 JSONL
    ↓
训练模型
    ↓
生成预测
    ↓
评估和可视化
```

## 1. 下载和预处理

### 一键预处理

`scripts/preprocess.sh` 会在缺少原始文件时自动下载 ChaosNLI 或 DiscoGeM，然后生成规范化 JSONL。已存在的文件会跳过下载。

```bash
bash scripts/preprocess.sh snli
bash scripts/preprocess.sh chaosnli
bash scripts/preprocess.sh discogem
```

ChaosNLI 首次下载后会自动从官方 SNLI 文件生成固定的 1,400 条 train 和 114 条 dev split。

也可以只下载原始数据：

```bash
uv run python -m hlv_toolkits.scripts.download_data chaosnli
uv run python -m hlv_toolkits.scripts.download_data discogem
```

下载源：

- ChaosNLI 官方 ZIP：Dropbox
- DiscoGeM 2.0 官方 archive：DiscoGeM GitHub 仓库

下载模块支持通过环境变量覆盖下载地址或存储目录，但常规命令不需要提供 path：

```bash
CHAOSNLI_URL=<url> DISCOGEM_URL=<url> bash scripts/preprocess.sh chaosnli
CHAOSNLI_DIR=<dir> bash scripts/preprocess.sh chaosnli
```

生成结果：

```text
data/processed/snli/{train,dev,test}.jsonl
data/processed/chaosnli/{train,dev}.jsonl
data/processed/discogem.jsonl
data/external/DiscoGeM/DiscoGeM 2.0/DiscoGeM2.0_annotation.tgz
data/external/chaosnli/chaosNLI_snli_train.jsonl
data/external/chaosnli/chaosNLI_snli_dev.jsonl
```

## 2. 训练

### SNLI hard-label

```bash
uv run python -m hlv_toolkits.scripts.train \
  --data_source snli \
  --model roberta-base \
  --device cuda:0 \
  --output_dir outputs/snli/hard \
  --num_epochs 3 \
  --train_batch_size 32 \
  --eval_batch_size 64 \
  --seeds 42
```

### ChaosNLI soft-label

```bash
uv run python -m hlv_toolkits.scripts.train \
  --data_source chaosnli \
  --model roberta-base \
  --device cuda:1 \
  --use_soft_labels \
  --soft_label_loss cross_entropy \
  --soft_label_metric_for_best_model tvd \
  --output_dir outputs/chaosnli/soft \
  --num_epochs 20 \
  --seeds 42
```

### DiscoGeM 2.0 soft-label

```bash
uv run python -m hlv_toolkits.scripts.train \
  --data_source discogem \
  --model roberta-base \
  --device cuda:1 \
  --discogem_label_mode soft \
  --discogem_label_level level2 \
  --discogem_language en \
  --output_dir outputs/discogem/soft \
  --num_epochs 20 \
  --seeds 42
```

### DiscoGeM 2.0 hard-label

```bash
uv run python -m hlv_toolkits.scripts.train \
  --data_source discogem \
  --model microsoft/deberta-v3-base \
  --device cuda:0 \
  --discogem_label_mode hard \
  --discogem_label_level level2 \
  --output_dir outputs/discogem/hard
```

### 从 processed 数据训练

如果已经运行过预处理，可以显式使用规范化数据：

```bash
uv run python -m hlv_toolkits.scripts.train \
  --data_source processed \
  --processed_task discogem \
  --device cuda:0 \
  --discogem_label_mode soft \
  --discogem_label_level level2 \
  --output_dir outputs/discogem/processed
```

`processed` 模式会根据 `--processed_task` 自动读取 `data/processed/` 下的对应文件。

### Shell wrappers

所有训练 wrapper 都支持 `DEVICE`：

```bash
DEVICE=cuda:0 bash scripts/snli/train_hard.sh
DEVICE=cuda:1 bash scripts/snli/train_soft.sh
DEVICE=cuda:1 bash scripts/discogem/train_discogem_soft.sh
DEVICE=cpu bash scripts/discogem/train_discogem_hard.sh
DEVICE=cuda:0 bash scripts/discogem/exp_discogem_screen.sh
```

对应入口：

- `scripts/snli/train_hard.sh`
- `scripts/snli/train_soft.sh`
- `scripts/discogem/train_discogem_hard.sh`
- `scripts/discogem/train_discogem_soft.sh`
- `scripts/discogem/exp_discogem_screen.sh`

## 3. 预测

训练完成后，使用最终模型生成统一 JSONL 预测：

```bash
uv run python -m hlv_toolkits.scripts.predict \
  --model_path outputs/snli/hard/seed_42/final_model \
  --data_source snli \
  --split test \
  --device cuda:0 \
  --batch_size 32 \
  --output_file outputs/snli/predictions/test.jsonl
```

ChaosNLI 和 DiscoGeM 只需修改 `--data_source`，内置数据路径会自动使用固定目录。

## 4. 评估和可视化

单标签 SNLI：

```bash
uv run python -m hlv_toolkits.scripts.evaluate \
  --predictions outputs/snli/predictions/test.jsonl \
  --ground_truth_source snli \
  --ground_truth_split test \
  --output_file outputs/snli/results/test.json
```

ChaosNLI 分布评估和可视化：

```bash
uv run python -m hlv_toolkits.scripts.evaluate \
  --predictions outputs/chaosnli/predictions/test.jsonl \
  --ground_truth_source chaosnli \
  --ground_truth_split test \
  --output_file outputs/chaosnli/results/test.json \
  --plots tvd ternary \
  --plot_dir outputs/chaosnli/figures \
  --ternary_source both
```

常用选项：

- `--no-plot`：关闭所有图；
- `--no-ternary_browser`：关闭交互式 ternary HTML；
- `--predictions_format machamp`：读取 Machamp/ChaosNLI 风格预测；
- `--ternary_source model|human|both`：选择 ternary 图数据来源。

评估指标包括：

- 单标签：`accuracy`；
- 分布标签：`accuracy`、`tvd_mean`、`jsd_mean`、`kl_mean`、`soft_micro_f1`、`soft_macro_f1`、`distance_correlation`。

## 5. 数据检查和统一入口

检查样例：

```bash
uv run python -m hlv_toolkits.scripts.inspect_data \
  --source snli \
  --split test \
  --num_examples 5
```

统一入口：

```bash
uv run python main.py train ...
uv run python main.py predict ...
uv run python main.py evaluate ...
uv run python main.py preprocess ...
```

## 预测文件格式

每行一个 JSON 对象，例如：

```json
{"id":"341#1","task":"nli","split":"test","source":"snli","outputs":{"probs":[0.7,0.2,0.1],"pred":0}}
```

标签映射：

```text
0 = entailment
1 = neutral
2 = contradiction
```

## 项目结构

```text
hlv_toolkits/
├── data/
│   ├── schemas.py
│   └── readers/
├── models/
│   └── trainer.py
├── eval/
├── visualization/
└── scripts/
    ├── download_data.py
    ├── train.py
    ├── predict.py
    ├── evaluate.py
    ├── preprocess.py
    ├── inspect_data.py
    └── split_chaosnli.py

scripts/
├── preprocess.sh
├── discogem/
│   ├── exp_discogem_screen.sh
│   ├── train_discogem_hard.sh
│   └── train_discogem_soft.sh
└── snli/
    ├── train_hard.sh
    └── train_soft.sh
```

更多说明：

- `data/README.md`：数据目录约定；
- `data/discogem.md`：DiscoGeM 2.0 label 和 processed 格式；
- `docs/hlv_metrics_tutorial.md`：指标教程。
