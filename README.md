# ai-checker

中文 AI 生成文本检测工具 —— 基于 `hfl/chinese-roberta-wwm-ext` 微调的二分类模型，用于区分人类写作与 AI 生成的文本。

## 工作流

| 阶段 | 环境 | 说明 |
|------|------|------|
| 数据处理 | 本地 | 清洗、划分、增强 C-ReD 数据集 |
| 训练 | Kaggle (2×T4) | HuggingFace Trainer + DDP 分布式训练 |
| 推理 | 本地 | OpenVINO IR 转换 (FP32/INT8) + 推理，支持 Intel GPU/NPU |

## 目录结构

```
ai-checker/
├── data/
│   ├── raw/C-ReD/              # 原始数据（只读，不入库）
│   ├── processed/              # 清洗划分后的 JSONL
│   └── augmented/              # 增强后的训练数据
├── src/
│   ├── data/
│   │   ├── clean.py            # 数据清洗
│   │   ├── split.py            # 数据集划分
│   │   └── augment.py          # nlpcda 数据增强
│   ├── train/
│   │   ├── config.py           # 训练配置
│   │   ├── trainer.py          # 训练主逻辑
│   │   └── visualize.py        # 训练可视化
│   ├── inference/
│   │   ├── convert.py          # 模型转换：FP32 OpenVINO IR（可选 INT8 量化）
│   │   ├── predict.py          # 推理脚本
│   │   └── compare.py          # FP32 IR vs INT8 IR 性能对比
│   └── run.ipynb               # Kaggle 训练入口
├── models/
│   ├── base/                   # 微调后的 FP32 模型
│   ├── ir/                     # FP32 OpenVINO IR 模型
│   ├── quantized/              # INT8 量化模型 (OpenVINO IR)
│   └── cache/                  # OpenVINO 编译缓存
├── requirements/
│   ├── requirements-local.txt
│   └── requirements-kaggle.txt
└── AGENTS.md                   # 开发者参考文档
```

## 快速开始

```bash
git clone https://github.com/dylanzihao/ai-checker.git
cd ai-checker

pip install -r requirements/requirements-local.txt

# 1. 数据处理
python src/data/clean.py
python src/data/split.py
python src/data/augment.py

# 2. 训练（本地单机调试）
python src/train/trainer.py

# 3. 转换 OpenVINO IR + 推理
python src/inference/convert.py --model-path models/base --output-dir models/ir
python src/inference/predict.py --model-path models/ir --text "待检测文本"
```

## 详细用法

### 数据处理

数据来源于 [C-ReD](https://arxiv.org/abs/2604.11796) 数据集（ACL 2026 Findings），包含 5 个领域 9 种 LLM 生成的 ~13 万条文本。

```bash
python src/data/clean.py      # 读取 raw/C-ReD 所有 CSV，清洗后按类别输出 JSONL
python src/data/split.py      # 8:1:1 分层划分 → train/val/test.jsonl
python src/data/augment.py    # nlpcda 增强训练集（同义词替换、随机插入/删除/交换、TF-IDF 替换）
```

增强参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--workers` | CPU 核心数-1 (最大 12) | 并行处理线程数 |
| `--multiplier` | 3 | 扩增倍率（1 原始 + 2 增强） |
| `--full` | 关闭 | 启用 SimbertBased 相似词替换（较慢） |
| `--max-samples` | 30000 | 增强后训练集上限 |

### 训练

**本地调试：**

```bash
python src/train/trainer.py
```

**Kaggle 提交：** 将 `src/run.ipynb` 上传至 Kaggle，连接 GPU 加速器后运行全部单元格。Notebook 自动从 GitHub 克隆源码、安装依赖、加载数据并启动分布式训练。

| 超参 | 值 |
|------|-----|
| 预训练模型 | `hfl/chinese-roberta-wwm-ext` |
| max_seq_length | 512 |
| batch_size (per GPU) | 8 |
| gradient_accumulation_steps | 2 |
| 有效 batch_size | 32 (8×2GPU×2) |
| learning_rate | 2e-5 |
| epochs | 3 |
| optimizer | AdamW |
| scheduler | linear warmup + linear decay |
| warmup_ratio | 0.1 |
| fp16 | true |

训练结束后自动生成可视化图表并保存至 `models/base/plots/`：loss 曲线、指标曲线、混淆矩阵、ROC 曲线、综合仪表盘。

### 推理

**FP32 OpenVINO IR 转换（不量化）：** 将 FP32 模型直接转换为 OpenVINO IR 格式，无需校准数据。

```bash
python src/inference/convert.py \
    --model-path models/base \
    --output-dir models/ir
```

**INT8 量化（可选）：** 加上 `--quantize`，使用 NNCF 进行 INT8 PTQ。

```bash
python src/inference/convert.py \
    --model-path models/base \
    --output-dir models/quantized \
    --quantize \
    --calibration-file data/processed/val.jsonl \
    --calibration-samples 300
```

**单条文本预测：**

```bash
# FP32 模式（PyTorch）
python src/inference/predict.py --model-path models/base --text "待检测文本"

# FP32 OpenVINO IR 模式
python src/inference/predict.py --model-path models/ir --text "待检测文本"

# INT8 OpenVINO IR 模式（需先完成量化）
python src/inference/predict.py --model-path models/quantized --text "待检测文本"
```

**批量文件预测：**

```bash
python src/inference/predict.py \
    --model-path models/ir \
    --input-file data/test.jsonl \
    --cache-dir models/cache \
    --output-file results.jsonl
```

**FP32 IR vs INT8 IR 性能对比：**

```bash
python src/inference/compare.py \
    --fp32-ir-model-path models/ir \
    --int8-model-path models/quantized \
    --input-file data/test.jsonl \
    --cache-dir models/cache
```

推理参数：

| 参数 | 说明 |
|------|------|
| `--model-path` | 模型路径（FP32 PyTorch 目录或 OpenVINO IR 目录） |
| `--text` | 单条待检测文本 |
| `--input-file` | 批量输入 JSONL 文件 |
| `--interactive` | 交互式推理模式 |
| `--output-file` | 预测结果输出路径 |
| `--device` | 推理设备：CPU / GPU / NPU（仅 OpenVINO 模式） |
| `--cache-dir` | OpenVINO 编译缓存目录 |
| `--batch-size` | 批量推理大小 |
| `--max-length` | 最大 token 长度（默认 512，超长文本自动滑动窗口） |
| `--eval-labels` | 标注文件路径（与 --input-file 同源，用于计算准确率） |

性能对比参数（`compare.py`）：

| 参数 | 说明 |
|------|------|
| `--fp32-ir-model-path` | FP32 OpenVINO IR 模型路径（默认 models/ir） |
| `--int8-model-path` | INT8 OpenVINO IR 模型路径（默认 models/quantized） |
| `--device` | 推理设备：CPU / GPU / NPU |
| `--eval-labels` | 标注文件路径（默认使用 --input-file 的 label 字段） |

## 依赖

- **本地：** `pip install -r requirements/requirements-local.txt`
  - pandas, numpy, nlpcda, torch, transformers, optimum-intel, openvino, nncf, matplotlib
- **Kaggle：** `pip install -r requirements/requirements-kaggle.txt`
  - torch, transformers, datasets, pandas, numpy, scikit-learn, tqdm, matplotlib

## 数据来源

[C-ReD: A Comprehensive Chinese Benchmark for LLM-Generated Text Detection](https://arxiv.org/abs/2604.11796) (ACL 2026 Findings)

## 许可证

MIT License. 详见 [LICENSE](LICENSE).
