# ai-checker

中文 AI 生成文本检测工具 —— 基于 `hfl/chinese-roberta-wwm-ext` 微调的二分类模型，用于区分人类写作（Human）与 AI 生成（AI-Generated）的文本。

## 工作流

| 阶段 | 执行环境 | 说明 |
|------|----------|------|
| 数据处理 | 本地 | 提取、清洗、划分 |
| 训练 | Kaggle | 通过 `run.ipynb` 从 GitHub 克隆代码并执行 |
| 推理 | 本地 | OpenVINO IR 转换（FP32/INT8）+ 推理，针对 Intel GPU/NPU 优化 |

## 目录结构

```
ai-checker/
├── data/
│   ├── raw/
│   │   ├── C-ReD/               # 原始数据，只读不修改
│   │   ├── MAGA-cn/             # 原始数据，只读不修改
│   │   ├── NLPCC-2025-Task1/    # 原始数据，只读不修改（train.json + dev.json）
│   │   └── unified.jsonl        # extract.py 生成的统一格式数据
│   ├── cleaned/
│   │   └── cleaned.jsonl        # clean.py 清洗后的数据
│   ├── eval/
│   │   └── test_with_label.json # NLPCC OOD 评估基准（不进训练）
│   └── processed/               # 划分后的数据（JSONL 格式）
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
├── src/
│   ├── data/                    # 数据处理脚本
│   │   ├── extract.py           # 数据提取（多数据集 loader 汇总）
│   │   ├── loaders/             # 各数据集读取模块
│   │   │   ├── cred.py          # C-ReD loader
│   │   │   ├── maga.py          # MAGA-cn loader
│   │   │   └── nlpcc.py         # NLPCC-2025-Task1 loader
│   │   ├── clean.py             # 数据清洗
│   │   └── split.py             # 数据集划分
│   ├── train/                   # 训练相关代码
│   │   ├── config.py            # 训练配置（路径解析 + TrainingArguments）
│   │   ├── trainer.py           # 训练主逻辑（HuggingFace Trainer API）
│   │   └── visualize.py         # matplotlib 训练成果可视化
│   ├── inference/               # 推理相关代码
│   │   ├── convert.py           # 模型转换：FP32 OpenVINO IR（可选 INT8 量化）
│   │   ├── predict.py           # 推理脚本
│   │   ├── compare.py           # FP32 IR vs INT8 IR 性能对比
│   │   └── evaluate.py          # 单模型详细评估（混淆矩阵/ROC/长度分层/场景细分）
│   └── run.ipynb                # Kaggle 训练入口 Notebook
├── models/                      # 模型保存（gitignored）
│   ├── base/                    # 微调后的 FP32 模型 + checkpoint + plots/
│   ├── ir/                      # FP32 OpenVINO IR 模型
│   ├── quantized/               # INT8 量化模型
│   ├── cache/                   # OpenVINO 编译缓存（自动生成）
│   └── eval/                    # 评估输出图表（混淆矩阵/ROC PNG）
├── requirements/
│   ├── requirements-local.txt   # 本地依赖（数据处理 + 推理）
│   └── requirements-kaggle.txt  # Kaggle 依赖（训练）
├── AGENTS.md                    # 开发者参考文档（权威规范）
└── README.md
```

## 数据

### 数据源与平衡策略

统一 label 约定：**1 = human，0 = AI-generated**。

| 数据集 | 内容 | 平衡策略 |
|--------|------|----------|
| [C-ReD](https://arxiv.org/abs/2604.11796) | 5 个类别（composition/film_review/news/paper/question_answer）× 9 种 LLM | 人类全取，机器按类别裁剪到与人类等量（每类 1:1） |
| [MAGA-cn](https://github.com/anyangsong/MAGA) | 仅 `MAGA-cn`（不使用 MGB），`model == "human"` 为人类文本 | AI 按 domain 裁剪到 human 的 0.7667 倍（抵消 NLPCC 的 3:1 不平衡，全局 1:1） |
| [NLPCC-2025-Task1](https://github.com/NLP2CT/NLPCC-2025-Task1) | train + dev，3 域 × 3 模型；label 约定与项目相反，loader 内翻转 | 全量保留（原始 3:1 不平衡，由 MAGA 裁剪抵消） |

`data/eval/test_with_label.json` 作为独立 **OOD 评估基准**，不进入训练。

### 处理流水线（提取 → 清洗 → 划分）

1. **提取**（`extract.py`）：各数据集 loader（`src/data/loaders/`）暴露 `load() -> list[dict]`，统一 schema `{text, label, category, source}` 后合并输出 `data/raw/unified.jsonl`。支持 `--workers` 并行（C-ReD 机器文本裁剪为串行 + 固定 seed）。
2. **清洗**（`clean.py`）：HTML/URL/emoji/控制字符/乱码去除 → 繁转简（OpenCC）→ 空白归一 → 剥包裹引号 → 去短文本（<10 字符），再按 text 去重。支持 `--workers` 并行。
3. **划分**（`split.py`）：按 `(category, label)` 分层划分 8:1:1，固定随机种子 42，不做全局下采样；输出仅含 `text` + `label`。

## 训练

- **模型**：`AutoModelForSequenceClassification`，预训练权重 `hfl/chinese-roberta-wwm-ext`（所有 `from_pretrained` 均 `local_files_only=True`，本地路径 `D:\Dylan\Model\chinese-roberta-wwm-ext`）
- **框架**：PyTorch + Transformers + Datasets + HuggingFace Trainer API；多 GPU 用 torchrun 启动 DDP（2×T4）

### 超参配置

```
max_seq_length: 512               # 覆盖 ~63% 数据无需截断
batch_size: 8                     # per GPU，2×T4 共 16
gradient_accumulation_steps: 2    # 有效 batch_size = 8 × 2 GPU × 2 = 32
learning_rate: 2e-5
weight_decay: 0.01
num_epochs: 3
optimizer: AdamW
scheduler: linear warmup + linear decay
warmup_ratio: 0.1
fp16: true                        # T4 混合精度加速
```

### run.ipynb（Kaggle 入口）

Notebook 从 GitHub 克隆源码 → 安装 `requirements-kaggle.txt` → （可选）预加载模型为 safetensors 缓存 → 用 `torch.distributed.run` 启动 `trainer.py`。Trainer 自动处理分布式初始化、混合精度、梯度累积、日志、评估与 checkpoint 保存。

训练结束后自动生成 PNG 图表到 `models/base/plots/`：`loss_curves.png`、`metrics_curves.png`、`confusion_matrix.png`、`roc_curve.png`、`dashboard.png`。

## 推理

### 模型转换与量化（convert.py）

- **FP32 OpenVINO IR（默认，不量化）**：`optimum-intel` 直接导出，无需校准集
- **INT8 量化（可选 `--quantize`）**：`optimum-intel` + `openvino` + `nncf` 做 PTQ，需校准集（从验证集抽 200–500 条，覆盖各类别与两种标签）
- 目标硬件：Intel GPU (OpenCL) / NPU

### 推理（predict.py）

- 三种模式：FP32 PyTorch / FP32 OpenVINO IR / INT8 OpenVINO IR（按模型目录自动识别）
- 输入：单条文本、jsonl 批量文件或交互式；输出 label（0/1）与置信度
- 支持 CPU / GPU / NPU 设备；超长文本（>512 tokens）自动滑动窗口（50% 重叠 token 级切分，log-softmax 平均聚合）

### 性能对比（compare.py）

独立对比 FP32 IR vs INT8 IR：推理耗时、吞吐量（texts/s）、加速比、标签一致性；两模型使用相同设备、batch size 与 max_length 保证公平。

### 详细评估（evaluate.py）

复用 `predict.py` 的推理类，输出 accuracy / macro-F1 / 每类 P·R·F1、混淆矩阵（含行百分比）、ROC-AUC、长度分层（<64/64-128/128-256/256-512/≥512）、场景细分（Normal/Attack/Varying，仅 NLPCC）。评估数据集：`data/processed/test.jsonl`（in-domain）+ `data/eval/test_with_label.json`（NLPCC OOD）。

### 编译缓存（仅 OpenVINO 模式）

首次推理时设备编译较慢（NPU/GPU 尤其）；指定 `--cache-dir` 后编译结果缓存到磁盘（默认 `models/cache/`），后续秒级启动。不同 device/batch-size/max-length 组合生成不同缓存。

## 快速开始

所有脚本从项目根目录（`ai-checker/`）执行。

```bash
pip install -r requirements/requirements-local.txt
```

### 数据处理

```bash
# 1. 数据提取：读取 C-ReD、MAGA-cn 与 NLPCC-2025-Task1，统一 schema 后合并
python src/data/extract.py
python src/data/extract.py --workers 8   # 并行提取

# 2. 数据清洗
python src/data/clean.py
python src/data/clean.py --workers 8     # 并行清洗

# 3. 数据集划分（分层 8:1:1，无全局下采样）
python src/data/split.py
```

### 训练

```bash
# 本地调试（单机 CPU/GPU）
python src/train/trainer.py

# Kaggle 提交：将 src/run.ipynb 上传至 Kaggle，连接 GPU 加速器后运行全部单元格
```

### 推理

```bash
# 1. FP32 OpenVINO IR 转换（不量化）
python src/inference/convert.py \
    --model-path models/base \
    --output-dir models/ir

# 2. INT8 量化（可选）
python src/inference/convert.py \
    --model-path models/base \
    --output-dir models/quantized \
    --quantize \
    --calibration-file data/processed/val.jsonl \
    --calibration-samples 300

# 3. 单条文本预测（FP32 PyTorch / FP32 IR / INT8 IR 自动识别）
python src/inference/predict.py --model-path models/base --text "待检测文本"
python src/inference/predict.py --model-path models/ir --text "待检测文本"
python src/inference/predict.py --model-path models/quantized --text "待检测文本"

# 4. 批量文件预测（带编译缓存加速首次启动）
python src/inference/predict.py --model-path models/ir \
    --input-file data/test.jsonl \
    --cache-dir models/cache --output-file results.jsonl

# 5. 性能对比：FP32 IR vs INT8 IR
python src/inference/compare.py \
    --fp32-ir-model-path models/ir \
    --int8-model-path models/quantized \
    --input-file data/test.jsonl \
    --cache-dir models/cache

# 6. 单模型详细评估（in-domain + NLPCC OOD，可选 --output-dir 输出图表）
python src/inference/evaluate.py --model-path models/base --device CPU
```

## 依赖

- **本地**（数据处理 + 推理）：`requirements/requirements-local.txt` — pandas, numpy, opencc-python-reimplemented, torch, transformers, optimum-intel, openvino, nncf, matplotlib
- **Kaggle**（训练）：`requirements/requirements-kaggle.txt` — torch, transformers, datasets, pandas, numpy, scikit-learn, tqdm, matplotlib

## 数据来源

- [C-ReD: A Comprehensive Chinese Benchmark for LLM-Generated Text Detection](https://arxiv.org/abs/2604.11796)（ACL 2026 Findings）
- [MAGA: Machine-Augment-Generated Text Detection Benchmark](https://github.com/anyangsong/MAGA)
- [NLPCC-2025 Shared-Task 1: LLM-Generated Text Detection](https://github.com/NLP2CT/NLPCC-2025-Task1)

## 许可证

MIT License. 详见 [LICENSE](LICENSE)。

更多开发细节（数据规范、代码规范、依赖管理等）见 [AGENTS.md](AGENTS.md)。
