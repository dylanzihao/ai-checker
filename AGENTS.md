# ai-checker Agents Guide

## 项目概览

训练 BERT（chinese-roberta-wwm-ext）用于中文 AI 文本检测（二分类：Human vs AI-Generated）。

### 三个阶段

| 阶段 | 执行环境 | 说明 |
|------|----------|------|
| 数据处理 | 本地 | 清洗、划分、增强 |
| 训练 | Kaggle | 通过 run.ipynb 从 GitHub 克隆代码并执行 |
| 推理 | 本地 | OpenVINO IR 转换（FP32/INT8）+ 推理，针对 Intel GPU/NPU 优化 |

## 目录结构规范

```
ai-checker/
├── data/
│   ├── raw/
│   │   └── C-ReD/               # 原始数据，只读不修改
│   ├── processed/               # 清洗划分后的数据（JSONL 格式）
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── augmented/               # 数据增强后的训练数据（JSONL 格式）
│       └── train.jsonl
├── src/
│   ├── data/                    # 数据处理脚本
│   │   ├── clean.py             # 数据清洗
│   │   ├── split.py             # 数据集划分
│   │   └── augment.py           # nlpcda 数据增强
│   ├── train/                   # 训练相关代码
│   │   ├── config.py            # 训练配置（路径解析 + TrainingArguments）
│   │   ├── trainer.py           # 训练主逻辑（HuggingFace Trainer API）
│   │   └── visualize.py         # matplotlib 训练成果可视化
│   ├── inference/               # 推理相关代码
│   │   ├── convert.py           # 模型转换：FP32 OpenVINO IR（可选 INT8 量化）
│   │   ├── predict.py           # 推理脚本
│   │   └── compare.py           # FP32 IR vs INT8 IR 性能对比
│   └── run.ipynb                # Kaggle 训练入口 Notebook
├── models/                      # 模型保存（gitignored）
│   ├── base/                    # 微调后的 FP32 模型
│   ├── ir/                      # FP32 OpenVINO IR 模型
│   ├── quantized/               # INT8 量化模型
│   └── cache/                   # OpenVINO 编译缓存（自动生成）
├── requirements/                # 依赖文件
│   ├── requirements-local.txt   # 本地依赖（数据处理 + 推理）
│   └── requirements-kaggle.txt  # Kaggle 依赖（训练）
├── AGENTS.md
└── README.md
```

## 数据处理规范

### 数据源

- **C-Red** (`data/raw/C-ReD/benchmark data/`)
- 5 个类别：`composition`, `film_review`, `news`, `paper`, `question_answer`
- 9 种 AI 模型：`claude-3.5-haiku`, `deepseek-r1`, `deepseek-v3`, `doubao-1.5-pro`, `gemini-2.5-flash`, `gpt-3.5-turbo`, `gpt-4o`, `qwen-2.5`, `qwen-3`
- label: 1 = human, 0 = AI-generated

### 数据清洗

- 去除空文本或极短文本（长度 < 10 字符）
- 去除重复文本（基于 text 去重）
- 统一列名和格式，human 数据补齐 `prompt` 列为空
- 统一 `label` 字段（int 类型）
- 去除 HTML 标签、多余空白字符
- 确保所有 CSV 列结构一致

### 数据集划分

- **比例**: 8:1:1（训练:验证:测试）
- **训练集目标**: ~30K 条
- **采样策略**: 按类别均匀采样
  - 5 个类别各采 1/5，约 6000 条/类
  - 每类内 human:AI = 1:1（约 3000 human + 3000 AI）
- **分层划分**: 按类别逐层划分，保证各 split 中类别分布和 human/AI 比例一致

### 数据增强（nlpcda）

- 仅对**训练集**进行增强
- 增强方法：全部使用
  - **同/近义词替换** (Similarword): 替换为同/近义词
  - **随机词替换** (Randomword): 随机替换词语
  - **随机删除字符** (RandomDeleteChar): 随机删除部分字符
  - **字符位置交换** (CharPositionExchange): 交换句子中字符位置
  - **等价字替换** (EquivalentChar): 替换为等价字
  - **同音字替换** (Homophone): 替换为同音字
- 增强倍率：每个原始样本生成 2-3 个增强样本
- 注意事项：
  - 增强后的训练集需保持 1:1 平衡
  - 增强样本的 label 与原始样本一致
  - 验证集和测试集不做增强

## 训练规范

### 模型

- **预训练模型**: `hfl/chinese-roberta-wwm-ext`
- **本地路径**: `D:\Dylan\Model\chinese-roberta-wwm-ext`
- **模型类**: `AutoModelForSequenceClassification`（HuggingFace 内置）
- **加载方式**: 所有 `from_pretrained` 均设置 `local_files_only=True`，不从 HuggingFace 自动下载
- **任务**: 二分类（AI-generated text detection）
- **损失函数**: CrossEntropyLoss（模型内置）

### 框架

- PyTorch + Transformers + Datasets + HuggingFace Trainer API
- 多 GPU 训练：Trainer 内置 DDP，使用 torchrun 启动（2×T4）

### 超参配置

```
max_seq_length: 512               # BERT 最大长度，覆盖 ~63% 数据无需截断
batch_size: 8                     # per GPU，2×T4 共 16; T4 16GB 下 8 为安全值
eval_batch_size: 16               # 评估 batch size（可大于训练 batch size）
gradient_accumulation_steps: 2    # 有效 batch_size = 8 × 2 GPU × 2 = 32
learning_rate: 2e-5
weight_decay: 0.01
num_epochs: 3
optimizer: AdamW
scheduler: linear warmup + linear decay
warmup_ratio: 0.1
fp16: true                        # T4 支持混合精度加速
```

### run.ipynb（Kaggle 入口）

该 Notebook 实现从 GitHub 克隆源码并完整执行训练流程：

1. **从 GitHub 克隆**当前仓库到 Kaggle 环境
2. **安装依赖**：`pip install -r /kaggle/working/ai-checker/requirements/requirements-kaggle.txt`
3. **（可选）预加载模型**：将 Kaggle Input 中的模型转为 safetensors 缓存到 `/kaggle/working/`，加速训练子进程加载
4. **启动训练**：使用 `torch.distributed.run` 启动 `trainer.py`（内部使用 HuggingFace Trainer API）
5. **Trainer 自动处理**：分布式初始化、混合精度、梯度累积、日志、评估、checkpoint 保存

### 训练可视化

训练结束后自动使用 matplotlib 生成 PNG 图表，保存到 `models/base/plots/`：

- **loss_curves.png**：train loss + eval loss 双线图
- **metrics_curves.png**：accuracy / precision / recall / f1 随步数变化
- **confusion_matrix.png**：测试集 2×2 混淆矩阵热力图（Human vs AI-Generated）
- **roc_curve.png**：ROC 曲线 + AUC 值
- **dashboard.png**：2×2 综合仪表盘（四合一概览）

图表静默保存，不弹窗。测试集评估使用 `trainer.predict()` 获取逐样本 logits 以支持混淆矩阵和 ROC 绘制。

## 推理规范

### 模型转换与量化

`convert.py` 统一负责模型转换，默认将 PyTorch FP32 模型直接转换为 OpenVINO IR（不量化），也可选做 INT8 量化。

- **FP32 OpenVINO IR 转换（默认，不量化）**
  - 使用 `optimum-intel` 将 PyTorch 模型导出为 OpenVINO IR（FP32 权重）
  - 无需校准数据集，直接转换
  - 输出：FP32 OpenVINO IR 模型（`.xml` + `.bin`）+ tokenizer
- **INT8 量化（可选 `--quantize`）**
  - 使用 `optimum-intel` + `openvino` + `nncf` 实现 INT8 PTQ（Post-Training Quantization）
  - 需要校准数据集：从验证集中抽取 200-500 条，覆盖各类别和 Human/AI 两种标签，不重复使用训练集
  - 针对 Intel GPU (OpenCL) 和 NPU 进行推理优化
  - 输出：量化后的 OpenVINO IR 模型（`.xml` + `.bin`），以及校准后的精度对比报告
- 目标硬件：Intel GPU / NPU

### 推理

- 支持三种模式：FP32（PyTorch）、FP32 OpenVINO IR、INT8 OpenVINO IR
- 输入：单条文本 或 jsonl 批量文件
- 输出：label（0/1）和置信度概率
- 支持 Intel GPU / NPU 设备选择
- 长文本处理：超过 max_length（512 tokens）时自动使用滑动窗口（50% 重叠 token 级切分，log-softmax 平均聚合）

### 性能对比（compare.py）

`compare.py` 独立负责 FP32 IR vs INT8 IR 的性能对比，`predict.py` 不再包含对比功能。

- 加载 FP32 OpenVINO IR 模型（`--fp32-ir-model-path`，默认 `models/ir`）与 INT8 OpenVINO IR 模型（`--int8-model-path`，默认 `models/quantized`）
- 输入：单条文本 或 jsonl 批量文件
- 输出：两个模型的推理耗时、吞吐量（texts/s）、加速比（speedup），以及标签一致性
- 支持 `--device`（CPU/GPU/NPU）和 `--cache-dir`（编译缓存）
- 两模型使用相同设备、batch size 和 max_length，保证对比公平

### 编译缓存（仅 OpenVINO 模式有效）

- 首次推理时 OpenVINO 会对模型进行设备编译（NPU/GPU 耗时较长，CPU 数秒）
- 指定 `--cache-dir` 后，编译结果缓存到磁盘，后续推理直接加载，秒级启动
- 不同 `--device`、`--batch-size`、`--max-length` 组合生成不同缓存
- 默认缓存路径：`models/cache/`
- 删除缓存目录后下次推理会重新编译

## 代码规范

- **优先使用标准库**：能用 HuggingFace / PyTorch 内置 API 的不要自己手写（如 `Trainer` > 手动训练循环、`AutoModelForSequenceClassification` > 自定义 `PreTrainedModel`、`datasets.Dataset` > 自定义 `torch.utils.data.Dataset`）。只有标准库无法满足需求时才自行实现
- 使用清晰的函数和类命名，建议添加类型注解
- 关键配置使用 config 或 yaml 管理，避免硬编码
- 数据处理脚本应为可复现的模块化设计
- 日志使用标准的 logging 模块，输出关键中间结果和数据统计
- 所有本地路径统一使用 `pathlib.Path` 管理

## 依赖管理

### 本地依赖（数据处理 + 推理）

- pandas, numpy
- nlpcda
- torch
- transformers
- optimum-intel, openvino, nncf
- pathlib

### Kaggle 依赖（训练）

- torch
- transformers
- datasets
- pandas, numpy
- scikit-learn
- tqdm
- matplotlib

## 程序交付使用

所有脚本从项目根目录（`ai-checker/`）执行。

### 数据处理

```bash
# 1. 数据清洗：读取 raw/C-ReD 所有 CSV，清洗后按类别保存为 JSONL
python src/data/clean.py

# 2. 数据集划分：读取清洗后的数据，分层采样划分 8:1:1
#    输出 data/processed/train.jsonl / val.jsonl / test.jsonl
python src/data/split.py

# 3. 数据增强：对训练集进行 nlpcda 增强
#    输出 data/augmented/train.jsonl
python src/data/augment.py
```

### 训练

```bash
# 本地调试（单机 CPU/GPU）
python src/train/trainer.py

# Kaggle 提交：将 run.ipynb 上传至 Kaggle，连接 GPU 加速器后运行全部单元格
# Notebook 会自动从 GitHub 克隆源码、安装依赖、加载数据、分布式训练并保存模型
```

### 推理

```bash
# 1. FP32 OpenVINO IR 转换（不量化）：加载微调后的 FP32 模型，直接导出 OpenVINO IR
#    无需校准文件
#    输出 --output-dir 下的 .xml + .bin（示例为 models/ir）
python src/inference/convert.py \
    --model-path models/base \
    --output-dir models/ir

# 2. INT8 量化（可选）：加 --quantize，用 NNCF 量化并导出 OpenVINO IR
#    需要校准文件（从验证集抽取的 200-500 条 JSONL）
#    输出 --output-dir 下的 .xml + .bin（示例为 models/quantized）
python src/inference/convert.py \
    --model-path models/base \
    --output-dir models/quantized \
    --quantize \
    --calibration-file data/processed/val.jsonl \
    --calibration-samples 300

# 3. 单条文本预测（FP32 PyTorch / FP32 IR / INT8 IR）
python src/inference/predict.py --model-path models/base --text "待检测文本"
python src/inference/predict.py --model-path models/ir --text "待检测文本"
python src/inference/predict.py --model-path models/quantized --text "待检测文本"

# 4. 批量文件预测（JSONL 格式，带编译缓存加速首次启动）
python src/inference/predict.py --model-path models/ir --input-file data/test.jsonl \
    --cache-dir models/cache --output-file results.jsonl

# 5. 性能对比：FP32 IR vs INT8 IR（独立程序 compare.py）
python src/inference/compare.py \
    --fp32-ir-model-path models/ir \
    --int8-model-path models/quantized \
    --input-file data/test.jsonl \
    --cache-dir models/cache
```
