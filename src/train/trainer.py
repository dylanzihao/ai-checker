"""
训练主逻辑：使用 HuggingFace Trainer API，支持 DDP + FP16。

执行方式：
    # 单卡
    python src/train/trainer.py

    # 多卡（torchrun）
    torchrun --nproc_per_node=2 src/train/trainer.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# 确保项目根目录在 Python path 中（兼容 torchrun 子进程）
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
)

from src.train.config import TrainConfig, default_config
from src.train.visualize import _softmax, visualize_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred) -> dict[str, float]:
    """分类指标回调（Trainer 调用）。"""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def load_datasets(
    data_dir: str,
    tokenizer,
    max_length: int,
) -> DatasetDict:
    """加载 JSONL 数据并批量 tokenize。"""
    data_files = {
        "train": str(Path(data_dir) / "train.jsonl"),
        "val": str(Path(data_dir) / "val.jsonl"),
    }
    test_file = Path(data_dir) / "test.jsonl"
    if test_file.exists():
        data_files["test"] = str(test_file)

    dataset = load_dataset("json", data_files=data_files)

    def tokenize_fn(examples: dict) -> dict:
        result = tokenizer(
            examples["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        result["labels"] = examples["label"]
        return result

    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text", "label"],
    )
    dataset.set_format("torch")

    for split, ds in dataset.items():
        human = int(sum(ds["labels"]))
        logger.info(f"  {split}: {len(ds)} 样本 (Human: {human}, AI: {len(ds) - human})")

    return dataset


def train(config: TrainConfig | None = None) -> None:
    """完整训练流程（Trainer API）。"""
    if config is None:
        config = default_config

    if Path("/kaggle").exists() and not torch.cuda.is_available():
        raise SystemExit("没有 GPU")

    # ---- Tokenizer ----
    model_name = config.resolve_model_name()
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    logger.info(f"Tokenizer 加载完成: {model_name}")

    # ---- 数据 ----
    data_dir = config.resolve_data_dir()
    logger.info(f"数据目录: {data_dir}")
    dataset = load_datasets(data_dir, tokenizer, config.max_seq_length)

    # ---- 模型 ----
    logger.info("初始化模型...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=config.num_labels,
        local_files_only=True,
    )
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Trainer ----
    training_args = config.make_training_args()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        compute_metrics=compute_metrics,
    )

    # ---- 训练 ----
    logger.info("开始训练...")
    trainer.train()

    # ---- 训练过程可视化 ----
    log_history = trainer.state.log_history
    logger.info(f"训练日志条数: {len(log_history)}")

    # ---- 测试集评估 + 可视化 ----
    if "test" in dataset:
        logger.info("测试集评估...")
        preds_output = trainer.predict(dataset["test"])
        y_true = preds_output.label_ids
        y_probs = _softmax(preds_output.predictions)

        y_pred = np.argmax(y_probs, axis=1)
        logger.info(
            f"  Test | Loss: {preds_output.metrics['test_loss']:.4f} | "
            f"Acc: {accuracy_score(y_true, y_pred):.4f} | "
            f"P: {precision_score(y_true, y_pred, zero_division=0):.4f} | "
            f"R: {recall_score(y_true, y_pred, zero_division=0):.4f} | "
            f"F1: {f1_score(y_true, y_pred, zero_division=0):.4f}",
        )

        visualize_all(log_history, y_true, y_probs, training_args.output_dir)

    # ---- 保存 ----
    logger.info("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"模型已保存至: {training_args.output_dir}")


if __name__ == "__main__":
    train()
