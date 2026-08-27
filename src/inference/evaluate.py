"""
评估脚本：对单个模型做详细评估。

支持三种模型（自动检测）：
- FP32 PyTorch（models/base）
- FP32 OpenVINO IR（models/ir）
- INT8 OpenVINO IR（models/quantized）

评估两个数据集（存在则评）：
1. data/processed/test.jsonl —— in-domain 测试集（label 已对齐: 1=human, 0=AI）
2. data/eval/test_with_label.json —— NLPCC OOD 基准（label 翻转: 0=human/1=machine -> 1=human/0=AI）

每个数据集输出：
- 整体指标: accuracy / macro-F1 / per-class P·R·F1
- 混淆矩阵（2×2 + 行百分比）
- ROC + AUC
- 长度分层: <64 / 64-128 / 128-256 / 256-512 / >=512 字符
- 场景细分（仅 test_with_label）: Normal / Attack(混合/释义/扰动) / Varying(64/128/256/512)

用法:
    python src/inference/evaluate.py --model-path models/base
    python src/inference/evaluate.py --model-path models/ir --device GPU
    python src/inference/evaluate.py --model-path models/base --output-dir models/eval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test.jsonl"
DEFAULT_NLPCC_EVAL = PROJECT_ROOT / "data" / "eval" / "test_with_label.json"

# NLPCC test_with_label 场景分段（按 id 值）
SCENES = [
    ("Normal", 1, 4000),
    ("Attack-混合", 4001, 5000),
    ("Attack-释义", 5001, 6000),
    ("Attack-扰动", 6001, 7000),
    ("Varying-64", 7001, 8000),
    ("Varying-128", 8001, 9000),
    ("Varying-256", 9001, 10000),
    ("Varying-512", 10001, 11000),
]

# 长度分层（字符数）
LENGTH_BUCKETS = [
    ("<64", lambda l: l < 64),
    ("64-128", lambda l: 64 <= l < 128),
    ("128-256", lambda l: 128 <= l < 256),
    ("256-512", lambda l: 256 <= l < 512),
    (">=512", lambda l: l >= 512),
]


def _metrics(y_true, y_pred):
    """整体指标（项目约定: 1=human, 0=AI）。"""
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "human_p": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "human_r": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "human_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "ai_p": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "ai_r": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "ai_f1": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
    }


def _print_header(title: str) -> None:
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


def _print_overall(m: dict) -> None:
    print(
        f"  accuracy={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}"
    )
    print(
        f"  Human  P={m['human_p']:.4f}  R={m['human_r']:.4f}  F1={m['human_f1']:.4f}"
    )
    print(
        f"  AI     P={m['ai_p']:.4f}  R={m['ai_r']:.4f}  F1={m['ai_f1']:.4f}"
    )


def _print_confusion_matrix(y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    total = cm.sum()
    print("  混淆矩阵 (行=真实, 列=预测):")
    print("            预测AI   预测Human")
    for i, label in enumerate(["真实AI  ", "真实Human"]):
        row = cm[i]
        pct = f"({row[i] / row.sum() * 100:.1f}%)" if row.sum() > 0 else ""
        print(f"  {label}   {row[0]:>6}    {row[1]:>6}   {pct}")
    print(f"  总计 {total}，正判率 {(cm[0, 0] + cm[1, 1]) / total * 100:.2f}%")


def _print_roc(y_true, y_prob) -> None:
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        print("  ROC/AUC: 无法计算（单类别）")
        return
    print(f"  ROC-AUC (pos=Human) = {auc:.4f}")


def _print_length_buckets(texts, y_true, y_pred) -> None:
    print("  长度分层:")
    print(f"    {'区间':<10} {'样本':>6} {'macro-F1':>9} {'human-R':>9} {'AI-R':>9}")
    for name, cond in LENGTH_BUCKETS:
        idx = [i for i, t in enumerate(texts) if cond(len(t))]
        if not idx:
            continue
        t = [y_true[i] for i in idx]
        p = [y_pred[i] for i in idx]
        print(
            f"    {name:<10} {len(idx):>6} "
            f"{f1_score(t, p, average='macro', zero_division=0):>9.4f} "
            f"{recall_score(t, p, pos_label=1, zero_division=0):>9.4f} "
            f"{recall_score(t, p, pos_label=0, zero_division=0):>9.4f}"
        )


def _print_scenes(ids, y_true, y_pred) -> None:
    print("  场景细分 (按 id):")
    print(f"    {'场景':<14} {'样本':>6} {'macro-F1':>9} {'human-R':>9} {'AI-R':>9}")
    for name, lo, hi in SCENES:
        idx = [i for i, v in enumerate(ids) if lo <= v <= hi]
        if not idx:
            continue
        t = [y_true[i] for i in idx]
        p = [y_pred[i] for i in idx]
        print(
            f"    {name:<14} {len(idx):>6} "
            f"{f1_score(t, p, average='macro', zero_division=0):>9.4f} "
            f"{recall_score(t, p, pos_label=1, zero_division=0):>9.4f} "
            f"{recall_score(t, p, pos_label=0, zero_division=0):>9.4f}"
        )
    # Attack / Varying 合并
    attack_idx = [i for i, v in enumerate(ids) if 4001 <= v <= 7000]
    varying_idx = [i for i, v in enumerate(ids) if 7001 <= v <= 11000]
    for label, idx in [("Attack-整体", attack_idx), ("Varying-整体", varying_idx)]:
        if not idx:
            continue
        t = [y_true[i] for i in idx]
        p = [y_pred[i] for i in idx]
        print(
            f"    {label:<14} {len(idx):>6} "
            f"{f1_score(t, p, average='macro', zero_division=0):>9.4f} "
            f"{recall_score(t, p, pos_label=1, zero_division=0):>9.4f} "
            f"{recall_score(t, p, pos_label=0, zero_division=0):>9.4f}"
        )


def _save_plots(y_true, y_pred, y_prob, output_dir: Path, name: str) -> None:
    """保存混淆矩阵 + ROC 曲线 PNG。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)", ha="center", va="center",
                    fontsize=14, color=color)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["AI-Generated", "Human"])
    ax.set_yticklabels(["AI-Generated", "Human"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {name}")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="#2c7bb6", lw=2, label=f"ROC (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {name}")
        ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
        fig.savefig(output_dir / f"{name}_roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ValueError:
        logger.warning("ROC 无法计算（单类别），跳过 %s 的 ROC 图", name)

    logger.info("PNG 已保存至 %s", output_dir)


def _build_classifier(model_path: str, device: str, max_length: int,
                      batch_size: int, cache_dir: str):
    """复用 predict.py 的推理类，自动检测模型类型。"""
    from src.inference.predict import (
        FP32Classifier,
        OVClassifier,
        _is_openvino_model,
        load_ov_model,
    )

    is_ov = _is_openvino_model(model_path)
    if is_ov:
        ov_batch_size = 32 if device == "CPU" else max(batch_size, 32)
        compiled_model, tokenizer = load_ov_model(
            model_path, device=device, max_length=max_length,
            batch_size=ov_batch_size, cache_dir=cache_dir,
        )
        classifier = OVClassifier(
            compiled_model, tokenizer, max_length=max_length,
            batch_size=ov_batch_size, static_shape=(device in ("NPU", "GPU")),
        )
        return classifier, "OpenVINO"
    classifier = FP32Classifier(model_path, max_length=max_length, batch_size=batch_size)
    return classifier, "FP32(PyTorch)"


def _load_jsonl_texts_labels(filepath: Path):
    """读取项目测试集（label 已对齐，返回 texts, y_true）。"""
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                texts.append(obj["text"])
                labels.append(int(obj["label"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return texts, labels


def _load_nlpcc(filepath: Path):
    """读取 NLPCC OOD 基准（label 翻转，返回 texts, y_true, ids）。"""
    data = json.load(open(filepath, "r", encoding="utf-8"))
    texts = [x["text"] for x in data]
    ids = [x["id"] for x in data]
    y_true = [1 - int(x["label"]) for x in data]  # 0=human -> 1=human
    return texts, y_true, ids


def _run_dataset(classifier, name, texts, y_true, ids, output_dir) -> dict:
    """对一个数据集做完整评估（推理 + 全部指标 + 可选 PNG）。"""
    _print_header(f"数据集: {name}")
    n_human = sum(y_true)
    print(f"  样本数={len(texts)} (human={n_human}, AI={len(texts) - n_human})")

    logger.info("[%s] 推理中 ...", name)
    results = classifier.predict(texts)
    y_pred = [r["label_id"] for r in results]
    y_prob = [r["human_score"] for r in results]

    m = _metrics(y_true, y_pred)
    _print_overall(m)
    _print_confusion_matrix(y_true, y_pred)
    _print_roc(y_true, y_prob)
    _print_length_buckets(texts, y_true, y_pred)
    if ids is not None:
        _print_scenes(ids, y_true, y_pred)

    if output_dir:
        _save_plots(y_true, y_pred, y_prob, output_dir, name)

    return m


def main():
    parser = argparse.ArgumentParser(
        description="ai-checker 详细评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, default="models/base",
                        help="模型路径: FP32 PyTorch 或 OpenVINO IR（默认: models/base）")
    parser.add_argument("--device", type=str, default="CPU",
                        choices=["CPU", "GPU", "NPU"],
                        help="OpenVINO 推理设备（默认: CPU，仅 OpenVINO 模式有效）")
    parser.add_argument("--test-file", type=str, default=str(DEFAULT_TEST_FILE),
                        help=f"项目测试集 JSONL（默认: {DEFAULT_TEST_FILE}）")
    parser.add_argument("--nlpcc-eval", type=str, default=str(DEFAULT_NLPCC_EVAL),
                        help=f"NLPCC OOD 基准 JSON（默认: {DEFAULT_NLPCC_EVAL}）")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="批量推理大小（默认: 32）")
    parser.add_argument("--max-length", type=int, default=512,
                        help="最大序列长度（默认: 512）")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="OpenVINO 编译缓存目录")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="可选，输出混淆矩阵/ROC PNG 的目录")
    args = parser.parse_args()

    classifier, mode = _build_classifier(
        args.model_path, args.device, args.max_length,
        args.batch_size, args.cache_dir,
    )
    logger.info("模型类型: %s", mode)

    output_dir = Path(args.output_dir) if args.output_dir else None

    # 1. 项目测试集
    test_file = Path(args.test_file)
    if test_file.exists():
        texts, y_true = _load_jsonl_texts_labels(test_file)
        _run_dataset(classifier, "test", texts, y_true, None, output_dir)
    else:
        logger.warning("测试集不存在，跳过: %s", test_file)

    # 2. NLPCC OOD 基准
    nlpcc_file = Path(args.nlpcc_eval)
    if nlpcc_file.exists():
        texts, y_true, ids = _load_nlpcc(nlpcc_file)
        _run_dataset(classifier, "NLPCC-test_with_label", texts, y_true, ids, output_dir)
    else:
        logger.warning("NLPCC 评估集不存在，跳过: %s", nlpcc_file)

    _print_header("评估完成")


if __name__ == "__main__":
    main()
