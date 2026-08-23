"""
推理脚本 —— 支持三种模式：FP32（PyTorch）、FP32 OpenVINO IR、INT8 OpenVINO IR。

- FP32 模式：加载 PyTorch 模型（models/base/），使用 HuggingFace pipeline 推理
- OpenVINO 模式：加载 OpenVINO IR 模型（models/ir 或 models/quantized/），支持 CPU / GPU / NPU 加速

用法:
    # FP32 单条文本推理（PyTorch）
    python src/inference/predict.py --model-path models/base --text "待检测文本"

    # FP32 OpenVINO IR 推理
    python src/inference/predict.py --model-path models/ir --text "待检测文本"

    # INT8 OpenVINO IR 推理（CPU）
    python src/inference/predict.py --model-path models/quantized --device CPU --text "待检测文本"

    # 批量文件推理
    python src/inference/predict.py --model-path models/ir --input-file data/test.jsonl --output-file results.jsonl

性能对比请使用独立程序 compare.py（FP32 IR vs INT8 IR）。

标签: 0 = AI-Generated, 1 = Human
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

LABEL_MAP = {0: "AI", 1: "Human"}
WINDOW_STRIDE_RATIO = 0.5


# ==============================================================================
# 滑动窗口工具函数
# ==============================================================================


def _needs_windowing(tokenizer, text: str, max_length: int) -> bool:
    """检测文本是否超过 max_length，需要滑动窗口。"""
    encoded = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    return len(encoded) > max_length


def _tokenize_windows(tokenizer, text: str, max_length: int) -> List[str]:
    """将长文本切分为滑动窗口子文本（token 级精确切分，保持语义连贯）。

    stride = max_length // 2，即 50% 重叠。
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    total = len(token_ids)
    window_content = max_length - 2  # 预留 [CLS] 和 [SEP]
    stride = max(int(window_content * WINDOW_STRIDE_RATIO), 1)

    windows = []
    start = 0
    while start < total:
        end = min(start + window_content, total)
        window_ids = token_ids[start:end]
        window_text = tokenizer.decode(window_ids, skip_special_tokens=True)
        windows.append(window_text)
        if end >= total:
            break
        start += stride

    return windows


def _predict_with_windows(predict_batch_fn, tokenizer, texts: List[str],
                          max_length: int, batch_size: int) -> np.ndarray:
    """对一批文本进行滑动窗口推理，返回每个文本的 softmax 概率 [n, 2]。

    短文本直接推理；长文本切为滑动窗口，对 log-softmax 取平均后聚合。
    """
    n = len(texts)
    result = np.zeros((n, 2))

    # 1. 分类：短文本 vs 长文本
    short_items = []   # (original_idx, text)
    long_items = []    # (original_idx, text)

    for idx, text in enumerate(texts):
        if _needs_windowing(tokenizer, text, max_length):
            long_items.append((idx, text))
        else:
            short_items.append((idx, text))

    # 2. 短文本：正常批处理
    if short_items:
        short_indices = [i for i, _ in short_items]
        short_texts = [t for _, t in short_items]
        all_short_probs = []
        for i in range(0, len(short_texts), batch_size):
            batch = short_texts[i:i + batch_size]
            probs = predict_batch_fn(batch)
            all_short_probs.append(probs)
        all_short_probs = np.concatenate(all_short_probs, axis=0)
        for i, orig_idx in enumerate(short_indices):
            result[orig_idx] = all_short_probs[i]

    # 3. 长文本：滑动窗口 → 批处理 → log-softmax 平均聚合
    if long_items:
        long_indices = []
        all_windows = []
        window_counts = []

        for orig_idx, text in long_items:
            windows = _tokenize_windows(tokenizer, text, max_length)
            long_indices.append(orig_idx)
            all_windows.extend(windows)
            window_counts.append(len(windows))

        if len(long_items) == 1:
            logger.info("Text[%d]: %d windows (sliding window)",
                        long_indices[0], window_counts[0])
        else:
            total_windows = sum(window_counts)
            logger.info("%d long texts → %d total windows (sliding window)",
                        len(long_items), total_windows)

        # 批量推理所有窗口
        all_window_probs = []
        for i in range(0, len(all_windows), batch_size):
            batch = all_windows[i:i + batch_size]
            probs = predict_batch_fn(batch)
            all_window_probs.append(probs)
        all_window_probs = np.concatenate(all_window_probs, axis=0)

        # log-softmax 平均聚合：mean(log(p)) → softmax
        offset = 0
        for text_i, count in enumerate(window_counts):
            window_probs = all_window_probs[offset:offset + count]
            avg_logprobs = np.mean(np.log(window_probs + 1e-10), axis=0)
            exp_avg = np.exp(avg_logprobs - np.max(avg_logprobs))
            result[long_indices[text_i]] = exp_avg / np.sum(exp_avg)
            offset += count

    return result


# ==============================================================================
# 模型检测
# ==============================================================================


def _is_openvino_model(model_path: str) -> bool:
    """检测模型目录是否为 OpenVINO IR 格式。"""
    return (
        os.path.isfile(os.path.join(model_path, "openvino_model.xml"))
        or os.path.isfile(os.path.join(model_path, "model.onnx"))
    )


# ==============================================================================
# FP32 推理（HuggingFace）
# ==============================================================================


class FP32Classifier:
    """基于 HuggingFace PyTorch 模型的推理分类器。

    长文本自动使用滑动窗口：token 级切分（50% 重叠），log-softmax 平均聚合。
    """

    def __init__(self, model_path: str, max_length: int = 512, batch_size: int = 8):
        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading FP32 model from %s on %s ...", model_path, self.device)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = max_length
        self.batch_size = batch_size

    def _predict_batch(self, texts: List[str]) -> np.ndarray:
        import torch

        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits.cpu().numpy()

        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs

    def predict(self, texts: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        all_probs = _predict_with_windows(
            self._predict_batch, self.tokenizer,
            texts, self.max_length, self.batch_size,
        )

        preds = np.argmax(all_probs, axis=1)
        results = []
        for idx, text in enumerate(texts):
            label_id = int(preds[idx])
            results.append({
                "text": text[:100] + ("..." if len(text) > 100 else ""),
                "label": LABEL_MAP[label_id],
                "label_id": label_id,
                "ai_score": round(float(all_probs[idx, 0]), 4),
                "human_score": round(float(all_probs[idx, 1]), 4),
            })

        return results[0] if single_input else results


# ==============================================================================
# OpenVINO 推理（FP32 IR / INT8 IR）
# ==============================================================================


def load_ov_model(model_dir: str, device: str = "CPU", max_length: int = 512,
                  batch_size: int = 1, cache_dir: Optional[str] = None):
    """加载 OpenVINO 模型并编译到指定设备。"""
    import openvino as ov

    core = ov.Core()

    ir_path = os.path.join(model_dir, "openvino_model.xml")
    onnx_path = os.path.join(model_dir, "model.onnx")

    if os.path.exists(ir_path):
        logger.info("Reading OpenVINO IR model from %s ...", ir_path)
        model = core.read_model(ir_path)
    elif os.path.exists(onnx_path):
        logger.info("Reading ONNX model from %s ...", onnx_path)
        model = core.read_model(onnx_path)
    else:
        raise FileNotFoundError(
            f"No model found in {model_dir} (expected openvino_model.xml or model.onnx)"
        )

    if device in ("NPU", "GPU"):
        logger.info("Reshaping model to static shape: [%d, %d]", batch_size, max_length)
        shape_map = {
            "input_ids": ov.PartialShape([batch_size, max_length]),
            "attention_mask": ov.PartialShape([batch_size, max_length]),
        }
        for inp in model.inputs:
            name = inp.any_name
            if name == "token_type_ids":
                shape_map[name] = ov.PartialShape([batch_size, max_length])
        model.reshape(shape_map)

    compile_config = {}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        compile_config["CACHE_DIR"] = os.path.abspath(cache_dir)

    logger.info("Compiling model for device: %s ...", device)
    compiled_model = core.compile_model(model, device, compile_config)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    logger.info("OpenVINO model loaded successfully on %s", device)
    return compiled_model, tokenizer


class OVClassifier:
    """基于 OpenVINO 的推理分类器，支持 NPU/GPU/CPU。

    长文本自动使用滑动窗口：token 级切分（50% 重叠），log-softmax 平均聚合。
    """

    def __init__(self, compiled_model, tokenizer, max_length: int = 512,
                 batch_size: int = 32, static_shape: bool = False):
        self.model = compiled_model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.static_shape = static_shape

    def _predict_batch(self, texts: List[str]) -> np.ndarray:
        original_count = len(texts)
        padding = "max_length" if self.static_shape else True

        if self.static_shape and original_count < self.batch_size:
            pad_count = self.batch_size - original_count
            texts = texts + [texts[-1]] * pad_count

        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=padding,
            max_length=self.max_length,
            return_tensors="np",
        )
        inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in encoded:
            inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)

        if self.static_shape and original_count < self.batch_size:
            inputs["attention_mask"][original_count:] = 0

        outputs = self.model(inputs)
        logits = outputs[0]

        if self.static_shape and original_count < self.batch_size:
            logits = logits[:original_count]

        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs

    def predict(self, texts: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        all_probs = _predict_with_windows(
            self._predict_batch, self.tokenizer,
            texts, self.max_length, self.batch_size,
        )

        preds = np.argmax(all_probs, axis=1)
        results = []
        for idx, text in enumerate(texts):
            label_id = int(preds[idx])
            results.append({
                "text": text[:100] + ("..." if len(text) > 100 else ""),
                "label": LABEL_MAP[label_id],
                "label_id": label_id,
                "ai_score": round(float(all_probs[idx, 0]), 4),
                "human_score": round(float(all_probs[idx, 1]), 4),
            })

        return results[0] if single_input else results


# ==============================================================================
# 工具函数
# ==============================================================================


def read_input_file(filepath: str) -> List[str]:
    """从 JSONL 或纯文本文件中读取待推理文本。"""
    texts = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                texts.append(obj.get("text", line))
            except json.JSONDecodeError:
                texts.append(line)
    return texts


def output_results(results, output_file: str = None):
    """输出推理结果到文件或控制台。"""
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            if isinstance(results, list):
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps(results, ensure_ascii=False) + "\n")
        logger.info("Results saved to %s", output_file)
    else:
        if isinstance(results, list):
            for r in results:
                label_str = "[人类]" if r["label_id"] == 1 else "[AI  ]"
                line = f"{label_str} human={r['human_score']:.4f} ai={r['ai_score']:.4f} | {r['text']}"
                print(line)
        else:
            r = results
            label_str = "[人类]" if r["label_id"] == 1 else "[AI  ]"
            print(f"{label_str} human={r['human_score']:.4f} ai={r['ai_score']:.4f} | {r['text']}")


def compute_accuracy(results: List[Dict], labels_file: str = None) -> Dict:
    """计算推理准确率（需要标注文件提供 label 字段）。"""
    if labels_file is None:
        return {}

    y_true = []
    y_pred = []
    skipped = 0
    with open(labels_file, "r", encoding="utf-8") as f:
        for line, result in zip(f, results):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "label" in obj and "label_id" in result:
                y_true.append(int(obj["label"]))
                y_pred.append(int(result["label_id"]))
            else:
                skipped += 1

    if not y_true:
        return {"skipped": skipped}

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "samples": len(y_true),
        "skipped": skipped,
    }


def run_interactive(classifier):
    """交互模式。"""
    is_ov = isinstance(classifier, OVClassifier)
    mode_str = "OpenVINO" if is_ov else "FP32(PyTorch)"
    print(f"\n{'=' * 60}")
    print(f"  ai-checker 交互式推理模式 [{mode_str}]")
    print(f"  输入文本后按回车检测，输入 :q 或 :quit 退出")
    print(f"{'=' * 60}\n")

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if text in (":q", ":quit"):
            print("再见！")
            break
        if not text:
            continue

        result = classifier.predict(text)
        label_str = LABEL_MAP[result["label_id"]]
        marker = "[人类]" if result["label_id"] == 1 else "[AI  ]"
        print(f"  {marker} {label_str}  human={result['human_score']:.4f}  ai={result['ai_score']:.4f}\n")


# ==============================================================================
# 入口
# ==============================================================================


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="ai-checker 推理脚本 —— 支持 FP32(PyTorch) 和 OpenVINO(FP32 IR / INT8 IR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model-path", type=str, default="models/base",
                        help="模型路径: FP32 PyTorch 目录 或 OpenVINO IR 目录（默认: models/base）")
    parser.add_argument("--device", type=str, default="CPU",
                        choices=["CPU", "GPU", "NPU"],
                        help="OpenVINO 推理设备（默认: CPU，仅 OpenVINO 模式有效）")

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--text", type=str, default=None,
                             help="单条文本推理")
    input_group.add_argument("--input-file", type=str, default=None,
                             help="输入文件路径（JSONL 或纯文本）")
    input_group.add_argument("--interactive", action="store_true",
                             help="交互模式")

    parser.add_argument("--max-length", type=int, default=512,
                        help="最大序列长度（默认: 512）")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="批量推理大小（默认: 8，OpenVINO 模式 CPU 默认: 32）")
    parser.add_argument("--output-file", type=str, default=None,
                        help="结果输出 JSONL 文件路径")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="OpenVINO 编译缓存目录")
    parser.add_argument("--eval-labels", type=str, default=None,
                        help="标注文件路径（与 --input-file 同源，用于计算准确率）")

    args = parser.parse_args()

    # 如果没有指定 --text / --input-file / --interactive，打印帮助
    if not (args.text or args.input_file or args.interactive):
        parser.print_help()
        return

    # 检测模型类型
    is_ov = _is_openvino_model(args.model_path)
    if is_ov:
        ov_batch_size = 32 if args.device == "CPU" else args.batch_size
    else:
        ov_batch_size = args.batch_size

    # 创建分类器
    if is_ov:
        compiled_model, tokenizer = load_ov_model(
            args.model_path,
            device=args.device,
            max_length=args.max_length,
            batch_size=ov_batch_size,
            cache_dir=args.cache_dir,
        )
        classifier = OVClassifier(
            compiled_model, tokenizer,
            max_length=args.max_length,
            batch_size=ov_batch_size,
            static_shape=(args.device in ("NPU", "GPU")),
        )
        logger.info("Using OpenVINO mode")
    else:
        fp32_batch_size = args.batch_size
        classifier = FP32Classifier(
            args.model_path,
            max_length=args.max_length,
            batch_size=fp32_batch_size,
        )
        logger.info("Using FP32(PyTorch) mode")

    # =====================================================================
    # 推理
    # =====================================================================

    results = None

    if args.interactive:
        run_interactive(classifier)
        return

    if args.input_file:
        texts = read_input_file(args.input_file)
        logger.info("Loaded %d texts from %s", len(texts), args.input_file)
        t0 = time.perf_counter()
        results = classifier.predict(texts)
        elapsed = time.perf_counter() - t0
        logger.info("Inference done in %.2fs (%.1f texts/s)",
                    elapsed, len(texts) / elapsed if elapsed > 0 else 0)
        output_results(results, args.output_file)

        if args.eval_labels:
            metrics = compute_accuracy(results, args.eval_labels)
            if metrics:
                logger.info("Evaluation metrics: %s", json.dumps(metrics, ensure_ascii=False))

    elif args.text:
        t0 = time.perf_counter()
        results = classifier.predict(args.text)
        elapsed = time.perf_counter() - t0
        logger.info("Inference done in %.3fs", elapsed)
        output_results(results, args.output_file)

    del classifier


if __name__ == "__main__":
    main()
