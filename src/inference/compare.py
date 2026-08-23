"""
性能对比脚本 —— FP32 OpenVINO IR vs INT8 OpenVINO IR。

加载两个 OpenVINO IR 模型，在同一设备上推理相同输入，
输出推理耗时、吞吐量（texts/s）、加速比（speedup）和标签一致性。

用法:
    # 单条文本对比
    python src/inference/compare.py \
        --fp32-ir-model-path models/ir \
        --int8-model-path models/quantized \
        --text "待检测文本"

    # 批量文件对比
    python src/inference/compare.py \
        --fp32-ir-model-path models/ir \
        --int8-model-path models/quantized \
        --input-file data/test.jsonl \
        --cache-dir models/cache

标签: 0 = AI-Generated, 1 = Human
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import List

from predict import compute_accuracy, load_ov_model, read_input_file, OVClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _to_label_list(results) -> List[int]:
    """将 predict 结果（单条 dict 或 list[dict]）统一为 label_id 列表。"""
    if isinstance(results, list):
        return [r["label_id"] for r in results]
    return [results["label_id"]]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="性能对比 —— FP32 OpenVINO IR vs INT8 OpenVINO IR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--fp32-ir-model-path", type=str, default="models/ir",
                        help="FP32 OpenVINO IR 模型路径（默认: models/ir）")
    parser.add_argument("--int8-model-path", type=str, default="models/quantized",
                        help="INT8 OpenVINO IR 模型路径（默认: models/quantized）")
    parser.add_argument("--device", type=str, default="CPU",
                        choices=["CPU", "GPU", "NPU"],
                        help="OpenVINO 推理设备（默认: CPU）")

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--text", type=str, default=None,
                             help="单条文本对比")
    input_group.add_argument("--input-file", type=str, default=None,
                             help="输入文件路径（JSONL 或纯文本）")

    parser.add_argument("--batch-size", type=int, default=8,
                        help="批量推理大小（默认: 8，CPU 时自动使用 32）")
    parser.add_argument("--max-length", type=int, default=512,
                        help="最大序列长度（默认: 512）")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="OpenVINO 编译缓存目录")
    parser.add_argument("--eval-labels", type=str, default=None,
                        help="标注文件路径（默认使用 --input-file 的 label 字段）")

    args = parser.parse_args()

    if not (args.text or args.input_file):
        parser.print_help()
        return

    ov_batch_size = 32 if args.device == "CPU" else args.batch_size
    static_shape = args.device in ("NPU", "GPU")

    # 加载两个模型（相同设备、batch size、max_length，保证对比公平）
    fp32_model, fp32_tokenizer = load_ov_model(
        args.fp32_ir_model_path,
        device=args.device,
        max_length=args.max_length,
        batch_size=ov_batch_size,
        cache_dir=args.cache_dir,
    )
    int8_model, int8_tokenizer = load_ov_model(
        args.int8_model_path,
        device=args.device,
        max_length=args.max_length,
        batch_size=ov_batch_size,
        cache_dir=args.cache_dir,
    )

    fp32_clf = OVClassifier(
        fp32_model, fp32_tokenizer,
        max_length=args.max_length,
        batch_size=ov_batch_size,
        static_shape=static_shape,
    )
    int8_clf = OVClassifier(
        int8_model, int8_tokenizer,
        max_length=args.max_length,
        batch_size=ov_batch_size,
        static_shape=static_shape,
    )

    if args.text:
        texts = [args.text]
    else:
        texts = read_input_file(args.input_file)
        logger.info("Loaded %d texts from %s", len(texts), args.input_file)

    # 预热：编译/首次推理开销不计入对比
    if texts:
        fp32_clf.predict([texts[0]])
        int8_clf.predict([texts[0]])

    # FP32 IR 推理
    t0 = time.perf_counter()
    fp32_results = fp32_clf.predict(texts)
    fp32_time = time.perf_counter() - t0

    # INT8 IR 推理
    t0 = time.perf_counter()
    int8_results = int8_clf.predict(texts)
    int8_time = time.perf_counter() - t0

    fp32_tps = len(texts) / fp32_time if fp32_time > 0 else 0
    int8_tps = len(texts) / int8_time if int8_time > 0 else 0
    speedup = fp32_time / int8_time if int8_time > 0 else 0

    logger.info("--- FP32 IR vs INT8 IR Comparison ---")
    logger.info("FP32 IR: %.3fs (%.1f texts/s)", fp32_time, fp32_tps)
    logger.info("INT8 IR: %.3fs (%.1f texts/s)", int8_time, int8_tps)
    logger.info("Speedup: %.2fx", speedup)

    fp32_labels = _to_label_list(fp32_results)
    int8_labels = _to_label_list(int8_results)
    match_count = sum(1 for a, b in zip(fp32_labels, int8_labels) if a == b)
    logger.info("Label match: %d/%d (%.1f%%)", match_count, len(fp32_labels),
                100.0 * match_count / len(fp32_labels) if fp32_labels else 0)

    # 评估指标（仅批量输入且含 label 字段时计算）
    labels_file = args.eval_labels or args.input_file
    if args.input_file and labels_file:
        fp32_metrics = compute_accuracy(fp32_results, labels_file)
        int8_metrics = compute_accuracy(int8_results, labels_file)
        if fp32_metrics and int8_metrics:
            logger.info("--- Evaluation Metrics (pos_label=Human) ---")
            logger.info("%-12s | %8s | %8s", "Metric", "FP32 IR", "INT8 IR")
            for m in ("accuracy", "precision", "recall", "f1"):
                logger.info("%-12s | %8.4f | %8.4f", m,
                            fp32_metrics.get(m, 0), int8_metrics.get(m, 0))


if __name__ == "__main__":
    main()
