"""
模型转换脚本 —— PyTorch → OpenVINO IR。

默认将 PyTorch FP32 模型直接转换为 OpenVINO IR（FP32 权重，不量化），
也可通过 --quantize 可选做 NNCF INT8 量化。

导出使用 Optimum Intel（生成与 NPU/GPU 编译器兼容的 IR）。

流程（不量化，默认）:
    1. 加载微调后的 PyTorch 模型（models/base/）
    2. Optimum Intel 导出 → OpenVINO IR（FP32）
    3. 保存 IR + tokenizer 到 --output-dir

流程（--quantize）:
    1. 加载微调后的 PyTorch 模型（models/base/）
    2. Optimum Intel 导出 → OpenVINO IR（FP32）
    3. 准备校准数据（从 JSONL 读取并 tokenize）
    4. core.read_model(ir) → nncf.quantize → INT8 量化
    5. 保存量化后 IR + tokenizer 到 --output-dir

用法:
    # FP32 OpenVINO IR 转换（不量化，默认）
    python src/inference/convert.py --model-path models/base --output-dir models/ir

    # INT8 量化（可选）
    python src/inference/convert.py \
        --model-path models/base \
        --output-dir models/quantized \
        --quantize \
        --calibration-file data/processed/val.jsonl \
        --calibration-samples 300
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile

import nncf
import numpy as np
import openvino as ov
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_pytorch_model(model_path: str):
    """加载微调后的 PyTorch 模型和分词器。"""
    device = torch.device("cpu")
    logger.info("Loading fine-tuned model from %s ...", model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info("Model loaded: %s", type(model).__name__)
    return model, tokenizer


def export_to_openvino_ir(model, tokenizer, output_dir: str) -> str:
    """使用 Optimum Intel 将 PyTorch 模型导出为 OpenVINO IR（FP32，不量化）。"""
    os.makedirs(output_dir, exist_ok=True)

    pt_tmp = os.path.join(output_dir, "_pt_model")
    os.makedirs(pt_tmp, exist_ok=True)
    model.save_pretrained(pt_tmp)
    tokenizer.save_pretrained(pt_tmp)

    logger.info("Exporting OpenVINO IR (FP32) via Optimum Intel ...")
    from optimum.intel import OVModelForSequenceClassification

    ov_model = OVModelForSequenceClassification.from_pretrained(
        pt_tmp,
        export=True,
        trust_remote_code=True,
    )
    ov_model.save_pretrained(output_dir)
    shutil.rmtree(pt_tmp, ignore_errors=True)

    ir_path = os.path.join(output_dir, "openvino_model.xml")
    if not os.path.isfile(ir_path):
        raise FileNotFoundError(f"Expected openvino_model.xml in {output_dir}")

    logger.info("OpenVINO IR exported: %s", ir_path)
    return ir_path


def load_calibration_data(filepath: str, tokenizer, max_length: int, max_samples: int):
    """从 JSONL 读取文本，tokenize 后返回 numpy 数组字典。"""
    texts = []
    logger.info("Reading calibration data from %s ...", filepath)
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
            if len(texts) >= max_samples:
                break

    if not texts:
        raise ValueError(f"No valid samples found in {filepath}")

    logger.info("Tokenizing %d calibration samples (max_length=%d) ...", len(texts), max_length)
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="np",
    )
    result = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in encoded:
        result["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)
    return result


def build_calib_dataset(calib_data: dict, max_samples: int):
    """将 tokenized 数据构造为 nncf.Dataset。"""
    samples = []
    num_samples = min(len(calib_data["input_ids"]), max_samples)
    for i in range(num_samples):
        sample = {key: calib_data[key][i:i + 1] for key in calib_data}
        samples.append(sample)

    logger.info("Built calibration dataset: %d samples", len(samples))
    return nncf.Dataset(samples)


def quantize(ov_model, calib_dataset, subset_size: int):
    """NNCF INT8 量化（权重 + 激活值）。"""
    logger.info("Starting NNCF INT8 quantization (weights + activations) ...")
    os.environ["NNCF_PROGRESS_BAR"] = "0"
    quantized = nncf.quantize(
        ov_model,
        calib_dataset,
        model_type=nncf.ModelType.TRANSFORMER,
        preset=nncf.QuantizationPreset.PERFORMANCE,
        fast_bias_correction=False,
        subset_size=subset_size,
        target_device=nncf.TargetDevice.ANY,
    )
    return quantized


def main():
    parser = argparse.ArgumentParser(
        description="模型转换 —— PyTorch → OpenVINO IR（默认 FP32 不量化，可选 --quantize）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model-path", type=str, default="models/base",
                        help="微调后的 PyTorch 模型路径（默认: models/base）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录（默认: 不量化时为 models/ir，量化时为 models/quantized）")
    parser.add_argument("--quantize", action="store_true",
                        help="启用 NNCF INT8 量化（默认关闭，仅转 FP32 IR）")
    parser.add_argument("--calibration-file", type=str, default="data/processed/val.jsonl",
                        help="校准数据 JSONL 文件（仅 --quantize 时使用，默认: data/processed/val.jsonl）")
    parser.add_argument("--calibration-samples", type=int, default=300,
                        help="校准样本数量（仅 --quantize 时使用，默认: 300）")
    parser.add_argument("--max-length", type=int, default=512,
                        help="校准数据最大序列长度（默认: 512）")
    parser.add_argument("--subset-size", type=int, default=None,
                        help="NNCF 校准子集大小（默认: 与 calibration-samples 一致，上限 300）")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="保留量化时导出的临时 FP32 IR 产物")

    args = parser.parse_args()

    output_dir = args.output_dir or ("models/quantized" if args.quantize else "models/ir")
    subset_size = args.subset_size if args.subset_size is not None else min(args.calibration_samples, 300)

    # 1. 加载 PyTorch 模型
    pt_model, tokenizer = load_pytorch_model(args.model_path)

    if not args.quantize:
        # 2. 直接导出 FP32 IR 到输出目录
        export_to_openvino_ir(pt_model, tokenizer, output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("FP32 OpenVINO IR saved to %s", output_dir)
        return

    # 3. 量化：先导出 FP32 IR 到临时目录，再做 NNCF INT8 量化
    tmp_root = tempfile.mkdtemp(prefix="convert_")
    try:
        ir_dir = os.path.join(tmp_root, "ir")
        ir_path = export_to_openvino_ir(pt_model, tokenizer, ir_dir)

        del pt_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        core = ov.Core()
        ov_model = core.read_model(ir_path)

        calib_data = load_calibration_data(
            args.calibration_file, tokenizer,
            max_length=args.max_length,
            max_samples=args.calibration_samples,
        )
        calib_dataset = build_calib_dataset(calib_data, args.calibration_samples)

        quantized_model = quantize(ov_model, calib_dataset, subset_size)

        os.makedirs(output_dir, exist_ok=True)
        ov.save_model(quantized_model, os.path.join(output_dir, "openvino_model.xml"))
        tokenizer.save_pretrained(output_dir)
        logger.info("Quantized model saved to %s", output_dir)
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
