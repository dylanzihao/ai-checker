"""
NLPCC-2025-Task1 数据集 loader。

- 数据路径: data/raw/NLPCC-2025-Task1/train.json + dev.json
- 仅使用 train + dev（test_with_label 保留为独立 OOD 评估基准，不进训练）
- label 约定: 原始 0=human/1=machine -> 翻转后 1=human/0=AI（与项目统一）
- category: train 用 source 字段（ASAP/CNewSum/CSL），dev 无 source 用默认 "NLPCC"
- 全量保留（不裁剪）：原始 3:1 不平衡，由 MAGA 的 AI 裁剪抵消，全局达到 1:1
- source = "NLPCC-2025-Task1"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "NLPCC-2025-Task1"
SOURCE = "NLPCC-2025-Task1"
FILES = ["train.json", "dev.json"]

SOURCE_MAP = {
    "asap": "ASAP",
    "cnewsum": "CNewSum",
    "csl": "CSL",
}
DEV_CATEGORY = "NLPCC"


def _read_json_array(filepath: Path) -> list[dict]:
    """读取一个 JSON 数组文件，转为统一 schema 记录（label 已翻转）。"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: list[dict] = []
    for obj in data:
        text = (obj.get("text") or "").strip()
        raw_label = obj.get("label")
        if not text or raw_label is None:
            continue
        label = 1 - int(raw_label)  # 翻转: 0=human -> 1=human
        src = (obj.get("source") or "").lower()
        category = SOURCE_MAP.get(src, DEV_CATEGORY)
        records.append({
            "text": text,
            "label": label,
            "category": category,
            "source": SOURCE,
        })
    return records


def load(workers: int = 1) -> list[dict]:
    """加载 NLPCC-2025-Task1 数据（train + dev），label 翻转，全量保留。"""
    if not RAW_DIR.exists():
        logger.error("NLPCC-2025-Task1 原始数据目录不存在: %s", RAW_DIR)
        return []

    records: list[dict] = []
    for fname in FILES:
        filepath = RAW_DIR / fname
        if not filepath.exists():
            logger.error("文件不存在: %s", filepath)
            continue
        logger.info("读取 %s", fname)
        records.extend(_read_json_array(filepath))

    human = sum(1 for r in records if r["label"] == 1)
    logger.info("NLPCC-2025-Task1: total=%d (human=%d, ai=%d)",
                len(records), human, len(records) - human)
    return records
