"""
MAGA-cn 数据集 loader。

- 数据路径: data/raw/MAGA-cn/train/MAGA-cn_train.jsonl + val/MAGA-cn_val.jsonl
- 仅使用 MAGA-cn 文件，不使用 MGB
- model == "human" 为人类文本 (label=1)，其余为机器文本 (label=0)
- 本身已 1:1 平衡，全部保留
- 支持并行解析（workers > 1）
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SOURCE = "MAGA-cn"
FILES = [
    PROJECT_ROOT / "data" / "raw" / "MAGA-cn" / "train" / "MAGA-cn_train.jsonl",
    PROJECT_ROOT / "data" / "raw" / "MAGA-cn" / "val" / "MAGA-cn_val.jsonl",
]


def _parse_lines(lines: list[str]) -> list[dict]:
    """解析一批 JSONL 行，返回统一 schema 记录。"""
    records: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = obj.get("text") or ""
        label = 1 if obj.get("model") == "human" else 0
        records.append({
            "text": text,
            "label": label,
            "category": obj.get("domain") or "",
            "source": SOURCE,
        })
    return records


def _chunk_lines(lines: list[str], num_chunks: int) -> list[list[str]]:
    """将行列表拆分为 num_chunks 个大致均匀的块。"""
    num_chunks = max(1, num_chunks)
    chunk_size = max(1, len(lines) // num_chunks)
    chunks: list[list[str]] = []
    for i in range(0, len(lines), chunk_size):
        chunks.append(lines[i:i + chunk_size])
    return chunks


def load(workers: int = 1) -> list[dict]:
    """加载 MAGA-cn 数据（train + val），全部保留。"""
    records: list[dict] = []
    for filepath in FILES:
        if not filepath.exists():
            logger.error("MAGA-cn 文件不存在: %s", filepath)
            continue

        logger.info("读取 %s", filepath.name)
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if workers <= 1:
            file_records = _parse_lines(lines)
        else:
            chunks = _chunk_lines(lines, workers * 4)
            file_records = []
            with mp.Pool(processes=workers) as pool:
                for chunk_records in pool.imap(_parse_lines, chunks):
                    file_records.extend(chunk_records)

        records.extend(file_records)

    human = sum(1 for r in records if r["label"] == 1)
    logger.info("MAGA-cn: total=%d (human=%d, ai=%d)",
                len(records), human, len(records) - human)
    return records
