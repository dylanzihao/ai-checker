"""
数据提取：读取各数据集 loader，合并为统一 schema 的 JSONL。

统一 schema: {text, label, category, source}，label: 1=human, 0=AI。

输出: data/raw/unified.jsonl

用法:
    python src/data/extract.py
    python src/data/extract.py --workers 8   # 并行提取（手动指定进程数）
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data.loaders import cred, maga, nlpcc

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "unified.jsonl"

# 新增数据集：在 loaders/ 新增模块实现 load()，并在此注册
LOADERS = [cred, maga, nlpcc]


def extract(workers: int = 1) -> list[dict]:
    """依次调用各 loader 并合并结果。"""
    records: list[dict] = []
    for loader in LOADERS:
        logger.info("=" * 60)
        logger.info("提取数据集: %s", loader.SOURCE)
        records.extend(loader.load(workers=workers))

    logger.info("=" * 60)
    logger.info("合并完成: total=%d", len(records))
    return records


def save(records: list[dict]) -> None:
    """保存为 JSONL。"""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("已保存到: %s", OUTPUT_FILE)


if __name__ == "__main__":
    mp.freeze_support()

    parser = argparse.ArgumentParser(description="数据提取（C-ReD + MAGA-cn）")
    parser.add_argument("--workers", type=int, default=1,
                        help="并行进程数（默认 1，串行）")
    args = parser.parse_args()

    save(extract(workers=args.workers))
