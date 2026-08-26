"""
数据清洗：读取 data/raw/unified.jsonl，清洗后保存。

清洗步骤（按顺序）:
1. 去除 HTML 标签
2. 去除 URL
3. 去除 emoji
4. 去除控制字符
5. 去除乱码字符（U+FFFD）
6. 繁→简转换（OpenCC）
7. 空白归一 + 去首尾空格
8. 去包裹引号
9. 去除空文本或极短文本（长度 < 10 字符）
+ 去除重复文本（基于 text 去重，串行）

输出: data/cleaned/cleaned.jsonl

用法:
    python src/data/clean.py
    python src/data/clean.py --workers 8   # 并行清洗（手动指定进程数）
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import re
import sys
from pathlib import Path

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
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "unified.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "cleaned" / "cleaned.jsonl"

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
WHITESPACE_RE = re.compile(r"\s+")

MIN_TEXT_LENGTH = 10
QUOTE_CHARS = frozenset('"\u201c\u201d\u2018\u2019`')


def _build_emoji_re() -> re.Pattern:
    """构造 emoji 匹配正则（覆盖主符号区 + 杂项符号 + 变体选择符）。"""
    ranges = [(0x1F000, 0x1FAFF), (0x2600, 0x27BF), (0x2B00, 0x2BFF)]
    parts = [f"{chr(a)}-{chr(b)}" for a, b in ranges]
    parts.append(chr(0xFE0F))
    return re.compile("[" + "".join(parts) + "]")


EMOJI_RE = _build_emoji_re()

# 并行 worker 中的 OpenCC 全局实例
_worker_cc = None


def _init_worker() -> None:
    """Worker 进程初始化：每个进程创建一个 OpenCC 实例。"""
    global _worker_cc
    _worker_cc = None
    try:
        from opencc import OpenCC
        _worker_cc = OpenCC("t2s")
    except Exception:
        logger.warning("opencc 不可用，跳过繁→简转换")


def _strip_wrapping_quotes(text: str) -> str:
    """循环剥离首尾成对的引号（ASCII/中文引号/反引号）。"""
    while len(text) >= 2 and text[0] in QUOTE_CHARS and text[-1] in QUOTE_CHARS:
        text = text[1:-1].strip()
    return text


def clean_text(text: str, cc=None) -> str:
    """清洗单条文本。"""
    if not isinstance(text, str):
        return ""
    text = HTML_TAG_RE.sub("", text)
    text = URL_RE.sub("", text)
    text = EMOJI_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    text = text.replace("\ufffd", "")
    if not text:
        return ""
    if cc is not None:
        text = cc.convert(text)
    text = WHITESPACE_RE.sub(" ", text)
    text = text.strip()
    text = _strip_wrapping_quotes(text)
    return text


def load_records(filepath: Path) -> list[dict]:
    """读取 JSONL 为记录列表。"""
    records: list[dict] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _clean_chunk(records: list[dict]) -> list[dict]:
    """清洗一个分块，返回保留的记录。"""
    global _worker_cc
    cc = _worker_cc
    cleaned: list[dict] = []
    for record in records:
        text = clean_text(record.get("text", ""), cc)
        if len(text) < MIN_TEXT_LENGTH:
            continue
        record["text"] = text
        cleaned.append(record)
    return cleaned


def _chunk_records(records: list[dict], num_chunks: int) -> list[list[dict]]:
    """将记录拆分为 num_chunks 个大致均匀的块。"""
    num_chunks = max(1, num_chunks)
    chunk_size = max(1, len(records) // num_chunks)
    chunks: list[list[dict]] = []
    for i in range(0, len(records), chunk_size):
        chunks.append(records[i:i + chunk_size])
    return chunks


def clean(workers: int = 1) -> list[dict]:
    """主清洗流程：并行清洗 + 串行去重。"""
    records = load_records(INPUT_FILE)
    logger.info("读取 %d 条", len(records))

    if workers <= 1:
        _init_worker()
        cleaned = _clean_chunk(records)
    else:
        chunks = _chunk_records(records, workers * 4)
        cleaned = []
        with mp.Pool(processes=workers, initializer=_init_worker) as pool:
            for chunk in pool.imap(_clean_chunk, chunks):
                cleaned.extend(chunk)

    logger.info("清洗后（去重前）: %d 条", len(cleaned))

    # 串行去重（基于 text）
    seen: set[str] = set()
    deduped: list[dict] = []
    for record in cleaned:
        text = record["text"]
        if text not in seen:
            seen.add(text)
            deduped.append(record)

    logger.info("去重后: %d 条", len(deduped))
    return deduped


def save(records: list[dict]) -> None:
    """保存为 JSONL。"""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("已保存到: %s", OUTPUT_FILE)


if __name__ == "__main__":
    mp.freeze_support()

    parser = argparse.ArgumentParser(description="数据清洗")
    parser.add_argument("--workers", type=int, default=1,
                        help="并行进程数（默认 1，串行）")
    args = parser.parse_args()

    save(clean(workers=args.workers))
