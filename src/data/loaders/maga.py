"""
MAGA-cn 数据集 loader。

- 数据路径: data/raw/MAGA-cn/train/MAGA-cn_train.jsonl + val/MAGA-cn_val.jsonl
- 仅使用 MAGA-cn 文件，不使用 MGB
- model == "human" 为人类文本 (label=1)，其余为机器文本 (label=0)
- 平衡策略: AI 文本按 domain 裁剪到 human 的 AI_RATIO 倍，以抵消 NLPCC 的 3:1 不平衡，
  使全局（C-ReD 1:1 + NLPCC 3:1 + MAGA）达到 1:1
- 支持并行解析（workers > 1）
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import random
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SOURCE = "MAGA-cn"
FILES = [
    PROJECT_ROOT / "data" / "raw" / "MAGA-cn" / "train" / "MAGA-cn_train.jsonl",
    PROJECT_ROOT / "data" / "raw" / "MAGA-cn" / "val" / "MAGA-cn_val.jsonl",
]

# AI 裁剪到 human 的 AI_RATIO 倍，使全局 1:1：
#   全局 human = 10997(C-ReD) + 72000(MAGA) + 9200(NLPCC) = 92197
#   全局 AI 目标 = 92197 = 10997(C-ReD) + MAGA_AI + 26000(NLPCC)
#   => MAGA_AI = 55200 = 72000 * 0.7667
AI_RATIO = 0.7667
RANDOM_SEED = 42


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


def _balance_ai(records: list[dict], rng: random.Random) -> list[dict]:
    """按 domain 分组，AI 裁剪到 human 的 AI_RATIO 倍（抵消 NLPCC 的 3:1）。"""
    human_by_domain: dict[str, list] = defaultdict(list)
    ai_by_domain: dict[str, list] = defaultdict(list)
    for r in records:
        if r["label"] == 1:
            human_by_domain[r["category"]].append(r)
        else:
            ai_by_domain[r["category"]].append(r)

    result: list[dict] = []
    for domain in sorted(human_by_domain):
        humans = human_by_domain[domain]
        target = int(len(humans) * AI_RATIO)
        ais = ai_by_domain.get(domain, [])
        rng.shuffle(ais)
        result.extend(humans)
        result.extend(ais[:target])
        logger.info("  %s: human=%d, ai=%d (裁剪自 %d)",
                    domain, len(humans), min(target, len(ais)), len(ais))
    return result


def load(workers: int = 1) -> list[dict]:
    """加载 MAGA-cn 数据（train + val），AI 按 domain 裁剪到 human 的 AI_RATIO 倍。"""
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

    rng = random.Random(RANDOM_SEED)
    records = _balance_ai(records, rng)

    human = sum(1 for r in records if r["label"] == 1)
    logger.info("MAGA-cn: total=%d (human=%d, ai=%d)",
                len(records), human, len(records) - human)
    return records
