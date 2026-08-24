"""
数据集划分：读取清洗后的数据，1:1 下采样后按 (category, label) 分层划分 8:1:1。

- 比例: 8:1:1（训练:验证:测试）
- 1:1 下采样: 划分前按 label 分组，多数类随机下采样到少数类数量（seed=42）
- 分层: 按 (category, label) 逐层划分，保证各 split 类别分布和 human/AI 比例一致
- 全量保留，固定随机种子（42）保证可复现
- 最终输出仅含 text + label

输出: data/processed/train.jsonl / val.jsonl / test.jsonl
"""

from __future__ import annotations

import json
import logging
import random
import sys
from collections import defaultdict
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
INPUT_FILE = PROJECT_ROOT / "data" / "cleaned" / "cleaned.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
RANDOM_SEED = 42


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


def rebalance_1to1(records: list[dict], seed: int) -> list[dict]:
    """按 label 分组，多数类随机下采样到少数类数量（1:1）。"""
    rng = random.Random(seed)
    humans = [r for r in records if r["label"] == 1]
    ais = [r for r in records if r["label"] == 0]
    n = min(len(humans), len(ais))
    rng.shuffle(humans)
    rng.shuffle(ais)
    logger.info("1:1 下采样: human=%d, ai=%d → 各 %d", len(humans), len(ais), n)
    return humans[:n] + ais[:n]


def stratified_split(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """按 (category, label) 分层划分 8:1:1。"""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r.get("category"), r.get("label"))].append(r)

    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []

    for group in groups.values():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    # 全局洗牌
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_split(filepath: Path, records: list[dict]) -> None:
    """写出一个 split，仅保留 text + label。"""
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(
                {"text": r["text"], "label": r["label"]},
                ensure_ascii=False,
            ) + "\n")


def split_and_save() -> None:
    """主划分流程。"""
    records = load_records(INPUT_FILE)
    logger.info("读取 %d 条", len(records))

    records = rebalance_1to1(records, RANDOM_SEED)

    train, val, test = stratified_split(records, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, data in [("train.jsonl", train), ("val.jsonl", val), ("test.jsonl", test)]:
        write_split(OUTPUT_DIR / filename, data)
        human = sum(1 for r in data if r["label"] == 1)
        logger.info("%s: %d 条 (Human: %d, AI: %d)",
                    filename, len(data), human, len(data) - human)


if __name__ == "__main__":
    split_and_save()
