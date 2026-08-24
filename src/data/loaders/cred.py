"""
C-ReD 数据集 loader。

- 数据路径: data/raw/C-ReD/benchmark data/<category>/*.csv
- 5 个类别，9 种 AI 模型 + 1 个人类
- 使用策略: 人类文本全取；机器文本按类别裁剪到与人类等量（按模型均匀抽样）
- 支持并行读取（workers > 1）
"""

from __future__ import annotations

import csv
import logging
import multiprocessing as mp
import random
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "C-ReD" / "benchmark data"
SOURCE = "C-ReD"
RANDOM_SEED = 42


def _read_csv_file(filepath: str) -> list[dict]:
    """读取单个 CSV 文件，返回记录（含临时 model 字段）。"""
    path = Path(filepath)
    category = path.parent.name.replace(" ", "_")
    is_human = "human" in path.stem.lower()
    label = 1 if is_human else 0
    model = path.stem.rsplit("_", 1)[-1]

    records: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            records.append({
                "text": text,
                "label": label,
                "category": category,
                "source": SOURCE,
                "model": model,
            })
    return records


def _sample_evenly(groups: dict[str, list], target: int, rng: random.Random) -> list:
    """跨模型组轮询均匀抽样，直到达到 target 数量（不足则全取）。"""
    names = list(groups.keys())
    if not names:
        return []
    for name in names:
        rng.shuffle(groups[name])

    idx = {name: 0 for name in names}
    result: list = []
    while len(result) < target:
        progressed = False
        for name in names:
            if idx[name] < len(groups[name]):
                result.append(groups[name][idx[name]])
                idx[name] += 1
                progressed = True
                if len(result) >= target:
                    break
        if not progressed:
            break
    return result


def load(workers: int = 1) -> list[dict]:
    """加载 C-ReD 数据：人类全取 + 机器按类别裁剪到与人类等量。"""
    if not RAW_DIR.exists():
        logger.error("C-ReD 原始数据目录不存在: %s", RAW_DIR)
        return []

    csv_files = sorted(RAW_DIR.glob("*/*.csv"))
    logger.info("发现 %d 个 C-ReD CSV 文件", len(csv_files))

    raw_records: list[dict] = []
    if workers <= 1:
        for f in csv_files:
            raw_records.extend(_read_csv_file(str(f)))
    else:
        with mp.Pool(processes=workers) as pool:
            for records in pool.imap(_read_csv_file, [str(f) for f in csv_files]):
                raw_records.extend(records)

    # 串行裁剪：按 category 把机器文本裁剪到与该类人类等量
    human_by_cat: dict[str, list] = defaultdict(list)
    ai_by_cat_model: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in raw_records:
        if r["label"] == 1:
            human_by_cat[r["category"]].append(r)
        else:
            ai_by_cat_model[r["category"]][r["model"]].append(r)

    rng = random.Random(RANDOM_SEED)
    result: list[dict] = []
    for cat in sorted(human_by_cat.keys()):
        humans = human_by_cat[cat]
        target = len(humans)
        ai_models = ai_by_cat_model.get(cat, {})
        total_ai = sum(len(v) for v in ai_models.values())
        sampled_ai = _sample_evenly(ai_models, target, rng)
        result.extend(humans)
        result.extend(sampled_ai)
        logger.info("  %s: human=%d, ai=%d (裁剪自 %d)",
                    cat, target, len(sampled_ai), total_ai)

    for r in result:
        r.pop("model", None)

    return result
