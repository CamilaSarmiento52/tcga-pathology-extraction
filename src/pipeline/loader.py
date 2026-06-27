import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator


def load_records(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_dev_subset(path: str | Path, n: int = 20, seed: int = 42) -> list[dict]:
    records = list(load_records(path))
    rng = random.Random(seed)

    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        groups[(r.get("cancer_type", ""), r.get("style", ""))].append(r)

    selected = []
    total = len(records)
    for key in sorted(groups):
        group = groups[key]
        quota = max(1, round(n * len(group) / total))
        selected.extend(rng.sample(group, min(quota, len(group))))

    rng.shuffle(selected)

    if len(selected) > n:
        return selected[:n]

    if len(selected) < n:
        selected_ids = {r["report_id"] for r in selected}
        remaining = [r for r in records if r["report_id"] not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    return selected
