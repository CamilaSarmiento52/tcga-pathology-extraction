import json
from pathlib import Path

from src.schema import PathologyExtraction


def build_output_record(extraction: PathologyExtraction, meta: dict) -> dict:
    record = extraction.model_dump()
    collisions = record.keys() & meta.keys()
    if collisions:
        raise ValueError(
            f"build_output_record: meta keys collide with extraction fields: {sorted(collisions)}. "
            "Rename the meta keys to avoid silently overwriting extracted values."
        )
    record.update(meta)
    return record


def write_result(record: dict, out_path: str | Path) -> None:
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def write_summary(stats: dict, out_path: str | Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
