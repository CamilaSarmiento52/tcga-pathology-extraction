import json
from pathlib import Path

import pytest

from src.pipeline.loader import load_dev_subset, load_records


def make_jsonl(records: list[dict], tmp_path: Path) -> Path:
    p = tmp_path / "test.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records))
    return p


@pytest.fixture
def sample_records():
    styles = ["synoptic", "narrative", "mixed"]
    cancer_types = ["BRCA", "LUAD", "LUSC"]
    return [
        {
            "report_id": f"TCGA-XX-{i:04d}.abc",
            "cancer_type": cancer_types[i % 3],
            "style": styles[i % 3],
            "text": f"Sample report {i}",
        }
        for i in range(60)
    ]


class TestLoadRecords:
    def test_loads_all_records(self, tmp_path, sample_records):
        p = make_jsonl(sample_records, tmp_path)
        assert len(list(load_records(p))) == 60

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text('{"a": 1}\n\n{"b": 2}\n')
        assert len(list(load_records(p))) == 2

    def test_yields_dicts(self, tmp_path, sample_records):
        p = make_jsonl(sample_records[:3], tmp_path)
        for r in load_records(p):
            assert isinstance(r, dict)

    def test_raises_if_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(load_records(tmp_path / "nonexistent.jsonl"))

    def test_raises_on_malformed_json_line(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"valid": 1}\n{"broken: bad json\n')
        with pytest.raises(json.JSONDecodeError):
            list(load_records(p))


class TestLoadDevSubset:
    def test_returns_exact_n(self, tmp_path, sample_records):
        p = make_jsonl(sample_records, tmp_path)
        assert len(load_dev_subset(p, n=10, seed=42)) == 10

    def test_deterministic_same_seed(self, tmp_path, sample_records):
        p = make_jsonl(sample_records, tmp_path)
        s1 = [r["report_id"] for r in load_dev_subset(p, n=10, seed=42)]
        s2 = [r["report_id"] for r in load_dev_subset(p, n=10, seed=42)]
        assert s1 == s2

    def test_different_seeds_differ(self, tmp_path, sample_records):
        p = make_jsonl(sample_records, tmp_path)
        s1 = [r["report_id"] for r in load_dev_subset(p, n=10, seed=42)]
        s2 = [r["report_id"] for r in load_dev_subset(p, n=10, seed=99)]
        assert s1 != s2

    def test_covers_multiple_cancer_types(self, tmp_path, sample_records):
        p = make_jsonl(sample_records, tmp_path)
        subset = load_dev_subset(p, n=15, seed=42)
        assert len({r["cancer_type"] for r in subset}) >= 2
