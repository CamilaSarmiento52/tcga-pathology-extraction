import json
from pathlib import Path

import pytest

from src.pipeline.result_writer import build_output_record, write_result, write_summary
from src.schema import PathologyExtraction


@pytest.fixture
def sample_extraction():
    return PathologyExtraction(
        report_id="TCGA-BH-A0B3",
        cancer_type="BRCA",
        primary_site="left breast",
    )


@pytest.fixture
def sample_meta():
    return {
        "model": "openai:gpt-4o",
        "prompt_version": "v1.2",
        "latency_ms": 1234.5,
        "input_tokens": 800,
        "output_tokens": 150,
        "validation_status": "valid",
        "run_id": "run_20260526T120000",
        "created_at": "2026-05-26T12:00:00+00:00",
        "truncated": False,
    }


class TestBuildOutputRecord:
    def test_raises_on_meta_key_collision(self, sample_extraction):
        # schema_version exists in the extraction — meta must not overwrite it silently
        bad_meta = {"schema_version": "v9.9", "model": "openai:gpt-4o"}
        with pytest.raises(ValueError, match="collide"):
            build_output_record(sample_extraction, bad_meta)

    def test_contains_extraction_fields(self, sample_extraction, sample_meta):
        record = build_output_record(sample_extraction, sample_meta)
        assert record["report_id"] == "TCGA-BH-A0B3"
        assert record["primary_site"] == "left breast"
        assert record["schema_version"] == "v1.3"

    def test_contains_meta_fields(self, sample_extraction, sample_meta):
        record = build_output_record(sample_extraction, sample_meta)
        assert record["model"] == "openai:gpt-4o"
        assert record["prompt_version"] == "v1.2"
        assert record["schema_version"] == "v1.3"
        assert record["latency_ms"] == 1234.5
        assert record["input_tokens"] == 800
        assert record["output_tokens"] == 150
        assert record["validation_status"] == "valid"
        assert record["run_id"] == "run_20260526T120000"
        assert record["created_at"] == "2026-05-26T12:00:00+00:00"


class TestWriteResult:
    def test_creates_file(self, tmp_path, sample_extraction, sample_meta):
        out = tmp_path / "results.jsonl"
        write_result(build_output_record(sample_extraction, sample_meta), out)
        assert out.exists()

    def test_appends_not_overwrites(self, tmp_path, sample_extraction, sample_meta):
        out = tmp_path / "results.jsonl"
        record = build_output_record(sample_extraction, sample_meta)
        write_result(record, out)
        write_result(record, out)
        lines = [l for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) == 2

    def test_output_is_valid_json(self, tmp_path, sample_extraction, sample_meta):
        out = tmp_path / "results.jsonl"
        write_result(build_output_record(sample_extraction, sample_meta), out)
        parsed = json.loads(out.read_text().strip())
        assert parsed["report_id"] == "TCGA-BH-A0B3"


class TestWriteSummary:
    def test_creates_json_file(self, tmp_path):
        out = tmp_path / "summary.json"
        write_summary({"total": 20, "pass_rate": 0.9}, out)
        assert out.exists()
        assert json.loads(out.read_text())["pass_rate"] == 0.9

    def test_raises_if_directory_not_found(self, sample_extraction, sample_meta):
        out = Path("/nonexistent/directory/results.jsonl")
        with pytest.raises(FileNotFoundError):
            write_result(build_output_record(sample_extraction, sample_meta), out)
