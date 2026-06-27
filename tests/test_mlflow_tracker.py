"""Tests for MLflowTracker context manager."""

from __future__ import annotations


import mlflow
import pytest

from src.pipeline.mlflow_tracker import MLflowTracker


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path):
    """Redirect MLflow to a temp directory for every test."""
    uri = f"sqlite:///{tmp_path}/mlruns.db"
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/cleanup.db")


_PARAMS = {
    "model": "openai:o4-mini",
    "prompt_version": "v1.4",
    "few_shot_path": "data/annotations/few_shot_examples_v1.3.jsonl",
    "schema_version": "v1.2",
    "record_count": 10,
}


class TestMLflowTrackerBasic:
    def test_run_created_and_finished(self, isolated_mlflow):
        with MLflowTracker(params=_PARAMS):
            pass
        runs = mlflow.search_runs(experiment_names=["pathology-extraction"])
        assert len(runs) == 1
        assert runs.iloc[0]["status"] == "FINISHED"

    def test_params_logged(self, isolated_mlflow):
        with MLflowTracker(params=_PARAMS):
            pass
        runs = mlflow.search_runs(experiment_names=["pathology-extraction"])
        row = runs.iloc[0]
        assert row["params.model"] == "openai:o4-mini"
        assert row["params.prompt_version"] == "v1.4"
        assert row["params.schema_version"] == "v1.2"
        assert row["params.record_count"] == "10"

    def test_metrics_logged(self, isolated_mlflow):
        with MLflowTracker(params=_PARAMS) as tracker:
            tracker.log_metrics({"weighted_f1": 0.75, "hallucination_rate": 0.05})
        runs = mlflow.search_runs(experiment_names=["pathology-extraction"])
        row = runs.iloc[0]
        assert abs(row["metrics.weighted_f1"] - 0.75) < 1e-6
        assert abs(row["metrics.hallucination_rate"] - 0.05) < 1e-6

    def test_artifact_uploaded(self, isolated_mlflow, tmp_path):
        artifact_file = tmp_path / "results.jsonl"
        artifact_file.write_text('{"report_id": "r1"}\n')
        with MLflowTracker(params=_PARAMS) as tracker:
            tracker.log_artifact(artifact_file)
        runs = mlflow.search_runs(experiment_names=["pathology-extraction"])
        run_id = runs.iloc[0]["run_id"]
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        assert any(a.path == "results.jsonl" for a in artifacts)

    def test_run_id_accessible(self, isolated_mlflow):
        with MLflowTracker(params=_PARAMS) as tracker:
            rid = tracker.run_id
        assert rid is not None
        assert len(rid) > 0

    def test_exception_does_not_suppress(self, isolated_mlflow):
        with pytest.raises(ValueError, match="test error"):
            with MLflowTracker(params=_PARAMS):
                raise ValueError("test error")
        # Run should still be ended (FAILED status)
        runs = mlflow.search_runs(
            experiment_names=["pathology-extraction"],
            filter_string="",
        )
        assert len(runs) == 1

    def test_missing_artifact_path_ignored(self, isolated_mlflow):
        with MLflowTracker(params=_PARAMS) as tracker:
            tracker.log_artifact("/nonexistent/path/file.jsonl")
        # Should not raise

    def test_run_name_format(self, isolated_mlflow):
        with MLflowTracker(params=_PARAMS):
            pass
        runs = mlflow.search_runs(experiment_names=["pathology-extraction"])
        name = runs.iloc[0]["tags.mlflow.runName"]
        assert "openai-o4-mini" in name
        assert "v1.4" in name
