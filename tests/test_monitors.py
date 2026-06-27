"""Tests for pipeline quality monitors."""

from __future__ import annotations

import mlflow
import pytest

from src.monitoring.field_null_rate_monitor import check_field_null_rate_drift
from src.monitoring.hallucination_alert import check_hallucination_rate
from src.monitoring.pass_rate_monitor import check_pass_rate_drift
from src.monitoring.tier1_drift import check_tier1_drift
from src.monitoring.token_usage_monitor import check_token_usage_drift


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path):
    uri = f"sqlite:///{tmp_path}/mlruns.db"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("pathology-extraction")
    yield uri
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/cleanup.db")


def _create_run(tier1_rate: float, hallucination_rate: float, pass_rate: float = 0.95) -> str:
    """Helper: create a finished MLflow run with given metrics."""
    with mlflow.start_run() as run:
        mlflow.log_metrics({"tier1_rate": tier1_rate, "hallucination_rate": hallucination_rate, "pass_rate": pass_rate})
    return run.info.run_id


def _create_run_with_null_rates(null_rates: dict[str, float]) -> str:
    """Helper: create a finished MLflow run with null_rate_* metrics."""
    with mlflow.start_run() as run:
        mlflow.log_metrics({"tier1_rate": 0.5, "hallucination_rate": 0.05, "pass_rate": 0.99, **null_rates})
    return run.info.run_id


class TestHallucinationAlert:
    def test_ok_below_threshold(self, isolated_mlflow):
        run_id = _create_run(tier1_rate=0.5, hallucination_rate=0.04)
        result = check_hallucination_rate(run_id)
        assert result["alert"] is False
        assert result["level"] == "OK"
        assert result["should_pause"] is False

    def test_ok_at_threshold(self, isolated_mlflow):
        run_id = _create_run(tier1_rate=0.5, hallucination_rate=0.08)
        result = check_hallucination_rate(run_id)
        assert result["alert"] is False
        assert result["level"] == "OK"

    def test_critical_above_threshold(self, isolated_mlflow):
        run_id = _create_run(tier1_rate=0.5, hallucination_rate=0.09)
        result = check_hallucination_rate(run_id)
        assert result["alert"] is True
        assert result["level"] == "CRITICAL"
        assert result["should_pause"] is True

    def test_rate_value_in_result(self, isolated_mlflow):
        run_id = _create_run(tier1_rate=0.5, hallucination_rate=0.15)
        result = check_hallucination_rate(run_id)
        assert abs(result["rate"] - 0.15) < 1e-4

    def test_missing_metric_returns_ok(self, isolated_mlflow):
        with mlflow.start_run() as run:
            mlflow.log_metrics({"some_other_metric": 1.0})
        result = check_hallucination_rate(run.info.run_id)
        assert result["alert"] is False
        assert result["level"] == "OK"


class TestTier1Drift:
    def test_ok_within_range(self, isolated_mlflow):
        # Seed 5 baseline runs with ~0.60 tier1_rate
        for _ in range(5):
            _create_run(tier1_rate=0.60, hallucination_rate=0.05)
        # Current run: 0.65 — only 5pp delta
        current_id = _create_run(tier1_rate=0.65, hallucination_rate=0.05)
        result = check_tier1_drift(current_id)
        assert result["alert"] is False
        assert result["level"] == "OK"

    def test_warn_above_threshold(self, isolated_mlflow):
        for _ in range(5):
            _create_run(tier1_rate=0.60, hallucination_rate=0.05)
        # Current run: 0.40 — 20pp delta, should trigger WARN
        current_id = _create_run(tier1_rate=0.40, hallucination_rate=0.05)
        result = check_tier1_drift(current_id)
        assert result["alert"] is True
        assert result["level"] == "WARN"
        assert result["delta"] > 0.10

    def test_insufficient_data_one_prior_run(self, isolated_mlflow):
        _create_run(tier1_rate=0.60, hallucination_rate=0.05)
        current_id = _create_run(tier1_rate=0.40, hallucination_rate=0.05)
        result = check_tier1_drift(current_id)
        assert result["level"] == "INSUFFICIENT_DATA"
        assert result["alert"] is False

    def test_insufficient_data_no_prior_runs(self, isolated_mlflow):
        current_id = _create_run(tier1_rate=0.50, hallucination_rate=0.05)
        result = check_tier1_drift(current_id)
        assert result["level"] == "INSUFFICIENT_DATA"
        assert result["alert"] is False

    def test_missing_metric_in_current_run(self, isolated_mlflow):
        for _ in range(5):
            _create_run(tier1_rate=0.60, hallucination_rate=0.05)
        with mlflow.start_run() as run:
            mlflow.log_metrics({"hallucination_rate": 0.05})
        result = check_tier1_drift(run.info.run_id)
        assert result["alert"] is False


class TestPassRateDrift:
    def test_ok_within_range(self, isolated_mlflow):
        for _ in range(5):
            _create_run(tier1_rate=0.5, hallucination_rate=0.05, pass_rate=0.99)
        current_id = _create_run(tier1_rate=0.5, hallucination_rate=0.05, pass_rate=0.95)
        result = check_pass_rate_drift(current_id)
        assert result["alert"] is False
        assert result["level"] == "OK"

    def test_warn_on_drop(self, isolated_mlflow):
        for _ in range(5):
            _create_run(tier1_rate=0.5, hallucination_rate=0.05, pass_rate=0.99)
        # 14pp drop — exceeds 10pp threshold
        current_id = _create_run(tier1_rate=0.5, hallucination_rate=0.05, pass_rate=0.85)
        result = check_pass_rate_drift(current_id)
        assert result["alert"] is True
        assert result["level"] == "WARN"
        assert result["delta"] > 0.10

    def test_no_alert_on_improvement(self, isolated_mlflow):
        for _ in range(5):
            _create_run(tier1_rate=0.5, hallucination_rate=0.05, pass_rate=0.85)
        # Improvement — delta is negative, should not alert
        current_id = _create_run(tier1_rate=0.5, hallucination_rate=0.05, pass_rate=0.99)
        result = check_pass_rate_drift(current_id)
        assert result["alert"] is False

    def test_insufficient_data(self, isolated_mlflow):
        _create_run(tier1_rate=0.5, hallucination_rate=0.05, pass_rate=0.99)
        current_id = _create_run(tier1_rate=0.5, hallucination_rate=0.05, pass_rate=0.80)
        result = check_pass_rate_drift(current_id)
        assert result["level"] == "INSUFFICIENT_DATA"
        assert result["alert"] is False

    def test_missing_metric_returns_ok(self, isolated_mlflow):
        for _ in range(5):
            _create_run(tier1_rate=0.5, hallucination_rate=0.05, pass_rate=0.99)
        with mlflow.start_run() as run:
            mlflow.log_metrics({"tier1_rate": 0.5, "hallucination_rate": 0.05})
        result = check_pass_rate_drift(run.info.run_id)
        assert result["alert"] is False
        assert result["level"] == "OK"


class TestFieldNullRateDrift:
    _BASELINE = {
        "null_rate_tumor_grade": 0.10,
        "null_rate_primary_site": 0.05,
        "null_rate_histological_diagnosis": 0.02,
        "null_rate_histological_subtype": 0.20,
        "null_rate_specimen_type": 0.08,
        "null_rate_tnm_stage_T": 0.15,
        "null_rate_tnm_stage_N": 0.15,
        "null_rate_tnm_stage_M": 0.62,
        "null_rate_patient_sex": 0.30,
    }

    def test_ok_all_within_range(self, isolated_mlflow):
        for _ in range(5):
            _create_run_with_null_rates(self._BASELINE)
        # Small increase — under 15pp threshold
        current_rates = {**self._BASELINE, "null_rate_tumor_grade": 0.20}
        current_id = _create_run_with_null_rates(current_rates)
        result = check_field_null_rate_drift(current_id)
        assert result["alert"] is False
        assert result["level"] == "OK"

    def test_warn_single_field_drift(self, isolated_mlflow):
        for _ in range(5):
            _create_run_with_null_rates(self._BASELINE)
        # tumor_grade null rate jumps from 0.10 to 0.30 — 20pp, above threshold
        current_rates = {**self._BASELINE, "null_rate_tumor_grade": 0.30}
        current_id = _create_run_with_null_rates(current_rates)
        result = check_field_null_rate_drift(current_id)
        assert result["alert"] is True
        assert result["level"] == "WARN"
        assert any(a["field"] == "tumor_grade" for a in result["field_alerts"])

    def test_warn_multiple_fields(self, isolated_mlflow):
        for _ in range(5):
            _create_run_with_null_rates(self._BASELINE)
        current_rates = {
            **self._BASELINE,
            "null_rate_tumor_grade": 0.30,
            "null_rate_primary_site": 0.25,
        }
        current_id = _create_run_with_null_rates(current_rates)
        result = check_field_null_rate_drift(current_id)
        assert result["alert"] is True
        assert len(result["field_alerts"]) == 2

    def test_no_alert_on_improvement(self, isolated_mlflow):
        for _ in range(5):
            _create_run_with_null_rates(self._BASELINE)
        # Null rate drops (model extracting more) — should not alert
        current_rates = {**self._BASELINE, "null_rate_tumor_grade": 0.01}
        current_id = _create_run_with_null_rates(current_rates)
        result = check_field_null_rate_drift(current_id)
        assert result["alert"] is False

    def test_insufficient_data(self, isolated_mlflow):
        _create_run_with_null_rates(self._BASELINE)
        current_id = _create_run_with_null_rates(
            {**self._BASELINE, "null_rate_tumor_grade": 0.99}
        )
        result = check_field_null_rate_drift(current_id)
        assert result["level"] == "INSUFFICIENT_DATA"
        assert result["alert"] is False

    def test_no_null_rate_metrics_logged(self, isolated_mlflow):
        for _ in range(5):
            _create_run(tier1_rate=0.5, hallucination_rate=0.05)
        current_id = _create_run(tier1_rate=0.5, hallucination_rate=0.05)
        result = check_field_null_rate_drift(current_id)
        assert result["alert"] is False
        assert result["level"] == "OK"


def _create_run_with_usage(mean_output_tokens: float, mean_latency_s: float) -> str:
    with mlflow.start_run() as run:
        mlflow.log_metrics({
            "tier1_rate": 0.5, "hallucination_rate": 0.05, "pass_rate": 0.99,
            "mean_output_tokens": mean_output_tokens,
            "mean_latency_s": mean_latency_s,
        })
    return run.info.run_id


class TestTokenUsageDrift:
    def test_ok_within_range(self, isolated_mlflow):
        for _ in range(5):
            _create_run_with_usage(mean_output_tokens=400, mean_latency_s=2.0)
        current_id = _create_run_with_usage(mean_output_tokens=450, mean_latency_s=2.2)
        result = check_token_usage_drift(current_id)
        assert result["alert"] is False
        assert result["level"] == "OK"

    def test_warn_output_token_spike(self, isolated_mlflow):
        for _ in range(5):
            _create_run_with_usage(mean_output_tokens=400, mean_latency_s=2.0)
        # 700 tokens = 1.75x baseline — above 1.5x threshold
        current_id = _create_run_with_usage(mean_output_tokens=700, mean_latency_s=2.0)
        result = check_token_usage_drift(current_id)
        assert result["alert"] is True
        assert result["level"] == "WARN"
        assert any(a["metric"] == "mean_output_tokens" for a in result["metric_alerts"])

    def test_warn_latency_spike(self, isolated_mlflow):
        for _ in range(5):
            _create_run_with_usage(mean_output_tokens=400, mean_latency_s=2.0)
        # 5.0s = 2.5x baseline — above 2.0x threshold
        current_id = _create_run_with_usage(mean_output_tokens=400, mean_latency_s=5.0)
        result = check_token_usage_drift(current_id)
        assert result["alert"] is True
        assert any(a["metric"] == "mean_latency_s" for a in result["metric_alerts"])

    def test_no_alert_on_improvement(self, isolated_mlflow):
        for _ in range(5):
            _create_run_with_usage(mean_output_tokens=400, mean_latency_s=2.0)
        # Fewer tokens — should not alert
        current_id = _create_run_with_usage(mean_output_tokens=200, mean_latency_s=1.0)
        result = check_token_usage_drift(current_id)
        assert result["alert"] is False

    def test_insufficient_data(self, isolated_mlflow):
        _create_run_with_usage(mean_output_tokens=400, mean_latency_s=2.0)
        current_id = _create_run_with_usage(mean_output_tokens=9999, mean_latency_s=99.0)
        result = check_token_usage_drift(current_id)
        assert result["level"] == "INSUFFICIENT_DATA"
        assert result["alert"] is False

    def test_missing_metrics_returns_ok(self, isolated_mlflow):
        for _ in range(5):
            _create_run(tier1_rate=0.5, hallucination_rate=0.05)
        current_id = _create_run(tier1_rate=0.5, hallucination_rate=0.05)
        result = check_token_usage_drift(current_id)
        assert result["alert"] is False
        assert result["level"] == "OK"
