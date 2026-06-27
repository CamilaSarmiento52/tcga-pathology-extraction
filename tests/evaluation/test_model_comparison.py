"""Tests for src/evaluation/model_comparison.py."""

import pytest
import pandas as pd

from src.evaluation.model_comparison import (
    GDPR_COMPLIANT,
    PRICING_PER_1M_TOKENS,
    build_comparison_matrix,
    compute_cost_per_record,
)


def _pred(report_id, model="openai:o4-mini", input_tokens=1000, output_tokens=500,
          latency_ms=2000.0, validation_status="valid", hallucination_flags=None, **fields):
    base = {
        "report_id": report_id,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "validation_status": validation_status,
        "hallucination_flags": hallucination_flags,
        "primary_site": "breast",
        "histological_diagnosis": "IDC",
        "histological_subtype": "ductal",
        "tumor_grade": "Grade 2",
        "tnm_stage": {"T": "pT2", "N": "pN1", "M": "pM0"},
        "specimen_type": "mastectomy",
    }
    base.update(fields)
    return base


def _gold(report_id, cancer_type="BRCA"):
    return {
        "report_id": report_id,
        "cancer_type": cancer_type,
        "primary_site": "breast",
        "histological_diagnosis": "IDC",
        "histological_subtype": "ductal",
        "tumor_grade": "Grade 2",
        "tnm_stage": {"T": "pT2", "N": "pN1", "M": "pM0"},
        "specimen_type": "mastectomy",
    }


class TestPricingConstants:
    def test_known_models_present(self):
        assert "openai:o4-mini" in PRICING_PER_1M_TOKENS
        assert "ollama:llama3.2" in PRICING_PER_1M_TOKENS

    def test_ollama_zero_cost(self):
        assert PRICING_PER_1M_TOKENS["ollama:llama3.2"]["input"] == 0.0
        assert PRICING_PER_1M_TOKENS["ollama:llama3.2"]["output"] == 0.0

    def test_gdpr_flags(self):
        assert GDPR_COMPLIANT["ollama:llama3.2"] is True
        assert GDPR_COMPLIANT["openai:o4-mini"] is False


class TestComputeCostPerRecord:
    def test_zero_cost_for_ollama(self):
        preds = [_pred("R1", model="ollama:llama3.2", input_tokens=1000, output_tokens=500)]
        assert compute_cost_per_record(preds, "ollama:llama3.2") == pytest.approx(0.0)

    def test_nonzero_cost_for_api_model(self):
        preds = [_pred("R1", model="openai:o4-mini", input_tokens=1_000_000, output_tokens=0)]
        cost = compute_cost_per_record(preds, "openai:o4-mini")
        assert cost == pytest.approx(1.10)  # $1.10 per 1M input tokens

    def test_empty_preds_returns_zero(self):
        assert compute_cost_per_record([], "openai:o4-mini") == pytest.approx(0.0)

    def test_unknown_model_returns_zero_cost(self):
        preds = [_pred("R1", model="unknown:model", input_tokens=5000, output_tokens=2000)]
        cost = compute_cost_per_record(preds, "unknown:model")
        assert cost == pytest.approx(0.0)


class TestBuildComparisonMatrix:
    def test_matrix_shape(self):
        golds = [_gold("R1"), _gold("R2")]
        model_results = {
            "openai:o4-mini": [_pred("R1"), _pred("R2")],
            "ollama:llama3.2": [_pred("R1", model="ollama:llama3.2"), _pred("R2", model="ollama:llama3.2")],
        }
        df = build_comparison_matrix(model_results, golds)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert set(df.columns) >= {
            "model", "weighted_f1", "hallucination_rate",
            "avg_cost_per_record", "gdpr_compliant", "avg_latency_ms",
        }

    def test_no_nan_values(self):
        golds = [_gold("R1")]
        model_results = {"openai:o4-mini": [_pred("R1")]}
        df = build_comparison_matrix(model_results, golds)
        assert not df.isnull().any().any()

    def test_gdpr_flag_in_matrix(self):
        golds = [_gold("R1")]
        model_results = {
            "openai:o4-mini": [_pred("R1")],
            "ollama:llama3.2": [_pred("R1", model="ollama:llama3.2")],
        }
        df = build_comparison_matrix(model_results, golds).set_index("model")
        assert df.loc["ollama:llama3.2", "gdpr_compliant"] == True
        assert df.loc["openai:o4-mini", "gdpr_compliant"] == False

    def test_empty_model_results(self):
        df = build_comparison_matrix({}, [_gold("R1")])
        assert len(df) == 0
