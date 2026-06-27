"""
Build a model comparison scoring matrix from evaluation results.

PRICING and GDPR_COMPLIANT are module-level constants — update them if
OpenAI pricing changes or a new model is added.
"""

from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import (
    compute_aggregate_metrics,
    compute_all_field_metrics,
    compute_combined_score,
    compute_semantic_metrics,
)

# USD per 1M tokens (OpenAI pricing as of 2026-05)
PRICING_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "openai:gpt-4o": {"input": 2.50, "output": 10.00},
    "openai:o4-mini": {"input": 1.10, "output": 4.40},
    "ollama:llama3.2": {"input": 0.00, "output": 0.00},  # local, no API cost
}

GDPR_COMPLIANT: dict[str, bool] = {
    "openai:gpt-4o": False,        # data leaves local environment
    "openai:o4-mini": False,
    "ollama:llama3.2": True,       # runs fully locally
    "ollama:mistral-nemo": True,
    "ollama:openbiollm": True,
}


def _resolve_model_key(model: str) -> str:
    """Return the model key as stored in PRICING/GDPR dicts, or the raw string."""
    return model if model in PRICING_PER_1M_TOKENS else model


def compute_cost_per_record(records: list[dict], model: str) -> float:
    """Average USD cost per record based on token counts in results."""
    key = _resolve_model_key(model)
    pricing = PRICING_PER_1M_TOKENS.get(key, {"input": 0.0, "output": 0.0})
    if not records:
        return 0.0
    total = sum(
        r.get("input_tokens", 0) / 1_000_000 * pricing["input"]
        + r.get("output_tokens", 0) / 1_000_000 * pricing["output"]
        for r in records
    )
    return total / len(records)


def build_comparison_matrix(
    model_results: dict[str, list[dict]],
    ground_truth: list[dict],
) -> pd.DataFrame:
    """Build a scoring matrix with one row per model.

    Columns: model, combined_score, weighted_f1, mean_similarity,
             mean_similarity_complete, hallucination_rate, avg_cost_per_record,
             gdpr_compliant, avg_latency_ms
    """
    rows = []
    for model, preds in model_results.items():
        agg = compute_aggregate_metrics(preds, ground_truth)
        sem = compute_semantic_metrics(preds, ground_truth)
        field_metrics = compute_all_field_metrics(preds, ground_truth)
        mean_sim = sum(s.mean_similarity for s in sem) / len(sem) if sem else 0.0
        mean_sim_complete = (
            sum(s.mean_similarity_complete for s in sem) / len(sem) if sem else 0.0
        )
        combined = compute_combined_score(field_metrics, sem)
        key = _resolve_model_key(model)
        rows.append(
            {
                "model": model,
                "combined_score": round(combined, 4),
                "weighted_f1": round(agg.weighted_f1, 4),
                "mean_similarity": round(mean_sim, 4),
                "mean_similarity_complete": round(mean_sim_complete, 4),
                "hallucination_rate": round(agg.hallucination_rate, 4),
                "avg_cost_per_record": round(compute_cost_per_record(preds, model), 6),
                "gdpr_compliant": GDPR_COMPLIANT.get(key, False),
                "avg_latency_ms": round(
                    sum(r.get("latency_ms", 0) for r in preds) / len(preds)
                    if preds
                    else 0.0,
                    1,
                ),
            }
        )
    return pd.DataFrame(rows)
