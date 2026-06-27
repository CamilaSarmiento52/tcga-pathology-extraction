"""
Monitor 5: Token usage and latency drift detection.

Compares the current run's mean_output_tokens and mean_latency_s against the
rolling mean of the last N completed runs. A spike in output tokens usually means
the model is generating verbose preamble before the JSON — an early warning before
quality metrics surface the problem.

Thresholds:
  mean_output_tokens: WARN if current > 1.5x baseline mean (50% above)
  mean_latency_s:     WARN if current > 2.0x baseline mean (100% above)
"""

from __future__ import annotations

import mlflow
import pandas as pd


_OUTPUT_TOKEN_MULTIPLIER = 1.5   # 50% above baseline
_LATENCY_MULTIPLIER = 2.0        # 100% above baseline (latency is noisier)

_CHECKS = {
    "mean_output_tokens": _OUTPUT_TOKEN_MULTIPLIER,
    "mean_latency_s": _LATENCY_MULTIPLIER,
}


def check_token_usage_drift(
    run_id: str,
    experiment_name: str = "pathology-extraction",
    n_baseline: int = 5,
) -> dict:
    """Compare current run's per-record token and latency metrics against baseline.

    Returns a dict with keys:
      alert (bool), level (str), message (str),
      metric_alerts (list[dict]) — one entry per metric that drifted.
    Returns level="INSUFFICIENT_DATA" when fewer than 2 prior runs exist.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return {
            "alert": False,
            "level": "OK",
            "message": f"Experiment '{experiment_name}' not found.",
            "metric_alerts": [],
        }

    current_run = client.get_run(run_id)
    current_metrics = current_run.data.metrics

    logged = [m for m in _CHECKS if m in current_metrics]
    if not logged:
        return {
            "alert": False,
            "level": "OK",
            "message": "No token/latency metrics logged in current run — skipping.",
            "metric_alerts": [],
        }

    runs_df: pd.DataFrame = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"attributes.status = 'FINISHED' and attributes.run_id != '{run_id}'",
        order_by=["attributes.end_time DESC"],
        max_results=n_baseline,
    )

    if runs_df.empty or len(runs_df) < 2:
        return {
            "alert": False,
            "level": "INSUFFICIENT_DATA",
            "message": f"Only {len(runs_df)} prior run(s) found — need ≥2 for drift baseline.",
            "metric_alerts": [],
        }

    metric_alerts = []
    for metric, multiplier in _CHECKS.items():
        current_val = current_metrics.get(metric)
        if current_val is None:
            continue
        col = f"metrics.{metric}"
        if col not in runs_df.columns:
            continue
        baseline_mean = runs_df[col].dropna().mean()
        if baseline_mean <= 0:
            continue
        if current_val > baseline_mean * multiplier:
            metric_alerts.append({
                "metric": metric,
                "current": round(current_val, 2),
                "baseline": round(baseline_mean, 2),
                "ratio": round(current_val / baseline_mean, 2),
            })

    if metric_alerts:
        parts = ", ".join(
            f"{a['metric']} ({a['ratio']:.1f}x baseline)" for a in metric_alerts
        )
        return {
            "alert": True,
            "level": "WARN",
            "message": f"Usage spike detected: {parts}.",
            "metric_alerts": metric_alerts,
        }

    return {
        "alert": False,
        "level": "OK",
        "message": f"Token usage and latency within range ({len(logged)} metric(s) checked).",
        "metric_alerts": [],
    }
