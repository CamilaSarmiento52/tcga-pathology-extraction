"""
Monitor 4: Per-field null rate drift detection.

For each tracked clinical field, compares the current run's null rate against
the rolling mean of the last N completed runs. Fires a WARN alert for any field
whose null rate has increased more than 15 percentage points above baseline —
indicating the model has stopped extracting that field.
"""

from __future__ import annotations

import mlflow
import pandas as pd

from src.evaluation.metrics import _NULL_RATE_FIELDS


_DRIFT_THRESHOLD = 0.15  # 15 percentage points

# MLflow metric names derived from the same field list used in metrics.py
_METRIC_NAMES: tuple[str, ...] = tuple(
    f"null_rate_{f.replace('.', '_')}" for f in _NULL_RATE_FIELDS
)


def check_field_null_rate_drift(
    run_id: str,
    experiment_name: str = "pathology-extraction",
    n_baseline: int = 5,
) -> dict:
    """Compare each field's null rate against rolling baseline.

    Returns a dict with keys:
      alert (bool), level (str), message (str),
      field_alerts (list[dict]) — one entry per drifted field.
    Returns level="INSUFFICIENT_DATA" when fewer than 2 prior runs exist.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return {
            "alert": False,
            "level": "OK",
            "message": f"Experiment '{experiment_name}' not found.",
            "field_alerts": [],
        }

    current_run = client.get_run(run_id)
    current_metrics = current_run.data.metrics

    # Check at least one null rate metric is logged in this run
    logged = [m for m in _METRIC_NAMES if m in current_metrics]
    if not logged:
        return {
            "alert": False,
            "level": "OK",
            "message": "No null_rate_* metrics logged in current run — skipping.",
            "field_alerts": [],
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
            "field_alerts": [],
        }

    field_alerts = []
    for metric in _METRIC_NAMES:
        current_rate = current_metrics.get(metric)
        if current_rate is None:
            continue
        col = f"metrics.{metric}"
        if col not in runs_df.columns:
            continue
        baseline_mean = runs_df[col].dropna().mean()
        delta = current_rate - baseline_mean  # positive = more nulls = worse
        if delta > _DRIFT_THRESHOLD:
            field = metric[len("null_rate_"):]  # strip prefix for display
            field_alerts.append({
                "field": field,
                "current": round(current_rate, 4),
                "baseline": round(baseline_mean, 4),
                "delta": round(delta, 4),
            })

    if field_alerts:
        fields_str = ", ".join(
            f"{a['field']} (+{a['delta']:.0%})" for a in field_alerts
        )
        return {
            "alert": True,
            "level": "WARN",
            "message": f"Null rate drift in: {fields_str}.",
            "field_alerts": field_alerts,
        }

    return {
        "alert": False,
        "level": "OK",
        "message": f"All {len(logged)} field null rates within range.",
        "field_alerts": [],
    }
