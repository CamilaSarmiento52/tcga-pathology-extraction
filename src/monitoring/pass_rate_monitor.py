"""
Monitor 2: Pass rate drift detection.

Compares the current run's pass_rate against the rolling mean of the last
N completed runs. Fires a WARN alert if it drops more than 10 percentage points
below baseline — the earliest structural signal that the model or schema broke.
"""

from __future__ import annotations

import mlflow
import pandas as pd


_DRIFT_THRESHOLD = 0.10  # 10 percentage points


def check_pass_rate_drift(
    run_id: str,
    experiment_name: str = "pathology-extraction",
    n_baseline: int = 5,
) -> dict:
    """Compare current run's pass_rate against rolling baseline.

    Returns a dict with keys: alert (bool), level (str), message (str), delta (float).
    Returns level="INSUFFICIENT_DATA" when fewer than 2 prior runs exist.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return {
            "alert": False,
            "level": "OK",
            "message": f"Experiment '{experiment_name}' not found.",
            "delta": 0.0,
        }

    current_run = client.get_run(run_id)
    current_rate = current_run.data.metrics.get("pass_rate")
    if current_rate is None:
        return {
            "alert": False,
            "level": "OK",
            "message": "pass_rate not logged in current run — skipping drift check.",
            "delta": 0.0,
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
            "delta": 0.0,
        }

    col = "metrics.pass_rate"
    if col not in runs_df.columns:
        return {
            "alert": False,
            "level": "OK",
            "message": "Prior runs do not have pass_rate logged.",
            "delta": 0.0,
        }

    baseline_mean = runs_df[col].dropna().mean()
    delta = baseline_mean - current_rate  # negative delta = improvement; we only warn on drops

    if delta > _DRIFT_THRESHOLD:
        return {
            "alert": True,
            "level": "WARN",
            "message": (
                f"Pass rate dropped {delta:.1%} below baseline mean "
                f"({baseline_mean:.1%} → {current_rate:.1%})."
            ),
            "delta": round(delta, 4),
        }

    return {
        "alert": False,
        "level": "OK",
        "message": (
            f"Pass rate within range: {current_rate:.1%} "
            f"(baseline mean {baseline_mean:.1%}, delta {delta:.1%})."
        ),
        "delta": round(delta, 4),
    }
