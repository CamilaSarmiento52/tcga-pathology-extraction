"""
Monitor 1: Tier 1 flagging rate drift detection.

Compares the current run's tier1_rate against the rolling mean of the last
N completed runs in the same experiment. Fires a WARN alert if the absolute
difference exceeds 10 percentage points.
"""

from __future__ import annotations

import mlflow
import pandas as pd


_DRIFT_THRESHOLD = 0.10  # 10 percentage points


def check_tier1_drift(
    run_id: str,
    experiment_name: str = "pathology-extraction",
    n_baseline: int = 5,
) -> dict:
    """Compare current run's tier1_rate against rolling baseline.

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

    # Fetch current run metric
    current_run = client.get_run(run_id)
    current_rate = current_run.data.metrics.get("tier1_rate")
    if current_rate is None:
        return {
            "alert": False,
            "level": "OK",
            "message": "tier1_rate not logged in current run — skipping drift check.",
            "delta": 0.0,
        }

    # Fetch prior completed runs (exclude current)
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

    col = "metrics.tier1_rate"
    if col not in runs_df.columns:
        return {
            "alert": False,
            "level": "OK",
            "message": "Prior runs do not have tier1_rate logged.",
            "delta": 0.0,
        }

    baseline_mean = runs_df[col].dropna().mean()
    delta = abs(current_rate - baseline_mean)

    if delta > _DRIFT_THRESHOLD:
        return {
            "alert": True,
            "level": "WARN",
            "message": (
                f"Tier 1 rate drifted {delta:.1%} from baseline mean "
                f"({baseline_mean:.1%} → {current_rate:.1%})."
            ),
            "delta": round(delta, 4),
        }

    return {
        "alert": False,
        "level": "OK",
        "message": (
            f"Tier 1 rate within range: {current_rate:.1%} "
            f"(baseline mean {baseline_mean:.1%}, delta {delta:.1%})."
        ),
        "delta": round(delta, 4),
    }
