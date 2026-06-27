"""
Monitor 3: Hallucination rate critical threshold alert.

Checks the hallucination_rate metric of a completed run. If it exceeds 8%,
fires a CRITICAL alert and sets should_pause=True.
"""

from __future__ import annotations

import mlflow


_CRITICAL_THRESHOLD = 0.08  # 8%


def check_hallucination_rate(
    run_id: str,
) -> dict:
    """Check hallucination_rate for a single run against the critical threshold.

    Returns a dict with keys: alert (bool), level (str), should_pause (bool),
    message (str), rate (float).
    """
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    rate = run.data.metrics.get("hallucination_rate")

    if rate is None:
        return {
            "alert": False,
            "level": "OK",
            "should_pause": False,
            "message": "hallucination_rate not logged in this run — skipping check.",
            "rate": 0.0,
        }

    if rate > _CRITICAL_THRESHOLD:
        return {
            "alert": True,
            "level": "CRITICAL",
            "should_pause": True,
            "message": (
                f"Hallucination rate {rate:.1%} exceeds critical threshold "
                f"({_CRITICAL_THRESHOLD:.0%}). Pipeline should be paused."
            ),
            "rate": round(rate, 4),
        }

    return {
        "alert": False,
        "level": "OK",
        "should_pause": False,
        "message": f"Hallucination rate {rate:.1%} is within acceptable range (≤{_CRITICAL_THRESHOLD:.0%}).",
        "rate": round(rate, 4),
    }
