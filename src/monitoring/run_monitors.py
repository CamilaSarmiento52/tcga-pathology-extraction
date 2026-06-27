"""
CLI entry point that runs all active monitors for a given MLflow run.

Usage:
    uv run python -m src.monitoring.run_monitors <run_id> [--experiment NAME]

Exits with code 1 if any CRITICAL alert fires.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[2].resolve()
_DEFAULT_TRACKING_URI = f"sqlite:///{_PROJECT_ROOT / 'mlflow.db'}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pipeline quality monitors")
    p.add_argument("run_id", help="MLflow run ID to evaluate")
    p.add_argument(
        "--experiment",
        default="pathology-extraction",
        help="MLflow experiment name (default: pathology-extraction)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI)
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)

    from src.monitoring.tier1_drift import check_tier1_drift
    from src.monitoring.hallucination_alert import check_hallucination_rate

    print(f"\nRunning monitors for run: {args.run_id}")
    print(f"Experiment: {args.experiment}")
    print("=" * 60)

    has_critical = False

    # Monitor 1: Tier 1 drift
    drift_result = check_tier1_drift(args.run_id, experiment_name=args.experiment)
    prefix = "⚠ WARN" if drift_result["alert"] else "✓ OK  "
    print(f"[Monitor 1 — Tier 1 Drift]  {prefix}: {drift_result['message']}")

    # Monitor 3: Hallucination rate
    hall_result = check_hallucination_rate(args.run_id, experiment_name=args.experiment)
    if hall_result["level"] == "CRITICAL":
        has_critical = True
        prefix = "✗ CRITICAL"
    else:
        prefix = "✓ OK      "
    print(f"[Monitor 3 — Hallucination] {prefix}: {hall_result['message']}")

    print("=" * 60)
    if has_critical:
        print("RESULT: CRITICAL alert(s) fired. Action required.")
        sys.exit(1)
    else:
        print("RESULT: All monitors passed.")


if __name__ == "__main__":
    main()
