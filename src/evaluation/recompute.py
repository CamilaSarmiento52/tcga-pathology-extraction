"""
Recompute all evaluation metrics from a saved eval_results_*.jsonl without
calling any model.  Useful for back-filling new metrics onto existing runs and
for validating that metric code is deterministic.

Usage:
    uv run python -m src.evaluation.recompute \
        --results data/processed/eval_results_o4mini_0shot.jsonl \
        --eval-set data/annotations/eval_set_v2.jsonl

Add --mlflow-run-id <RUN_ID> to compare against a previously logged MLflow run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[2].resolve()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recompute metrics from saved JSONL")
    p.add_argument("--results", required=True, help="Path to eval_results_*.jsonl")
    p.add_argument("--eval-set", required=True, help="Path to gold eval_set_*.jsonl")
    p.add_argument(
        "--mlflow-run-id",
        default=None,
        help="Optional: compare output against this MLflow run's logged metrics",
    )
    p.add_argument(
        "--no-semantic",
        action="store_true",
        help="Skip semantic similarity (faster; combined_score will not be shown)",
    )
    return p.parse_args()


def _load(path: str) -> list[dict]:
    from src.pipeline.loader import load_records
    records = list(load_records(Path(path)))
    if not records:
        print(f"ERROR: no records loaded from {path}", file=sys.stderr)
        sys.exit(1)
    return records


def main() -> None:
    args = _parse_args()

    preds = _load(args.results)
    golds = _load(args.eval_set)

    from src.evaluation.metrics import (
        compute_aggregate_metrics,
        compute_all_field_metrics,
        compute_combined_score,
        compute_semantic_metrics,
    )

    agg = compute_aggregate_metrics(preds, golds)
    field_metrics = compute_all_field_metrics(preds, golds)

    print(f"\nRecomputed metrics")
    print(f"  Results file : {args.results}")
    print(f"  Eval set     : {args.eval_set}")
    print(f"  Predictions  : {len(preds)}  |  Gold records: {len(golds)}")
    print("=" * 60)

    print(f"\n  weighted_f1       : {agg.weighted_f1:.4f}")
    print(f"  mean_f1           : {agg.mean_f1:.4f}")
    print(f"  hallucination_rate: {agg.hallucination_rate:.4f}")

    print(f"\n  Per-field F1:")
    for fm in field_metrics:
        print(f"    {fm.field:<30} f1={fm.f1:.4f}")

    combined_score: float | None = None
    if not args.no_semantic:
        try:
            semantic_metrics = compute_semantic_metrics(preds, golds)
            scored = [sm for sm in semantic_metrics if sm.n_scored_pairs > 0]
            mean_sim = sum(sm.mean_similarity for sm in scored) / len(scored) if scored else 0.0
            mean_sim_complete = (
                sum(sm.mean_similarity_complete for sm in semantic_metrics)
                / len(semantic_metrics)
                if semantic_metrics else 0.0
            )
            combined_score = compute_combined_score(field_metrics, semantic_metrics)

            print(f"\n  mean_similarity          : {mean_sim:.4f}")
            print(f"  mean_similarity_complete : {mean_sim_complete:.4f}")
            print(f"\n  Per-field similarity (completeness-aware):")
            for sm in semantic_metrics:
                print(
                    f"    {sm.field:<30} sim_complete={sm.mean_similarity_complete:.4f}"
                    f"  (omit={sm.n_omission}, hall={sm.n_hallucination},"
                    f" both_null={sm.n_both_null}, present={sm.n_both_present})"
                )
            print(f"\n  combined_score           : {combined_score:.4f}")
        except Exception as exc:
            print(f"\n  (semantic metrics skipped: {exc})")

    # --- optional MLflow comparison ---
    if args.mlflow_run_id:
        import os
        import mlflow

        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            f"sqlite:///{_PROJECT_ROOT / 'mlflow.db'}",
        )
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        logged = client.get_run(args.mlflow_run_id).data.metrics

        compare_keys = [
            ("weighted_f1",            agg.weighted_f1),
            ("mean_f1",                agg.mean_f1),
            ("hallucination_rate",     agg.hallucination_rate),
        ]
        if combined_score is not None:
            compare_keys.append(("combined_score", combined_score))

        print(f"\n{'─' * 60}")
        print(f"Comparison vs MLflow run {args.mlflow_run_id}")
        print(f"  {'metric':<30} {'recomputed':>12} {'logged':>12} {'match':>8}")
        print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*8}")
        all_match = True
        for key, recomputed in compare_keys:
            logged_val = logged.get(key)
            if logged_val is None:
                print(f"  {key:<30} {recomputed:>12.4f} {'(not logged)':>12}  {'?':>6}")
                continue
            match = abs(recomputed - logged_val) < 1e-3
            all_match = all_match and match
            marker = "✓" if match else "✗ MISMATCH"
            print(f"  {key:<30} {recomputed:>12.4f} {logged_val:>12.4f}  {marker:>8}")

        print("=" * 60)
        if all_match:
            print("RESULT: All metrics match the MLflow run.")
        else:
            print("RESULT: Mismatch(es) found — see rows marked ✗ above.")
            sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("(Pass --mlflow-run-id <RUN_ID> to compare against a logged run.)")


if __name__ == "__main__":
    main()
