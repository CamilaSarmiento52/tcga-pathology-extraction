import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parents[2] / ".env")

from src.evaluation.hitl_thresholds import tier_distribution
from src.evaluation.metrics import (
    compute_aggregate_metrics,
    compute_all_field_metrics,
    compute_field_null_rates,
    compute_m_null_rate,
)
from src.evaluation.model_comparison import PRICING_PER_1M_TOKENS
from src.pipeline.loader import load_dev_subset, load_records
from src.pipeline.mlflow_tracker import MLflowTracker
from src.pipeline.model_caller import call_model
from src.pipeline.prompt_constructor import MAX_REPORT_TOKENS, build_prompt, count_tokens
from src.pipeline.result_writer import build_output_record, write_result, write_summary
from src.pipeline.validator import VALIDATOR_VERSION, check_vocab, parse_json, validate_record, validate_schema
from src.schema import SCHEMA_VERSION

_PROMPT_DIR = Path(__file__).parent.parent / "prompts"
_DEFAULT_EVAL_SET = str(Path(__file__).parents[2] / "data" / "annotations" / "eval_set_v2.jsonl")
_DEFAULT_MODEL = "openai:gpt-4o"


def _prompt_version_from_path(path: Path) -> str:
    """Extract version string from a prompt template filename.

    Expects filenames like ``extraction_v1.5.txt`` → ``"v1.5"``.
    Falls back to the full stem if the pattern does not match.
    """
    match = re.search(r"_(v\d+\.\d+)$", path.stem)
    return match.group(1) if match else path.stem


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pathology extraction pipeline")
    p.add_argument("--input", required=True, help="Path to input JSONL corpus")
    p.add_argument("--output", required=True, help="Path to output JSONL results file")
    p.add_argument("--model", default=_DEFAULT_MODEL, help="provider:model string (e.g. openai:o4-mini, ollama:mistral:7b)")
    p.add_argument(
        "--prompt-file",
        default=str(_PROMPT_DIR / "extraction_v1.9.txt"),
        help="Path to prompt template file (e.g. src/prompts/extraction_v1.9.txt). Version is derived from the filename.",
    )
    p.add_argument("--few-shot", default=None, help="Path to few-shot examples JSONL")
    p.add_argument("--n-records", type=int, default=None, help="Dev mode: limit to N records")
    p.add_argument("--seed", type=int, default=42, help="Random seed for dev subset sampling")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum output tokens the model may generate per record (default 8192).",
    )
    p.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="Context window for local Ollama models (e.g. 16384). Ignored for OpenAI. "
        "Unset = Ollama default (4096), which silently truncates long prompts.",
    )
    p.add_argument("--debug", action="store_true", help="Save full raw_text for failed records")
    p.add_argument("--verbose", action="store_true", help="Print raw model response to stderr for each record")
    p.add_argument(
        "--eval-set",
        default=_DEFAULT_EVAL_SET,
        help="Path to ground truth eval set JSONL for computing F1/hallucination metrics (default: eval_set_v2.jsonl)",
    )
    p.add_argument(
        "--no-semantic",
        action="store_true",
        help="Skip semantic similarity metrics (no torch/transformers required)",
    )
    return p.parse_args()


def _run_monitors(run_id: str | None) -> None:
    if run_id is None:
        return
    from src.monitoring.field_null_rate_monitor import check_field_null_rate_drift
    from src.monitoring.hallucination_alert import check_hallucination_rate
    from src.monitoring.pass_rate_monitor import check_pass_rate_drift
    from src.monitoring.tier1_drift import check_tier1_drift
    from src.monitoring.token_usage_monitor import check_token_usage_drift

    print(f"\n{'=' * 50}")
    print("Monitor results")
    print(f"{'=' * 50}")

    pass_r = check_pass_rate_drift(run_id)
    prefix = "WARN" if pass_r["alert"] else "OK  "
    print(f"  Pass rate:         [{prefix}] {pass_r['message']}")

    drift = check_tier1_drift(run_id)
    prefix = "WARN" if drift["alert"] else "OK  "
    print(f"  Tier 1 drift:      [{prefix}] {drift['message']}")

    hall = check_hallucination_rate(run_id)
    prefix = "CRITICAL" if hall["level"] == "CRITICAL" else "OK      "
    print(f"  Hallucination:     [{prefix}] {hall['message']}")

    null_drift = check_field_null_rate_drift(run_id)
    prefix = "WARN" if null_drift["alert"] else "OK  "
    print(f"  Field null rates:  [{prefix}] {null_drift['message']}")

    token_drift = check_token_usage_drift(run_id)
    prefix = "WARN" if token_drift["alert"] else "OK  "
    print(f"  Token usage:       [{prefix}] {token_drift['message']}")

    print(f"{'=' * 50}\n")

    if hall["level"] == "CRITICAL":
        print("CRITICAL alert fired — review pipeline outputs before proceeding.", file=sys.stderr)
        sys.exit(1)


def _print_delta_summary(run_id: str | None) -> None:
    """Print a one-line delta for key metrics vs the most recent prior run."""
    if run_id is None:
        return
    import mlflow as _mlflow

    client = _mlflow.tracking.MlflowClient()
    current = client.get_run(run_id).data.metrics

    runs_df = _mlflow.search_runs(
        experiment_names=["pathology-extraction"],
        filter_string=f"attributes.status = 'FINISHED' and attributes.run_id != '{run_id}'",
        order_by=["attributes.end_time DESC"],
        max_results=1,
    )
    if runs_df.empty:
        return  # no prior run to compare against

    prev = {
        col[len("metrics."):]: runs_df.iloc[0][col]
        for col in runs_df.columns
        if col.startswith("metrics.") and not _pd_isna(runs_df.iloc[0][col])
    }

    # (metric_key, display_label, is_rate, higher_is_better)
    _METRICS = [
        ("pass_rate",           "Pass rate",          True,  True),
        ("hallucination_rate",  "Hallucination",      True,  False),
        ("weighted_f1",         "Weighted F1",        False, True),
        ("mean_output_tokens",  "Mean output tokens", False, False),
    ]

    rows = []
    for key, label, is_rate, higher_better in _METRICS:
        cur = current.get(key)
        prv = prev.get(key)
        if cur is None:
            continue
        if prv is None:
            if is_rate:
                rows.append((label, f"{cur:.1%}", "  —"))
            else:
                rows.append((label, f"{cur:.4g}", "  —"))
            continue
        delta = cur - prv
        if is_rate:
            sign = "+" if delta >= 0 else ""
            arrow = "↑" if (delta > 0) == higher_better else ("↓" if delta != 0 else " ")
            rows.append((label, f"{cur:.1%}", f"{arrow} {sign}{delta*100:+.1f}pp"))
        else:
            sign = "+" if delta >= 0 else ""
            arrow = "↑" if (delta > 0) == higher_better else ("↓" if delta != 0 else " ")
            rows.append((label, f"{cur:.4g}", f"{arrow} {sign}{delta:+.4g}"))

    if not rows:
        return

    print(f"{'─' * 50}")
    print("Delta vs previous run")
    print(f"{'─' * 50}")
    label_w = max(len(r[0]) for r in rows)
    for label, val, delta in rows:
        print(f"  {label:<{label_w}}  {val:>8}   {delta}")
    print(f"{'─' * 50}\n")


def _pd_isna(val) -> bool:
    try:
        import math
        return val is None or (isinstance(val, float) and math.isnan(val))
    except Exception:
        return False


def _compute_total_cost(records: list[dict], model: str) -> float:
    pricing = PRICING_PER_1M_TOKENS.get(model, {"input": 0.0, "output": 0.0})
    return sum(
        r.get("input_tokens", 0) / 1_000_000 * pricing["input"]
        + r.get("output_tokens", 0) / 1_000_000 * pricing["output"]
        for r in records
    )


def main() -> None:
    args = _parse_args()

    template_path = Path(args.prompt_file)
    if not template_path.exists():
        print(f"ERROR: prompt template not found: {template_path}", file=sys.stderr)
        sys.exit(1)

    # Version is always derived from the filename — never manually typed.
    prompt_version = _prompt_version_from_path(template_path)

    run_id = f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"

    if args.n_records:
        records = load_dev_subset(args.input, n=args.n_records, seed=args.seed)
    else:
        records = list(load_records(args.input))

    tracker_params = {
        "model": args.model,
        "prompt_version": prompt_version,
        "few_shot_path": args.few_shot or "",
        "schema_version": SCHEMA_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "record_count": len(records),
        "num_ctx": args.num_ctx,
        "max_tokens": args.max_tokens,
    }
    if args.eval_set and not args.no_semantic:
        # Constants only — importing semantic does not import torch
        from src.evaluation.semantic import EMBEDDING_MODEL, SEMANTIC_VERSION

        tracker_params["embedding_model"] = EMBEDDING_MODEL
        tracker_params["semantic_version"] = SEMANTIC_VERSION

    _mlflow_run_id: str | None = None
    with MLflowTracker(params=tracker_params) as tracker:
        tracker.log_prompt(template_path, prompt_version)
        stats: dict = {
            "run_id": run_id,
            "model": args.model,
            "prompt_version": prompt_version,
            "few_shot_path": args.few_shot,
            "num_ctx": args.num_ctx,
            "max_tokens": args.max_tokens,
            "total": len(records),
            "valid": 0,
            "json_failed": 0,
            "schema_failed": 0,
            "vocab_flagged": 0,
            "truncated": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_latency_ms": 0.0,
        }

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Resume: deduplicate output file and skip already-processed records
        done_ids: set[str] = set()
        if out_path.exists():
            seen: dict[str, str] = {}
            with open(out_path, encoding="utf-8") as _f:
                for _line in _f:
                    try:
                        _rid = json.loads(_line)["report_id"]
                        if _rid not in seen:
                            seen[_rid] = _line
                    except Exception:
                        pass
            # Rewrite file deduplicated
            out_path.write_text("\n".join(v.strip() for v in seen.values()) + "\n", encoding="utf-8")
            done_ids = set(seen.keys())
            if done_ids:
                print(f"Resuming — skipping {len(done_ids)} already-processed records.")
        records = [r for r in records if r["report_id"] not in done_ids]
        stats["total"] = len(records) + len(done_ids)

        for record in tqdm(records, desc="Extracting"):
            prompt, was_truncated = build_prompt(record, template_path, args.few_shot)
            if was_truncated:
                logger.warning(
                    "Report truncated to %d tokens before sending to model: report_id=%s",
                    MAX_REPORT_TOKENS,
                    record["report_id"],
                )
            # Pre-flight: local models silently truncate prompts that exceed their
            # context window (Ollama default 4096). Warn before the call so this is
            # never silent. count_tokens uses cl100k as an approximation.
            if not args.model.startswith("openai:"):
                effective_ctx = args.num_ctx if args.num_ctx is not None else 4096
                prompt_tokens = count_tokens(prompt)
                if prompt_tokens > effective_ctx:
                    logger.warning(
                        "Prompt (~%d tokens) exceeds num_ctx=%d for local model — it will be "
                        "truncated from the front (instructions/few-shot lost): report_id=%s. "
                        "Pass --num-ctx >= %d to fix.",
                        prompt_tokens,
                        effective_ctx,
                        record["report_id"],
                        prompt_tokens,
                    )
            with mlflow.start_span(name="extract_record", span_type="CHAIN") as span:
                span.set_attribute("report_id", record["report_id"])
                span.set_attribute("cancer_type", record["cancer_type"])
                span.set_attribute("was_truncated", was_truncated)
                response = call_model(prompt, model=args.model, max_tokens=args.max_tokens, num_ctx=args.num_ctx)
            if args.verbose:
                tqdm.write(
                    f"\n[verbose] {record['report_id']}\n{response.raw_text}\n",
                    file=sys.stderr,
                )
            extraction, status, vocab_flags = validate_record(
                response.raw_text, record["report_id"], record["cancer_type"]
            )

            if status == "schema_failed":
                parsed = parse_json(response.raw_text)
                if parsed is not None:
                    repaired = validate_schema(
                        {**parsed, "report_id": record["report_id"], "cancer_type": record["cancer_type"]}
                    )
                    if repaired is not None:
                        vocab_flags = check_vocab(repaired)
                        status = "vocab_flagged" if vocab_flags else "valid"
                        extraction = repaired

            stats[status] = stats.get(status, 0) + 1
            stats["total_input_tokens"] += response.input_tokens
            stats["total_output_tokens"] += response.output_tokens
            stats["total_latency_ms"] += response.latency_ms
            if was_truncated:
                stats["truncated"] += 1

            created_at = datetime.now(timezone.utc).isoformat()

            if extraction is None:
                write_result(
                    {
                        "report_id": record["report_id"],
                        "cancer_type": record["cancer_type"],
                        "validation_status": status,
                        "raw_text": response.raw_text if args.debug else response.raw_text[:500],
                        "model": args.model,
                        "prompt_version": prompt_version,
                        "few_shot_path": args.few_shot,
                        "schema_version": None,
                        "latency_ms": round(response.latency_ms, 1),
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "truncated": was_truncated,
                        "run_id": run_id,
                        "created_at": created_at,
                    },
                    out_path,
                )
            else:
                meta = {
                    "model": args.model,
                    "prompt_version": prompt_version,
                    "few_shot_path": args.few_shot,
                    "latency_ms": round(response.latency_ms, 1),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "validation_status": status,
                    "run_id": run_id,
                    "created_at": created_at,
                    "truncated": was_truncated,
                }
                write_result(build_output_record(extraction, meta), out_path)

      
        avg_latency = stats["total_latency_ms"] / max(len(records), 1)

        # Compute pass_rate from the full output file (not just this run's stats),
        # so resume runs report the correct rate across all 180 records.
        all_results = list(load_records(out_path))
        _full_statuses = [r.get("validation_status") for r in all_results]
        pass_rate = sum(
            1 for s in _full_statuses if s in ("valid", "vocab_flagged")
        ) / len(_full_statuses) if _full_statuses else 0.0

        print(f"\n{'=' * 50}")
        print(f"Run ID:           {run_id}")
        print(f"Model:            {args.model}")
        print(f"Prompt:           {prompt_version}")
        print(f"Few-shot:         {args.few_shot or 'none'}")
        print(f"Records:          {stats['total']}")
        print(f"Valid:            {stats['valid']}")
        print(f"Vocab flagged:    {stats['vocab_flagged']}")
        print(f"JSON failed:      {stats['json_failed']}")
        print(f"Schema failed:    {stats['schema_failed']}")
        print(f"Pass rate:        {pass_rate:.1%}")
        print(f"Truncated:        {stats['truncated']}")
        print(f"Avg latency:      {avg_latency:.0f} ms")
        print(f"Total tokens in:  {stats['total_input_tokens']:,}")
        print(f"Total tokens out: {stats['total_output_tokens']:,}")
        print(f"{'=' * 50}\n")

        summary_path = out_path.with_suffix("").with_suffix(".summary.json")
        stats["pass_rate"] = round(pass_rate, 4)
        stats["avg_latency_ms"] = round(avg_latency, 1)

        # --- MLflow metrics --- (all_results already loaded above for pass_rate)
        tier_dist = tier_distribution(all_results)
        mlflow_metrics: dict[str, float] = {
            "pass_rate": round(pass_rate, 4),
            "tier1_rate": round(tier_dist.get("tier_1", 0.0), 4),
            "tier2_rate": round(tier_dist.get("tier_2", 0.0), 4),
            "tier3_rate": round(tier_dist.get("tier_3", 0.0), 4),
            "mean_latency_s": round(avg_latency / 1000, 4),
            "total_cost_usd": round(_compute_total_cost(all_results, args.model), 6),
            "total_input_tokens": stats["total_input_tokens"],
            "total_output_tokens": stats["total_output_tokens"],
            "mean_input_tokens": round(stats["total_input_tokens"] / max(len(records), 1), 1),
            "mean_output_tokens": round(stats["total_output_tokens"] / max(len(records), 1), 1),
        }

        if args.eval_set and Path(args.eval_set).exists():
            golds = list(load_records(args.eval_set))
            tracker.log_eval_dataset(args.eval_set, golds)
            agg = compute_aggregate_metrics(all_results, golds)
            mlflow_metrics["weighted_f1"] = round(agg.weighted_f1, 4)
            mlflow_metrics["mean_f1"] = round(agg.mean_f1, 4)
            # Per-field F1 — logged regardless of semantic scoring.
            field_metrics = compute_all_field_metrics(all_results, golds)
            for fm in field_metrics:
                mlflow_metrics[f"f1_{fm.field.replace('.', '_')}"] = round(fm.f1, 4)

            if not args.no_semantic:
                try:
                    from src.evaluation.metrics import (
                        compute_combined_score,
                        compute_semantic_metrics,
                    )

                    semantic_metrics = compute_semantic_metrics(all_results, golds)
                    for sm in semantic_metrics:
                        mlflow_metrics[f"similarity_complete_{sm.field}"] = round(
                            sm.mean_similarity_complete, 4
                        )
                        mlflow_metrics[f"omissions_{sm.field}"] = sm.n_omission
                        mlflow_metrics[f"hallucinations_{sm.field}"] = sm.n_hallucination
                    # Completeness-aware mean — every field is scored over all records.
                    if semantic_metrics:
                        mlflow_metrics["mean_similarity_complete"] = round(
                            sum(sm.mean_similarity_complete for sm in semantic_metrics)
                            / len(semantic_metrics),
                            4,
                        )
                    # Composite: completeness-aware similarity for free-text fields,
                    # F1 for TNM. Needs similarity, so it is logged only here.
                    mlflow_metrics["combined_score"] = round(
                        compute_combined_score(field_metrics, semantic_metrics), 4
                    )
                except Exception as exc:
                    logger.warning("Semantic similarity metrics skipped: %s", exc)
        # Drift guard: ground-truth baseline is ~62% null across all cancer types
        mlflow_metrics["m_null_rate"] = round(compute_m_null_rate(all_results), 4)

        # Per-field null rates — logged for completeness drift monitoring
        mlflow_metrics.update(compute_field_null_rates(all_results))

        # Write summary after all metrics are computed so combined_score is included.
        for key in ("weighted_f1", "mean_f1", "combined_score", "mean_similarity_complete"):
            if key in mlflow_metrics:
                stats[key] = mlflow_metrics[key]
        write_summary(stats, summary_path)
        print(f"Summary written to: {summary_path}")

        tracker.log_metrics(mlflow_metrics)
        tracker.log_artifact(out_path)
        tracker.log_artifact(summary_path)
        tracker.log_artifact(template_path)  # prompt template — enables cross-run comparison
        print(f"MLflow run ID:      {tracker.run_id}")
        _mlflow_run_id = tracker.run_id

    _run_monitors(_mlflow_run_id)
    _print_delta_summary(_mlflow_run_id)


if __name__ == "__main__":
    main()
