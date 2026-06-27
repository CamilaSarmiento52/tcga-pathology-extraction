"""
Stratified evaluation: group records by cancer_type or report_style, then
compute AggregateMetrics per stratum.

The `style` field is not present in pipeline output or annotation records —
it lives in the pilot corpus (data/processed/pilot_corpus_v1.jsonl). Call
`load_style_lookup(pilot_corpus_path)` and pass the result to
`attach_style(records, lookup)` before stratifying by report_style.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.metrics import AggregateMetrics, compute_aggregate_metrics


def load_style_lookup(pilot_corpus_path: str | Path) -> dict[str, str]:
    """Build {report_id: style} from pilot_corpus_v1.jsonl."""
    lookup: dict[str, str] = {}
    with open(pilot_corpus_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec.get("report_id")
            style = rec.get("style")
            if rid and style:
                lookup[rid] = style
    return lookup


def attach_style(records: list[dict], lookup: dict[str, str]) -> list[dict]:
    """Return records with 'style' field added from lookup (in-place copy)."""
    result = []
    for r in records:
        rc = dict(r)
        rc.setdefault("style", lookup.get(r.get("report_id", ""), "unknown"))
        result.append(rc)
    return result


def stratify(records: list[dict], by: str) -> dict[str, list[dict]]:
    """Group records by a field value.

    by='cancer_type' → groups on record['cancer_type']
    by='report_style' → groups on record['style']
    """
    key_fn = {
        "cancer_type": lambda r: r.get("cancer_type", "unknown"),
        "report_style": lambda r: r.get("style", "unknown"),
    }
    if by not in key_fn:
        raise ValueError(f"Unsupported stratifier '{by}'. Choose from: {list(key_fn)}")
    groups: dict[str, list[dict]] = {}
    for rec in records:
        k = key_fn[by](rec)
        groups.setdefault(k, []).append(rec)
    return groups


def compute_stratified_metrics(
    preds: list[dict],
    golds: list[dict],
    by: str,
) -> dict[str, AggregateMetrics]:
    """Compute AggregateMetrics for each stratum.

    Returns an empty dict for a stratum if it has no gold records.
    """
    gold_groups = stratify(golds, by)
    pred_map = {r["report_id"]: r for r in preds}
    results: dict[str, AggregateMetrics] = {}
    for stratum, stratum_golds in gold_groups.items():
        rids = {r["report_id"] for r in stratum_golds}
        stratum_preds = [pred_map[rid] for rid in rids if rid in pred_map]
        if not stratum_golds:
            continue
        results[stratum] = compute_aggregate_metrics(stratum_preds, stratum_golds)
    return results
