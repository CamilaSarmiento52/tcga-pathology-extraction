"""
Human-in-the-loop tier classification for extraction results.

Tier 1 — auto-accept: fully valid, no vocab flags, all high-difficulty TNM fields present.
Tier 2 — human review: vocab-flagged, or T/N missing, or 1-2 flags.
Tier 3 — reject/re-run: structural failure (JSON/schema) or more than 2 hallucination flags.

M-stage is deliberately excluded from triage: it is null in ~62% of ground truth
across all three cancer types (BRCA, LUAD, LUSC — see
notebooks/05_ground_truth_inspection.ipynb), so a null M is usually the correct
answer and routing it to human review would flood Tier 2 with fine records.
Wrongly-null M is still penalised in evaluation (tnm_stage.M field metrics), and
the run-level `m_null_rate` MLflow metric guards against under-extraction drift.
"""

from __future__ import annotations

TIER_1 = "tier_1"
TIER_2 = "tier_2"
TIER_3 = "tier_3"

# tnm_stage.M excluded by design — null M is clinically normal (see module docstring)
HIGH_DIFFICULTY_FIELDS = ["tnm_stage.T", "tnm_stage.N"]


def _get_nested(record: dict, dotted: str):
    parent, _, child = dotted.partition(".")
    nested = record.get(parent)
    if not isinstance(nested, dict):
        return None
    return nested.get(child)


def classify_record(record: dict) -> str:
    """Assign a HITL tier to a single pipeline output record."""
    status = record.get("validation_status")
    flags = record.get("hallucination_flags") or []
    n_flags = len(flags)

    if status in ("json_failed", "schema_failed") or n_flags > 2:
        return TIER_3

    if status == "vocab_flagged" or n_flags >= 1:
        return TIER_2

    if any(_get_nested(record, f) is None for f in HIGH_DIFFICULTY_FIELDS):
        return TIER_2

    return TIER_1


def classify_all(records: list[dict]) -> list[str]:
    return [classify_record(r) for r in records]


def tier_distribution(records: list[dict]) -> dict[str, float]:
    """Return fraction of records in each tier (values sum to 1.0)."""
    tiers = classify_all(records)
    n = len(tiers)
    if n == 0:
        return {TIER_1: 0.0, TIER_2: 0.0, TIER_3: 0.0}
    return {
        TIER_1: tiers.count(TIER_1) / n,
        TIER_2: tiers.count(TIER_2) / n,
        TIER_3: tiers.count(TIER_3) / n,
    }
