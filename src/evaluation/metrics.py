"""
Field-level and aggregate evaluation metrics for the extraction pipeline.

Evaluation compares pipeline output records (predictions) against manually
annotated ground truth records, matched on report_id.

Hallucination rate uses the pipeline's hallucination_flags (vocab violations
from validator.py), not the annotation's explanatory flags.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Parenthetical qualifiers to strip from free-text fields before comparison
# e.g. "lobectomy (right upper lobe)" → "lobectomy"
_PARENTHETICAL = re.compile(r"\s*\([^)]*\)")
_FREE_TEXT_FIELDS = frozenset({"primary_site", "specimen_type"})

FIELD_WEIGHTS: dict[str, int] = {
    "primary_site": 1,
    "specimen_type": 1,
    "histological_diagnosis": 2,
    "histological_subtype": 2,
    "tumor_grade": 2,
    "tnm_stage.T": 3,
    "tnm_stage.N": 3,
    "tnm_stage.M": 1,
}

ALL_FIELDS: list[str] = list(FIELD_WEIGHTS.keys())

# Fields scored by the continuous semantic layer (semantic.py); tumor_grade is
# included but scored by the deterministic grade parser, never by embeddings.
SEMANTIC_FIELDS: list[str] = [
    "primary_site",
    "specimen_type",
    "histological_diagnosis",
    "histological_subtype",
    "tumor_grade",
]


@dataclass
class FieldMetrics:
    field: str
    precision: float
    recall: float
    f1: float
    exact_match: float
    null_accuracy: float
    n_predictions: int
    n_gold: int


@dataclass
class SemanticFieldMetrics:
    field: str
    # Co-present (legacy): mean over pairs where both pred and gold are non-null.
    mean_similarity: float
    n_scored_pairs: int
    # Completeness-aware: every record scored — both-null -> 1.0, exactly-one-null
    # -> 0.0, both-present -> the semantic_score cascade. The four counts partition
    # all records so a low score can be traced to omissions vs hallucinations.
    mean_similarity_complete: float
    n_both_null: int
    n_omission: int  # gold present, pred null
    n_hallucination: int  # gold null, pred present
    n_both_present: int


@dataclass
class AggregateMetrics:
    weighted_f1: float
    mean_f1: float
    overall_exact_match: float
    hallucination_rate: float
    n_records: int


def get_field_value(record: dict, field: str) -> str | None:
    """Extract a field value, supporting dotted paths (e.g. 'tnm_stage.T')."""
    if "." not in field:
        return record.get(field)
    parent, child = field.split(".", 1)
    nested = record.get(parent)
    if not isinstance(nested, dict):
        return None
    return nested.get(child)


def normalize(val: str | None, field: str = "") -> str | None:
    if val is None:
        return None
    s = val.strip().lower()
    if field in _FREE_TEXT_FIELDS:
        s = _PARENTHETICAL.sub("", s).strip()
    return s


def compute_field_metrics(
    preds: list[dict],
    golds: list[dict],
    field: str,
) -> FieldMetrics:
    """Compute precision, recall, F1, exact match and null accuracy for one field.

    Joining is done on report_id. Predictions missing from preds are treated as
    all-null. Both None → true negative. Pred None, gold non-null → false negative.
    Pred non-null, gold None → false positive.

    For primary_site and specimen_type, parenthetical qualifiers are stripped
    before comparison (e.g. "lobectomy (right upper lobe)" matches "lobectomy").
    """
    gold_map = {r["report_id"]: r for r in golds}
    pred_map = {r["report_id"]: r for r in preds}

    tp = fp = fn = tn = 0
    exact_hits = 0
    null_hits = 0

    for rid, gold in gold_map.items():
        pred = pred_map.get(rid, {})
        g_val = normalize(get_field_value(gold, field), field)
        p_val = normalize(get_field_value(pred, field), field)

        # null accuracy
        if (g_val is None) == (p_val is None):
            null_hits += 1

        if g_val is None and p_val is None:
            tn += 1
            exact_hits += 1
        elif g_val is not None and p_val == g_val:
            tp += 1
            exact_hits += 1
        elif g_val is not None and p_val != g_val:
            if p_val is None:
                fn += 1
            else:
                # wrong non-null value counts as fp + fn
                fp += 1
                fn += 1
        elif g_val is None and p_val is not None:
            fp += 1

    n = len(gold_map)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return FieldMetrics(
        field=field,
        precision=precision,
        recall=recall,
        f1=f1,
        exact_match=exact_hits / n if n > 0 else 0.0,
        null_accuracy=null_hits / n if n > 0 else 0.0,
        n_predictions=len(pred_map),
        n_gold=n,
    )


def compute_hallucination_rate(preds: list[dict]) -> float:
    """Fraction of structurally valid records that have ≥1 vocab hallucination flag.

    Denominator: records that passed JSON + Pydantic layers (validation_status in
    {"valid", "vocab_flagged"}). Records with hallucination_flags are always
    marked "vocab_flagged" by the pipeline, so "valid" records always have null
    flags. Including both statuses gives the true hallucination rate.
    """
    structural_pass = [
        r for r in preds
        if r.get("validation_status") in ("valid", "vocab_flagged")
    ]
    if not structural_pass:
        return 0.0
    flagged = [
        r for r in structural_pass
        if r.get("hallucination_flags") and len(r["hallucination_flags"]) > 0
    ]
    return len(flagged) / len(structural_pass)


def compute_m_null_rate(preds: list[dict]) -> float:
    """Fraction of structurally valid records with a null tnm_stage.M.

    Drift guard for M-stage under-extraction: M is null in ~62% of ground truth
    across all three cancer types (see notebooks/05_ground_truth_inspection.ipynb),
    so a null M is usually correct and never triaged to human review. A run whose
    M null rate climbs well above that baseline signals the model has stopped
    extracting M when it is stated.
    """
    structural_pass = [
        r for r in preds
        if r.get("validation_status") in ("valid", "vocab_flagged")
    ]
    if not structural_pass:
        return 0.0
    null_m = [
        r for r in structural_pass
        if get_field_value(r, "tnm_stage.M") is None
    ]
    return len(null_m) / len(structural_pass)


_NULL_RATE_FIELDS: tuple[str, ...] = (
    "primary_site",
    "histological_diagnosis",
    "histological_subtype",
    "tumor_grade",
    "specimen_type",
    "tnm_stage.T",
    "tnm_stage.N",
    "tnm_stage.M",
)


def compute_field_null_rates(preds: list[dict]) -> dict[str, float]:
    """Per-field null rate over structurally valid records.

    Returns a dict keyed by ``null_rate_<field>`` (dots replaced by underscores).
    Only counts records that passed structural validation to avoid inflating rates
    with failed records that always have null fields.
    """
    structural_pass = [
        r for r in preds
        if r.get("validation_status") in ("valid", "vocab_flagged")
    ]
    if not structural_pass:
        return {f"null_rate_{f.replace('.', '_')}": 0.0 for f in _NULL_RATE_FIELDS}
    n = len(structural_pass)
    return {
        f"null_rate_{field.replace('.', '_')}": round(
            sum(1 for r in structural_pass if get_field_value(r, field) is None) / n, 4
        )
        for field in _NULL_RATE_FIELDS
    }


def compute_all_field_metrics(
    preds: list[dict],
    golds: list[dict],
) -> list[FieldMetrics]:
    return [compute_field_metrics(preds, golds, f) for f in ALL_FIELDS]


def compute_semantic_metrics(
    preds: list[dict],
    golds: list[dict],
) -> list[SemanticFieldMetrics]:
    """Per-field semantic similarity, reported two ways.

    * Co-present (legacy ``mean_similarity`` / ``n_scored_pairs``): mean over pairs
      where both sides are non-null — wording fidelity only.
    * Completeness-aware (``mean_similarity_complete``): every record is scored, so
      omissions and hallucinations are penalised. both-null -> 1.0 (correct
      absence), exactly-one-null -> 0.0, both-present -> the ``semantic_score``
      cascade. The four null-category counts partition all records.

    Null is detected with ``normalize`` here, exactly as ``semantic_score`` does
    internally, so the two views stay consistent. Imports the semantic scorer
    lazily so importing this module never pulls torch.
    """
    from src.evaluation.semantic import semantic_score

    gold_map = {r["report_id"]: r for r in golds}
    pred_map = {r["report_id"]: r for r in preds}
    n_records = len(gold_map)

    results: list[SemanticFieldMetrics] = []
    for field in SEMANTIC_FIELDS:
        copresent: list[float] = []
        complete_total = 0.0
        n_both_null = n_omission = n_hallucination = n_both_present = 0
        for rid, gold in gold_map.items():
            pred = pred_map.get(rid, {})
            g_val = normalize(get_field_value(gold, field), field)
            p_val = normalize(get_field_value(pred, field), field)
            if g_val is None and p_val is None:
                n_both_null += 1
                complete_total += 1.0
            elif g_val is not None and p_val is None:
                n_omission += 1  # complete score 0.0
            elif g_val is None and p_val is not None:
                n_hallucination += 1  # complete score 0.0
            else:
                # Both present — semantic_score never returns None here. Pass the
                # raw values so its own normalization/cascade runs as before.
                score = semantic_score(
                    get_field_value(pred, field), get_field_value(gold, field), field
                )
                n_both_present += 1
                copresent.append(score)
                complete_total += score
        results.append(
            SemanticFieldMetrics(
                field=field,
                mean_similarity=sum(copresent) / len(copresent) if copresent else 0.0,
                n_scored_pairs=len(copresent),
                mean_similarity_complete=complete_total / n_records if n_records else 0.0,
                n_both_null=n_both_null,
                n_omission=n_omission,
                n_hallucination=n_hallucination,
                n_both_present=n_both_present,
            )
        )
    return results


def compute_combined_score(
    field_metrics: list[FieldMetrics],
    semantic_metrics: list[SemanticFieldMetrics],
) -> float:
    """Single weighted score blending completeness-aware similarity (the free-text
    fields in ``SEMANTIC_FIELDS``) and F1 (the remaining closed-vocab TNM fields),
    weighted by ``FIELD_WEIGHTS``.

    Pure — takes already-computed metric lists, no model calls, so it can be
    back-filled offline from saved results. Intended as a *relative* ranking metric:
    SapBERT cosine has a non-zero floor, so the similarity half is more generous
    than the F1 half; this is not an absolute accuracy.
    """
    sim_by_field = {sm.field: sm.mean_similarity_complete for sm in semantic_metrics}
    f1_by_field = {fm.field: fm.f1 for fm in field_metrics}

    total_weight = sum(FIELD_WEIGHTS.values())
    if total_weight == 0:
        return 0.0

    numerator = 0.0
    for field, weight in FIELD_WEIGHTS.items():
        if field in SEMANTIC_FIELDS:
            numerator += weight * sim_by_field.get(field, 0.0)
        else:
            numerator += weight * f1_by_field.get(field, 0.0)
    return numerator / total_weight


def compute_aggregate_metrics(
    preds: list[dict],
    golds: list[dict],
) -> AggregateMetrics:
    field_metrics = compute_all_field_metrics(preds, golds)

    total_weight = sum(FIELD_WEIGHTS[fm.field] for fm in field_metrics)
    weighted_f1 = (
        sum(fm.f1 * FIELD_WEIGHTS[fm.field] for fm in field_metrics) / total_weight
        if total_weight > 0
        else 0.0
    )
    mean_f1 = sum(fm.f1 for fm in field_metrics) / len(field_metrics) if field_metrics else 0.0
    overall_exact = (
        sum(fm.exact_match for fm in field_metrics) / len(field_metrics)
        if field_metrics
        else 0.0
    )
    return AggregateMetrics(
        weighted_f1=weighted_f1,
        mean_f1=mean_f1,
        overall_exact_match=overall_exact,
        hallucination_rate=compute_hallucination_rate(preds),
        n_records=len(golds),
    )
