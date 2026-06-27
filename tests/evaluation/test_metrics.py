"""Tests for src/evaluation/metrics.py."""

import pytest

from src.evaluation.metrics import (
    AggregateMetrics,
    FieldMetrics,
    compute_aggregate_metrics,
    compute_all_field_metrics,
    compute_combined_score,
    compute_field_metrics,
    compute_hallucination_rate,
    compute_m_null_rate,
    compute_semantic_metrics,
    get_field_value,
    normalize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _gold(report_id="R1", cancer_type="BRCA", **kwargs):
    base = {
        "report_id": report_id,
        "cancer_type": cancer_type,
        "primary_site": "left breast",
        "histological_diagnosis": "invasive ductal carcinoma",
        "histological_subtype": "ductal",
        "tumor_grade": "Grade 2",
        "tnm_stage": {"T": "pT2", "N": "pN1", "M": "pM0"},
        "specimen_type": "mastectomy",
    }
    base.update(kwargs)
    return base


def _pred(report_id="R1", validation_status="valid", hallucination_flags=None, **kwargs):
    base = {
        "report_id": report_id,
        "validation_status": validation_status,
        "hallucination_flags": hallucination_flags,
        "primary_site": "left breast",
        "histological_diagnosis": "invasive ductal carcinoma",
        "histological_subtype": "ductal",
        "tumor_grade": "Grade 2",
        "tnm_stage": {"T": "pT2", "N": "pN1", "M": "pM0"},
        "specimen_type": "mastectomy",
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# get_field_value
# ---------------------------------------------------------------------------


class TestGetFieldValue:
    def test_flat_field(self):
        assert get_field_value({"primary_site": "breast"}, "primary_site") == "breast"

    def test_dotted_path(self):
        rec = {"tnm_stage": {"T": "pT2", "N": "pN1"}}
        assert get_field_value(rec, "tnm_stage.T") == "pT2"

    def test_missing_flat(self):
        assert get_field_value({}, "primary_site") is None

    def test_missing_nested_parent(self):
        assert get_field_value({}, "tnm_stage.T") is None

    def test_nested_parent_none(self):
        assert get_field_value({"tnm_stage": None}, "tnm_stage.T") is None

    def test_nested_child_absent(self):
        assert get_field_value({"tnm_stage": {"N": "pN0"}}, "tnm_stage.T") is None


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_lowercases(self):
        assert normalize("Grade 2") == "grade 2"

    def test_strips_whitespace(self):
        assert normalize("  pT2  ") == "pt2"

    def test_none_passthrough(self):
        assert normalize(None) is None


# ---------------------------------------------------------------------------
# compute_field_metrics
# ---------------------------------------------------------------------------


class TestComputeFieldMetrics:
    def test_perfect_match(self):
        golds = [_gold()]
        preds = [_pred()]
        fm = compute_field_metrics(preds, golds, "primary_site")
        assert fm.f1 == pytest.approx(1.0)
        assert fm.exact_match == pytest.approx(1.0)

    def test_case_insensitive_match(self):
        golds = [_gold(primary_site="Left Breast")]
        preds = [_pred(primary_site="left breast")]
        fm = compute_field_metrics(preds, golds, "primary_site")
        assert fm.f1 == pytest.approx(1.0)

    def test_pred_null_gold_nonnull_is_fn(self):
        golds = [_gold(primary_site="left breast")]
        preds = [_pred(primary_site=None)]
        fm = compute_field_metrics(preds, golds, "primary_site")
        assert fm.recall == pytest.approx(0.0)
        assert fm.f1 == pytest.approx(0.0)

    def test_pred_nonnull_gold_null_is_fp(self):
        golds = [_gold(primary_site=None)]
        preds = [_pred(primary_site="left breast")]
        fm = compute_field_metrics(preds, golds, "primary_site")
        assert fm.precision == pytest.approx(0.0)

    def test_both_null_is_true_negative(self):
        golds = [_gold(primary_site=None)]
        preds = [_pred(primary_site=None)]
        fm = compute_field_metrics(preds, golds, "primary_site")
        assert fm.exact_match == pytest.approx(1.0)
        assert fm.null_accuracy == pytest.approx(1.0)

    def test_nested_field_tnm(self):
        golds = [_gold()]
        preds = [_pred()]
        fm = compute_field_metrics(preds, golds, "tnm_stage.T")
        assert fm.f1 == pytest.approx(1.0)

    def test_missing_pred_counts_as_null(self):
        golds = [_gold(report_id="R1"), _gold(report_id="R2")]
        preds = [_pred(report_id="R1")]  # R2 missing → treated as all-null
        fm = compute_field_metrics(preds, golds, "primary_site")
        # R1 matches (tp=1), R2 pred=null gold=non-null (fn=1)
        assert fm.recall == pytest.approx(0.5)

    def test_wrong_value_counts_as_fp_and_fn(self):
        golds = [_gold(primary_site="left breast")]
        preds = [_pred(primary_site="right breast")]
        fm = compute_field_metrics(preds, golds, "primary_site")
        assert fm.precision == pytest.approx(0.0)
        assert fm.recall == pytest.approx(0.0)

    def test_n_gold_and_n_predictions(self):
        golds = [_gold(report_id="R1"), _gold(report_id="R2")]
        preds = [_pred(report_id="R1")]
        fm = compute_field_metrics(preds, golds, "primary_site")
        assert fm.n_gold == 2
        assert fm.n_predictions == 1


# ---------------------------------------------------------------------------
# compute_hallucination_rate
# ---------------------------------------------------------------------------


class TestComputeHallucinationRate:
    def test_no_flags(self):
        preds = [_pred(validation_status="valid", hallucination_flags=None)]
        assert compute_hallucination_rate(preds) == pytest.approx(0.0)

    def test_all_flagged(self):
        preds = [_pred(validation_status="valid", hallucination_flags=["invalid_tnm_T: 'bad'"])]
        assert compute_hallucination_rate(preds) == pytest.approx(1.0)

    def test_only_structurally_valid_records_counted(self):
        preds = [
            _pred(report_id="R1", validation_status="vocab_flagged", hallucination_flags=["flag"]),
            _pred(report_id="R2", validation_status="json_failed", hallucination_flags=["flag"]),
        ]
        # Only R1 passed JSON+Pydantic → rate = 1/1
        assert compute_hallucination_rate(preds) == pytest.approx(1.0)

    def test_vocab_flagged_counted_in_denominator(self):
        preds = [
            _pred(report_id="R1", validation_status="valid", hallucination_flags=None),
            _pred(report_id="R2", validation_status="vocab_flagged", hallucination_flags=["flag"]),
        ]
        # 2 structurally valid, 1 flagged → rate = 0.5
        assert compute_hallucination_rate(preds) == pytest.approx(0.5)

    def test_empty_list_returns_zero(self):
        assert compute_hallucination_rate([]) == pytest.approx(0.0)

    def test_empty_flags_list_not_flagged(self):
        preds = [_pred(validation_status="valid", hallucination_flags=[])]
        assert compute_hallucination_rate(preds) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_m_null_rate
# ---------------------------------------------------------------------------


class TestComputeMNullRate:
    def test_all_m_present_zero(self):
        preds = [_pred()]  # default tnm_stage has M="pM0"
        assert compute_m_null_rate(preds) == pytest.approx(0.0)

    def test_null_m_counted(self):
        preds = [
            _pred(report_id="R1", tnm_stage={"T": "pT2", "N": "pN1", "M": None}),
            _pred(report_id="R2"),
        ]
        assert compute_m_null_rate(preds) == pytest.approx(0.5)

    def test_missing_tnm_stage_counts_as_null_m(self):
        preds = [_pred(tnm_stage=None)]
        assert compute_m_null_rate(preds) == pytest.approx(1.0)

    def test_structural_failures_excluded_from_denominator(self):
        preds = [
            _pred(report_id="R1", tnm_stage={"T": "pT2", "N": "pN1", "M": None}),
            _pred(report_id="R2", validation_status="json_failed", tnm_stage=None),
        ]
        assert compute_m_null_rate(preds) == pytest.approx(1.0)

    def test_empty_preds_zero(self):
        assert compute_m_null_rate([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Field normalisation for free-text fields
# ---------------------------------------------------------------------------


class TestFreeTextFieldNormalization:
    def test_specimen_type_strips_parenthetical(self):
        """'lobectomy (right upper lobe)' should match 'lobectomy' for specimen_type."""
        golds = [_gold(specimen_type="lobectomy")]
        preds = [_pred(specimen_type="lobectomy (right upper lobe)")]
        fm = compute_field_metrics(preds, golds, "specimen_type")
        assert fm.f1 == pytest.approx(1.0)

    def test_primary_site_strips_parenthetical(self):
        """'right breast (upper outer quadrant)' should match 'right breast' for primary_site."""
        golds = [_gold(primary_site="right breast")]
        preds = [_pred(primary_site="right breast (upper outer quadrant)")]
        fm = compute_field_metrics(preds, golds, "primary_site")
        assert fm.f1 == pytest.approx(1.0)

    def test_parenthetical_stripping_does_not_apply_to_tnm(self):
        """Parenthetical stripping must not affect TNM fields — pN0(sn) stays as-is."""
        golds = [_gold()]
        preds = [_pred(tnm_stage={"T": "pT2", "N": "pN0(sn)", "M": "pM0"})]
        fm = compute_field_metrics(preds, golds, "tnm_stage.N")
        # pN0(sn) != pN1 → mismatch (no stripping applied to TNM fields)
        assert fm.f1 == pytest.approx(0.0)

    def test_normalize_strips_parenthetical_for_free_text_field(self):
        from src.evaluation.metrics import normalize
        assert normalize("lobectomy (right upper lobe)", "specimen_type") == "lobectomy"
        assert normalize("lobectomy (right upper lobe)", "tnm_stage.N") == "lobectomy (right upper lobe)"


# ---------------------------------------------------------------------------
# compute_aggregate_metrics
# ---------------------------------------------------------------------------


class TestComputeAggregateMetrics:
    def test_perfect_run(self):
        golds = [_gold()]
        preds = [_pred()]
        agg = compute_aggregate_metrics(preds, golds)
        assert isinstance(agg, AggregateMetrics)
        assert agg.weighted_f1 == pytest.approx(1.0)
        assert agg.hallucination_rate == pytest.approx(0.0)
        assert agg.n_records == 1

    def test_all_null_preds(self):
        golds = [_gold()]
        preds = [
            _pred(
                primary_site=None,
                histological_diagnosis=None,
                histological_subtype=None,
                tumor_grade=None,
                tnm_stage={"T": None, "N": None, "M": None},
                specimen_type=None,
            )
        ]
        agg = compute_aggregate_metrics(preds, golds)
        assert agg.weighted_f1 == pytest.approx(0.0)

    def test_returns_aggregate_metrics_type(self):
        agg = compute_aggregate_metrics([_pred()], [_gold()])
        assert isinstance(agg, AggregateMetrics)


# ---------------------------------------------------------------------------
# compute_combined_score
# ---------------------------------------------------------------------------


class TestCombinedScore:
    def test_weighted_blend_of_similarity_and_f1(self):
        # Similarity fields all exact-match (-> 1.0, no embedding needed); TNM
        # fields all wrong (-> F1 0.0). Composite must be the weighted mean:
        # semantic weight 1+1+2+2+2 = 8, TNM weight 3+3+1 = 7, total 15 -> 8/15.
        gold = _gold()
        pred = _pred(tnm_stage={"T": "pT9", "N": "pN9", "M": "pM9"})
        field_metrics = compute_all_field_metrics([pred], [gold])
        semantic_metrics = compute_semantic_metrics([pred], [gold])

        score = compute_combined_score(field_metrics, semantic_metrics)
        assert score == pytest.approx(8 / 15)

    def test_perfect_match_is_one(self):
        field_metrics = compute_all_field_metrics([_pred()], [_gold()])
        semantic_metrics = compute_semantic_metrics([_pred()], [_gold()])
        assert compute_combined_score(field_metrics, semantic_metrics) == pytest.approx(1.0)
