"""Tests for src/evaluation/stratifier.py."""

import json
import pytest

from src.evaluation.stratifier import (
    attach_style,
    compute_stratified_metrics,
    load_style_lookup,
    stratify,
)


def _rec(report_id, cancer_type="BRCA", style=None, **kwargs):
    r = {"report_id": report_id, "cancer_type": cancer_type}
    if style is not None:
        r["style"] = style
    r.update(kwargs)
    return r


def _gold(report_id, cancer_type="BRCA", **fields):
    base = {
        "report_id": report_id,
        "cancer_type": cancer_type,
        "primary_site": "breast",
        "histological_diagnosis": "IDC",
        "histological_subtype": "ductal",
        "tumor_grade": "Grade 2",
        "tnm_stage": {"T": "pT2", "N": "pN1", "M": "pM0"},
        "specimen_type": "mastectomy",
    }
    base.update(fields)
    return base


def _pred(report_id, cancer_type="BRCA", **fields):
    base = {
        "report_id": report_id,
        "cancer_type": cancer_type,
        "validation_status": "valid",
        "hallucination_flags": None,
        "primary_site": "breast",
        "histological_diagnosis": "IDC",
        "histological_subtype": "ductal",
        "tumor_grade": "Grade 2",
        "tnm_stage": {"T": "pT2", "N": "pN1", "M": "pM0"},
        "specimen_type": "mastectomy",
    }
    base.update(fields)
    return base


# ---------------------------------------------------------------------------
# load_style_lookup
# ---------------------------------------------------------------------------


class TestLoadStyleLookup:
    def test_builds_lookup(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            json.dumps({"report_id": "R1", "style": "synoptic"}) + "\n"
            + json.dumps({"report_id": "R2", "style": "narrative"}) + "\n"
        )
        lookup = load_style_lookup(corpus)
        assert lookup == {"R1": "synoptic", "R2": "narrative"}

    def test_skips_missing_fields(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(json.dumps({"report_id": "R1"}) + "\n")
        lookup = load_style_lookup(corpus)
        assert lookup == {}


# ---------------------------------------------------------------------------
# attach_style
# ---------------------------------------------------------------------------


class TestAttachStyle:
    def test_adds_style(self):
        records = [{"report_id": "R1"}]
        lookup = {"R1": "synoptic"}
        result = attach_style(records, lookup)
        assert result[0]["style"] == "synoptic"

    def test_missing_report_id_gets_unknown(self):
        records = [{"report_id": "R99"}]
        lookup = {"R1": "synoptic"}
        result = attach_style(records, lookup)
        assert result[0]["style"] == "unknown"

    def test_does_not_mutate_originals(self):
        original = {"report_id": "R1"}
        attach_style([original], {"R1": "synoptic"})
        assert "style" not in original


# ---------------------------------------------------------------------------
# stratify
# ---------------------------------------------------------------------------


class TestStratify:
    def test_by_cancer_type(self):
        records = [
            _rec("R1", cancer_type="BRCA"),
            _rec("R2", cancer_type="LUAD"),
            _rec("R3", cancer_type="BRCA"),
        ]
        groups = stratify(records, "cancer_type")
        assert set(groups.keys()) == {"BRCA", "LUAD"}
        assert len(groups["BRCA"]) == 2

    def test_by_report_style(self):
        records = [
            _rec("R1", style="synoptic"),
            _rec("R2", style="narrative"),
        ]
        groups = stratify(records, "report_style")
        assert set(groups.keys()) == {"synoptic", "narrative"}

    def test_invalid_by_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            stratify([], "invalid_key")

    def test_empty_records(self):
        assert stratify([], "cancer_type") == {}


# ---------------------------------------------------------------------------
# compute_stratified_metrics
# ---------------------------------------------------------------------------


class TestComputeStratifiedMetrics:
    def test_returns_metrics_per_stratum(self):
        golds = [_gold("R1", cancer_type="BRCA"), _gold("R2", cancer_type="LUAD")]
        preds = [_pred("R1", cancer_type="BRCA"), _pred("R2", cancer_type="LUAD")]
        result = compute_stratified_metrics(preds, golds, "cancer_type")
        assert set(result.keys()) == {"BRCA", "LUAD"}

    def test_empty_stratum_skipped(self):
        golds = [_gold("R1", cancer_type="BRCA")]
        preds = []
        result = compute_stratified_metrics(preds, golds, "cancer_type")
        assert "BRCA" in result

    def test_missing_preds_handled(self):
        golds = [_gold("R1", cancer_type="BRCA"), _gold("R2", cancer_type="BRCA")]
        preds = [_pred("R1", cancer_type="BRCA")]  # R2 missing
        result = compute_stratified_metrics(preds, golds, "cancer_type")
        assert result["BRCA"].n_records == 2
