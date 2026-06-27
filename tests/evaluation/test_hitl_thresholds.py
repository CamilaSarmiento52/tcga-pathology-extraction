"""Tests for src/evaluation/hitl_thresholds.py."""

import pytest

from src.evaluation.hitl_thresholds import (
    TIER_1,
    TIER_2,
    TIER_3,
    classify_all,
    classify_record,
    tier_distribution,
)


def _rec(
    validation_status="valid",
    hallucination_flags=None,
    tnm_t="pT2",
    tnm_n="pN1",
    tnm_m="pM0",
):
    return {
        "validation_status": validation_status,
        "hallucination_flags": hallucination_flags,
        "tnm_stage": {"T": tnm_t, "N": tnm_n, "M": tnm_m},
    }


class TestClassifyRecord:
    def test_tier1_clean_record(self):
        assert classify_record(_rec()) == TIER_1

    def test_tier3_json_failed(self):
        assert classify_record(_rec(validation_status="json_failed")) == TIER_3

    def test_tier3_schema_failed(self):
        assert classify_record(_rec(validation_status="schema_failed")) == TIER_3

    def test_tier3_more_than_two_flags(self):
        rec = _rec(hallucination_flags=["f1", "f2", "f3"])
        assert classify_record(rec) == TIER_3

    def test_tier2_vocab_flagged(self):
        assert classify_record(_rec(validation_status="vocab_flagged")) == TIER_2

    def test_tier2_one_flag(self):
        rec = _rec(hallucination_flags=["invalid_tnm_T: 'bad'"])
        assert classify_record(rec) == TIER_2

    def test_tier2_exactly_two_flags(self):
        rec = _rec(hallucination_flags=["f1", "f2"])
        assert classify_record(rec) == TIER_2

    def test_tier2_high_difficulty_field_null(self):
        rec = _rec(tnm_t=None)
        assert classify_record(rec) == TIER_2

    def test_tier2_tnm_n_null(self):
        rec = _rec(tnm_n=None)
        assert classify_record(rec) == TIER_2

    def test_tier1_tnm_m_null(self):
        rec = _rec(tnm_m=None)
        assert classify_record(rec) == TIER_1

    def test_tier1_all_tnm_present_no_flags(self):
        rec = _rec(tnm_t="pT1", tnm_n="pN0", tnm_m="pM0", hallucination_flags=[])
        assert classify_record(rec) == TIER_1

    def test_missing_tnm_stage_key(self):
        rec = {"validation_status": "valid", "hallucination_flags": None}
        assert classify_record(rec) == TIER_2

    def test_tnm_stage_none(self):
        rec = {"validation_status": "valid", "hallucination_flags": None, "tnm_stage": None}
        assert classify_record(rec) == TIER_2


class TestClassifyAll:
    def test_classifies_list(self):
        records = [_rec(), _rec(validation_status="json_failed")]
        result = classify_all(records)
        assert result == [TIER_1, TIER_3]


class TestTierDistribution:
    def test_all_tier1(self):
        records = [_rec(), _rec()]
        dist = tier_distribution(records)
        assert dist[TIER_1] == pytest.approx(1.0)
        assert dist[TIER_2] == pytest.approx(0.0)
        assert dist[TIER_3] == pytest.approx(0.0)

    def test_fractions_sum_to_one(self):
        records = [_rec(), _rec(validation_status="json_failed"), _rec(tnm_t=None)]
        dist = tier_distribution(records)
        assert sum(dist.values()) == pytest.approx(1.0)

    def test_empty_records(self):
        dist = tier_distribution([])
        assert dist == {TIER_1: 0.0, TIER_2: 0.0, TIER_3: 0.0}
