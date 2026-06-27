"""Tests for src/evaluation/semantic.py.

Unit tests mock `embed` so no model is downloaded; the single integration test
that loads the real SapBERT model is marked slow (deselected by default, run with
`uv run pytest -m slow`).
"""

import numpy as np
import pytest

import src.evaluation.semantic as semantic
from src.evaluation.metrics import SEMANTIC_FIELDS, compute_semantic_metrics
from src.evaluation.semantic import (
    cosine_similarity,
    extract_laterality,
    parse_grade,
    semantic_score,
)


@pytest.fixture
def fake_embed(monkeypatch):
    """Replace embed with a deterministic fake; returns the call log."""
    calls: list[str] = []
    vectors = {}

    def _embed(text: str) -> np.ndarray:
        calls.append(text)
        # Stable pseudo-random unit-ish vector per distinct string
        if text not in vectors:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vectors[text] = rng.normal(size=8)
        return vectors[text]

    monkeypatch.setattr(semantic, "embed", _embed)
    return calls


def _high_cosine_embed(monkeypatch):
    """Make every pair score cosine ~0.99 to test that hard rules override it."""
    monkeypatch.setattr(semantic, "embed", lambda text: np.array([1.0, 0.05]))


# ---------------------------------------------------------------------------
# parse_grade
# ---------------------------------------------------------------------------


class TestParseGrade:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("Grade 2", "G2"),
            ("G2", "G2"),
            ("NHG2", "G2"),
            ("Grade 2 (moderately differentiated, stated)", "G2"),
            ("well differentiated", "G1"),
            ("moderately differentiated", "G2"),
            ("poorly differentiated", "G3"),
            ("undifferentiated", "G4"),
            ("Grade 2-3", "G2-G3"),
            ("moderate to poorly differentiated", "G2-G3"),
            ("low grade", "G1"),
            ("Intermediate", "G2"),
            ("high grade", "G3"),
            ("not a grade at all", None),
            (None, None),
        ],
    )
    def test_canonical_forms(self, value, expected):
        assert parse_grade(value) == expected


# ---------------------------------------------------------------------------
# extract_laterality
# ---------------------------------------------------------------------------


class TestExtractLaterality:
    def test_left_right_bilateral(self):
        assert extract_laterality("left breast") == "left"
        assert extract_laterality("Right upper lobe") == "right"
        assert extract_laterality("bilateral breasts") == "bilateral"
        assert extract_laterality("left and right breast") == "bilateral"
        assert extract_laterality("breast") is None
        assert extract_laterality(None) is None


# ---------------------------------------------------------------------------
# semantic_score cascade
# ---------------------------------------------------------------------------


class TestSemanticScore:
    def test_null_either_side_returns_none(self, fake_embed):
        assert semantic_score(None, "left breast", "primary_site") is None
        assert semantic_score("left breast", None, "primary_site") is None

    def test_exact_match_short_circuits_before_embedding(self, fake_embed):
        score = semantic_score("Left Breast", "left breast", "primary_site")
        assert score == pytest.approx(1.0)
        assert fake_embed == []  # embed never called

    def test_parenthetical_normalization_counts_as_exact(self, fake_embed):
        score = semantic_score(
            "lobectomy (right upper lobe)", "lobectomy", "specimen_type"
        )
        assert score == pytest.approx(1.0)
        assert fake_embed == []

    def test_grade_equivalent_forms_score_one(self, fake_embed):
        assert semantic_score("G2", "Grade 2", "tumor_grade") == pytest.approx(1.0)
        assert semantic_score(
            "Grade 2 (moderately differentiated)", "moderately differentiated", "tumor_grade"
        ) == pytest.approx(1.0)
        assert fake_embed == []  # grade never reaches the embedding

    def test_grade_mismatch_scores_zero(self, fake_embed):
        assert semantic_score("Grade 2", "Grade 3", "tumor_grade") == pytest.approx(0.0)
        assert fake_embed == []

    def test_unparseable_grade_falls_back_to_exact(self, fake_embed):
        assert semantic_score("weird grade", "Grade 2", "tumor_grade") == pytest.approx(0.0)
        assert fake_embed == []

    def test_laterality_conflict_forces_zero(self, monkeypatch):
        _high_cosine_embed(monkeypatch)
        score = semantic_score("left breast", "right breast", "primary_site")
        assert score == pytest.approx(0.0)

    def test_same_laterality_proceeds_to_cosine(self, monkeypatch):
        _high_cosine_embed(monkeypatch)
        score = semantic_score("left breast", "breast, left side", "primary_site")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_concept_field_uses_cosine(self, fake_embed):
        score = semantic_score(
            "invasive ductal carcinoma", "infiltrating ductal carcinoma",
            "histological_diagnosis",
        )
        assert isinstance(score, float)
        assert len(fake_embed) == 2


# ---------------------------------------------------------------------------
# compute_semantic_metrics
# ---------------------------------------------------------------------------


class TestComputeSemanticMetrics:
    def test_means_and_pair_counts(self, monkeypatch):
        _high_cosine_embed(monkeypatch)
        golds = [
            {
                "report_id": "R1",
                "primary_site": "left breast",
                "specimen_type": "mastectomy",
                "histological_diagnosis": "invasive ductal carcinoma",
                "histological_subtype": None,
                "tumor_grade": "Grade 2",
            },
            {
                "report_id": "R2",
                "primary_site": "right lung",
                "specimen_type": None,
                "histological_diagnosis": None,
                "histological_subtype": None,
                "tumor_grade": None,
            },
        ]
        preds = [
            {
                "report_id": "R1",
                "primary_site": "left breast",        # exact -> 1.0
                "specimen_type": "total mastectomy",  # cosine ~1.0 (mocked)
                "histological_diagnosis": None,        # null pair -> excluded
                "histological_subtype": None,          # both null -> excluded
                "tumor_grade": "G3",                   # grade mismatch -> 0.0
            },
            {
                "report_id": "R2",
                "primary_site": "left lung",           # laterality conflict -> 0.0
                "specimen_type": None,
                "histological_diagnosis": None,
                "histological_subtype": None,
                "tumor_grade": None,
            },
        ]
        results = {m.field: m for m in compute_semantic_metrics(preds, golds)}

        assert set(results) == set(SEMANTIC_FIELDS)
        assert results["primary_site"].n_scored_pairs == 2
        assert results["primary_site"].mean_similarity == pytest.approx(0.5)
        assert results["specimen_type"].n_scored_pairs == 1
        assert results["specimen_type"].mean_similarity == pytest.approx(1.0, abs=0.01)
        assert results["histological_diagnosis"].n_scored_pairs == 0
        assert results["histological_diagnosis"].mean_similarity == pytest.approx(0.0)
        assert results["tumor_grade"].n_scored_pairs == 1
        assert results["tumor_grade"].mean_similarity == pytest.approx(0.0)

        # Completeness-aware view scores every record (n_records = 2).
        # primary_site: 1.0 (exact) + 0.0 (laterality) -> 0.5
        ps = results["primary_site"]
        assert ps.mean_similarity_complete == pytest.approx(0.5)
        assert (ps.n_both_present, ps.n_both_null, ps.n_omission, ps.n_hallucination) == (2, 0, 0, 0)
        # specimen_type: ~1.0 (cosine) + 1.0 (both null) -> ~1.0
        st = results["specimen_type"]
        assert st.mean_similarity_complete == pytest.approx(1.0, abs=0.01)
        assert (st.n_both_present, st.n_both_null) == (1, 1)
        # histological_diagnosis: 0.0 (omission) + 1.0 (both null) -> 0.5
        hd = results["histological_diagnosis"]
        assert hd.mean_similarity_complete == pytest.approx(0.5)
        assert (hd.n_omission, hd.n_both_null) == (1, 1)
        # histological_subtype: both null on both records -> 1.0
        assert results["histological_subtype"].mean_similarity_complete == pytest.approx(1.0)
        assert results["histological_subtype"].n_both_null == 2
        # tumor_grade: 0.0 (grade mismatch) + 1.0 (both null) -> 0.5
        assert results["tumor_grade"].mean_similarity_complete == pytest.approx(0.5)

    def test_null_category_partition(self, monkeypatch):
        # Exact / null only — no embedding needed; mock is a safety net.
        _high_cosine_embed(monkeypatch)
        golds = [
            {"report_id": "A", "primary_site": "left breast"},  # both present
            {"report_id": "B", "primary_site": "left breast"},  # omission
            {"report_id": "C", "primary_site": None},           # hallucination
            {"report_id": "D", "primary_site": None},           # both null
        ]
        preds = [
            {"report_id": "A", "primary_site": "left breast"},
            {"report_id": "B", "primary_site": None},
            {"report_id": "C", "primary_site": "left breast"},
            {"report_id": "D", "primary_site": None},
        ]
        ps = {m.field: m for m in compute_semantic_metrics(preds, golds)}["primary_site"]
        assert (ps.n_both_present, ps.n_omission, ps.n_hallucination, ps.n_both_null) == (1, 1, 1, 1)
        # 1.0 (exact) + 0.0 + 0.0 + 1.0 (both null) over 4 records
        assert ps.mean_similarity_complete == pytest.approx(0.5)
        # Co-present view unchanged: only record A is scored.
        assert ps.n_scored_pairs == 1
        assert ps.mean_similarity == pytest.approx(1.0)

    def test_metrics_module_does_not_import_torch(self):
        import subprocess
        import sys

        code = (
            "import sys; import src.evaluation.metrics; "
            "sys.exit(1 if 'torch' in sys.modules else 0)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True)
        assert result.returncode == 0, "importing metrics must not import torch"


# ---------------------------------------------------------------------------
# Real-model integration (slow — run with `uv run pytest -m slow`)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRealModel:
    def test_synonyms_score_higher_than_unrelated(self):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        same = cosine_similarity(
            semantic.embed("invasive ductal carcinoma"),
            semantic.embed("infiltrating ductal carcinoma"),
        )
        different = cosine_similarity(
            semantic.embed("mastectomy"),
            semantic.embed("lung biopsy"),
        )
        assert same > different + 0.1
        assert same > 0.8
