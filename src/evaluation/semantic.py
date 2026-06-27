"""
Continuous semantic similarity scoring for free-text extraction fields.

Version: v1.2 (SEMANTIC_VERSION — increment on any change to the embedding model,
grade parser vocab, laterality rule, scoring cascade, or the reported aggregation
methodology). v1.1 added the completeness-aware aggregation in metrics.py (null
mismatches scored). v1.2 makes the per-pair cascade fairer to correct-but-differently-
worded values: the grade parser now reads Roman numerals (Grade II-III == Grade 2-3),
the differentiation crosswalk covers "moderately to poorly differentiated", and an
expert-curated synonym table canonicalises equivalent histological_subtype acronyms
(NST == NOS) before scoring so they exact-match instead of falling to cosine (~0.53).

Per pred/gold pair the score cascade is:
  exact match after normalization -> 1.0
  histological_subtype synonyms   -> canonicalised, then exact-match -> 1.0
  tumor_grade                     -> ordinal parser: 1.0 equal / 0.5 overlapping range
                                     / 0.0 disjoint (never cosine; reads Roman + en-dash)
  primary_site laterality conflict-> 0.0 regardless of similarity
  otherwise                       -> SapBERT cosine similarity

SapBERT (PubMedBERT fine-tuned on UMLS synonym pairs) is used because its training
objective is exactly this problem: different surface forms of the same medical
concept embed close together. torch/transformers are imported lazily so the
extraction pipeline path never pays the import cost.
"""

from __future__ import annotations

import re

import numpy as np

from src.evaluation.metrics import normalize

SEMANTIC_VERSION = "v1.2"
EMBEDDING_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# Fields scored by embedding cosine; tumor_grade is parser-only (see semantic_score)
CONCEPT_FIELDS = frozenset(
    {"primary_site", "specimen_type", "histological_diagnosis", "histological_subtype"}
)

_model = None  # lazy (tokenizer, model) singleton
_embedding_cache: dict[str, np.ndarray] = {}

_PARENS = re.compile(r"\s*\([^)]*\)")
_GRADE_RANGE = re.compile(r"\bgrade\s*([1-4])\s*-\s*([1-4])\b")
_GRADE_NUM = re.compile(r"\b(?:grade|nhg|g)\s*([1-4])\b")
_LATERALITY = re.compile(r"\b(left|right|bilateral)\b")

# Unicode dash variants (en/em/figure dash, minus sign, etc.) -> ASCII hyphen. Gold
# values write ranges with an en-dash ("Grade 2–3"), which would otherwise miss the
# hyphen-only range regex and silently parse as just the first grade.
_DASHES = re.compile(r"[‐-―−]")

# Roman numerals used for grade (I–IV) -> Arabic, applied before the grade regexes
# so "Grade II-III" parses identically to "Grade 2-3". Longest-first match order and
# word boundaries keep "ii"/"iii" from being mangled into "i" + "i" or hitting words.
_ROMAN_GRADE = {"iv": "4", "iii": "3", "ii": "2", "i": "1"}
_ROMAN_RE = re.compile(r"\b(iv|iii|ii|i)\b")

# Differentiation descriptors -> ordinal grade (standard correspondence)
_DIFFERENTIATION = [
    ("moderate to poorly differentiated", "G2-G3"),
    ("moderately to poorly differentiated", "G2-G3"),
    ("poorly to moderately differentiated", "G2-G3"),
    ("well differentiated", "G1"),
    ("moderately differentiated", "G2"),
    ("poorly differentiated", "G3"),
    ("undifferentiated", "G4"),
]

# Expert-curated equivalences for histological_subtype: surface forms that denote the
# same concept but embed only ~0.53 apart in SapBERT as bare acronyms. Mapped to a
# shared sentinel so the exact-match path scores them 1.0 instead of cosine. Grow this
# as the pathologist (Mireia) signs off on further clusters. NOTE: "ductal" is left out
# deliberately — whether it is equivalent to NST/NOS is a domain call for her to make.
_SUBTYPE_SYNONYMS = {
    "nst": "__nst_nos__",
    "nos": "__nst_nos__",
    "no special type": "__nst_nos__",
    "not otherwise specified": "__nst_nos__",
    "nos (no special type)": "__nst_nos__",
    "nos (not otherwise specified)": "__nst_nos__",
}

# IASLC / descriptive grades -> ordinal crosswalk (low≈G1, intermediate≈G2, high≈G3)
_DESCRIPTIVE = {
    "low": "G1",
    "low grade": "G1",
    "intermediate": "G2",
    "intermediate grade": "G2",
    "high": "G3",
    "high grade": "G3",
}


def _get_model():
    """Lazy singleton load of the pinned SapBERT tokenizer + model (CPU, eval mode)."""
    global _model
    if _model is None:
        import torch  # noqa: F401 — ensures a clear ImportError before transformers
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL)
        model.eval()
        _model = (tokenizer, model)
    return _model


def embed(text: str) -> np.ndarray:
    """CLS-token SapBERT embedding, cached per unique string within the process."""
    cached = _embedding_cache.get(text)
    if cached is not None:
        return cached

    import torch

    tokenizer, model = _get_model()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        output = model(**inputs)
        vector = output.last_hidden_state[0, 0, :].numpy()
    _embedding_cache[text] = vector
    return vector


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def parse_grade(value: str | None) -> str | None:
    """Canonicalise a tumour grade to an ordinal token (G1–G4 or a G-range).

    Handles numeric forms (Grade 2, G2, NHG2), ranges (Grade 2-3), differentiation
    descriptors (well/moderately/poorly/undifferentiated) and the IASLC
    low/intermediate/high crosswalk. Unparseable -> None (caller falls back to
    exact match).
    """
    if value is None:
        return None
    s = _PARENS.sub("", value.strip().lower()).strip()
    if not s:
        return None
    s = _DASHES.sub("-", s)
    s = _ROMAN_RE.sub(lambda m: _ROMAN_GRADE[m.group(1)], s)

    m = _GRADE_RANGE.search(s)
    if m:
        return f"G{m.group(1)}-G{m.group(2)}"

    for phrase, grade in _DIFFERENTIATION:
        if phrase in s:
            return grade

    m = _GRADE_NUM.search(s)
    if m:
        return f"G{m.group(1)}"

    return _DESCRIPTIVE.get(s)


def _grade_set(token: str) -> set[int]:
    """Expand a canonical grade token to the set of ordinal grades it covers.

    "G3" -> {3}; "G2-G3" -> {2, 3}. Used to score range/single overlap.
    """
    nums = [int(c) for c in token if c.isdigit()]
    if len(nums) == 2:
        return set(range(min(nums), max(nums) + 1))
    return set(nums)


def extract_laterality(value: str | None) -> str | None:
    if value is None:
        return None
    found = set(_LATERALITY.findall(value.lower()))
    if "bilateral" in found or {"left", "right"} <= found:
        return "bilateral"
    if found:
        return found.pop()
    return None


def semantic_score(pred: str | None, gold: str | None, field: str) -> float | None:
    """Continuous similarity score for one pred/gold pair; None if either is null.

    Null mismatches are extraction errors, not wording differences — they belong to
    the exact-match FN/FP accounting, so they are excluded here (returned as None).
    """
    p = normalize(pred, field)
    g = normalize(gold, field)
    if p is None or g is None:
        return None

    if field == "histological_subtype":
        # Collapse expert-curated synonyms (e.g. NST/NOS) to a shared token so the
        # exact-match path below scores them 1.0 instead of falling through to cosine.
        p = _SUBTYPE_SYNONYMS.get(p, p)
        g = _SUBTYPE_SYNONYMS.get(g, g)

    if p == g:
        return 1.0

    if field == "tumor_grade":
        pg, gg = parse_grade(p), parse_grade(g)
        if pg is None or gg is None:
            return 0.0  # unparseable falls back to exact match, which already failed
        if pg == gg:
            return 1.0
        # Partial credit when one side's grade(s) fall inside the other's range,
        # e.g. pred "Grade 3" vs gold "Grade 2-3" — right region, different specificity.
        if _grade_set(pg) & _grade_set(gg):
            return 0.5
        return 0.0

    if field == "primary_site":
        pl, gl = extract_laterality(p), extract_laterality(g)
        if pl is not None and gl is not None and pl != gl:
            return 0.0

    if field not in CONCEPT_FIELDS:
        return 0.0  # non-concept fields never reach the embedding

    return cosine_similarity(embed(p), embed(g))
