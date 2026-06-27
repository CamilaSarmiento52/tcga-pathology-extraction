import json
import re
from typing import Optional

from pydantic import ValidationError

from src.schema import SCHEMA_VERSION, PathologyExtraction

VALIDATOR_VERSION = "v1.3"

# T: optional prefix + T + stage code + optional parenthetical modifier (e.g. pT4b(mult))
_TNM_T = re.compile(r"^(p|c|y|r|a)?T([0-4][a-c]?|X|is)(\([^)]+\))?$", re.IGNORECASE)
# N: optional prefix + N + stage code + optional 'mi' suffix + optional parentheticals
# Covers: pN0(sn), pN1mi, pN0(i-), pN0(i+), pN0(i-)(sn), pN0 (i+) (sn)
_TNM_N = re.compile(r"^(p|c|y|r)?N([0-3][a-c]?|X)(mi)?(\s*\([^)]+\))*$", re.IGNORECASE)
_TNM_M = re.compile(r"^(p|c)?M([01][a-c]?|X)$", re.IGNORECASE)

_GRADE_VOCAB = frozenset(
    {
        # Standard numeric grades
        "Grade 1", "Grade 2", "Grade 3", "Grade 4",
        "G1", "G2", "G3", "G4", "GX",
        # Range notations (model outputs when report uses range)
        "Grade 1-2", "Grade 2-3", "Grade 3-4",
        # Nottingham Histological Grade labels
        "NHG1", "NHG2", "NHG3",
        # IASLC grades (lung)
        "Low", "Intermediate", "High",
        # Differentiation descriptors
        "well differentiated", "moderately differentiated",
        "poorly differentiated", "undifferentiated",
        "low grade", "intermediate grade", "high grade",
        "moderate to poorly differentiated",
        "moderately to poorly differentiated",
        "poorly to moderately differentiated",
    }
)

# Normalise grade surface form before vocab check — mirrors semantic.py parse_grade
# so Roman numerals and dash variants ("Grade II-III", "Grade 2–3") all accept.
_GRADE_DASHES = re.compile(r"[‐-―−]")
_GRADE_ROMAN_RE = re.compile(r"\b(iv|iii|ii|i)\b")
_GRADE_ROMAN = {"iv": "4", "iii": "3", "ii": "2", "i": "1"}
_GRADE_VOCAB_LOWER = frozenset(g.lower() for g in _GRADE_VOCAB)


def _normalise_grade(s: str) -> str:
    s = _GRADE_DASHES.sub("-", s.strip().lower())
    return _GRADE_ROMAN_RE.sub(lambda m: _GRADE_ROMAN[m.group(1)], s)


def parse_json(raw_text: str) -> Optional[dict]:
    text = raw_text.strip()
    if "```" in text:
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    if start == -1:
        return None
    try:
        return json.loads(text[start:].strip())
    except json.JSONDecodeError:
        return None


def validate_schema(data: dict) -> Optional[PathologyExtraction]:
    data = dict(data)
    # Force current schema version regardless of what the LLM outputs
    data["schema_version"] = SCHEMA_VERSION
    # hallucination_flags coercion now lives in the schema's field_validator.
    try:
        return PathologyExtraction.model_validate(data)
    except ValidationError:
        return None


def build_correction_message(error: ValidationError) -> str:
    """Build a model-readable correction prompt from a Pydantic ValidationError.

    Lists each offending field and what was wrong, so a local model's retry can
    target the specific problem rather than receiving a generic instruction.
    """
    lines = [
        f"- {'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
        for err in error.errors()
    ]
    fields = "\n".join(lines)
    return (
        "\n\nYour previous response did not match the required schema:\n"
        f"{fields}\n"
        "Please output ONLY a corrected JSON object that fixes these fields, "
        "starting with { and ending with }, no markdown."
    )


def check_vocab(record: PathologyExtraction) -> list[str]:
    flags: list[str] = []
    if record.tnm_stage:
        if record.tnm_stage.T and not _TNM_T.match(record.tnm_stage.T):
            flags.append(f"invalid_tnm_T: {record.tnm_stage.T!r}")
        if record.tnm_stage.N and not _TNM_N.match(record.tnm_stage.N):
            flags.append(f"invalid_tnm_N: {record.tnm_stage.N!r}")
        if record.tnm_stage.M and not _TNM_M.match(record.tnm_stage.M):
            flags.append(f"invalid_tnm_M: {record.tnm_stage.M!r}")
    if record.tumor_grade and _normalise_grade(record.tumor_grade) not in _GRADE_VOCAB_LOWER:
        flags.append(f"unrecognised_grade: {record.tumor_grade!r}")
    return flags


def validate_record(
    raw_text: str,
    report_id: str,
    cancer_type: str,
) -> tuple[Optional[PathologyExtraction], str, list[str]]:
    data = parse_json(raw_text)
    if data is None:
        return None, "json_failed", []

    data["report_id"] = report_id
    data["cancer_type"] = cancer_type

    record = validate_schema(data)
    if record is None:
        return None, "schema_failed", []

    vocab_flags = check_vocab(record)
    if vocab_flags:
        existing = record.hallucination_flags or []
        record = record.model_copy(update={"hallucination_flags": existing + vocab_flags})
        return record, "vocab_flagged", vocab_flags

    return record, "valid", []
