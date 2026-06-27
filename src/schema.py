"""
Pydantic v2 output schema for the pathology report extraction pipeline.

Version: v1.3
Fields: Group A (patient identification) + Group B (tumour characterization).
Group C (margin status, LVI, tumour size, IHC) is deferred to iteration 2.

v1.3 adds coerce-safe enum constraints on `sex` and `tnm_stage.confidence`, moves
`hallucination_flags` coercion from the validator into the schema, and introduces
`LLMExtraction` — the model-facing subset used as a structured-output response_format.
"""

from typing import Literal, Optional

from pydantic import BaseModel, field_validator

SCHEMA_VERSION = "v1.3"


class TNMStage(BaseModel):
    """AJCC/UICC TNM staging components extracted from the report."""

    T: Optional[str] = None
    N: Optional[str] = None
    M: Optional[str] = None
    edition: Optional[str] = None  # e.g. "AJCC8" — too unstable for a Literal
    confidence: Optional[Literal["stated", "inferred"]] = None

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalise_confidence(cls, v):
        """Normalise case/whitespace; coerce any unrecognised value to None.

        Never raises — an off-vocabulary confidence must not fail the whole record.
        """
        if v is None:
            return None
        s = str(v).strip().lower()
        return s if s in {"stated", "inferred"} else None


class PatientIdentification(BaseModel):
    """Group A — patient identification fields.

    Per EDA findings: sex is conditionally present in TCGA reports.
    date_of_birth confirmed absent in TCGA dataset — removed in v1.1.
    institution_id and clinical_process_number removed in v1.2: institution_id
    is joined from corpus metadata post-extraction; clinical_process_number is
    not present in TCGA reports.
    """

    sex: Optional[Literal["male", "female"]] = None

    @field_validator("sex", mode="before")
    @classmethod
    def _normalise_sex(cls, v):
        """Normalise common model shortcuts to the allowed enum.

        Maps "M"/"F" → "male"/"female", lowercases, and coerces any value outside
        the allowed set to None. Never raises — preserves the rest of the record.
        """
        if v is None:
            return None
        s = str(v).strip().lower()
        mapping = {
            "m": "male",
            "male": "male",
            "f": "female",
            "female": "female",
        }
        return mapping.get(s)  # unrecognised → None


class LLMExtraction(BaseModel):
    """Model-facing output contract — exactly the fields the LLM is asked to produce.

    Deliberately excludes report_id, cancer_type, and schema_version, which are
    injected by the pipeline post-extraction (backlog item 6). Used as the
    structured-output response_format for the OpenAI provider so the model is never
    asked to emit those fields. `PathologyExtraction` extends this with the injected
    fields, so it is a strict superset.
    """

    # Group A — Patient identification
    patient: Optional[PatientIdentification] = None

    # Group B — Tumour characterization (Priority 1)
    primary_site: Optional[str] = None
    histological_diagnosis: Optional[str] = None
    histological_subtype: Optional[str] = None
    tumor_grade: Optional[str] = None
    tnm_stage: Optional[TNMStage] = None
    specimen_type: Optional[str] = None

    # Pipeline metadata
    extraction_notes: Optional[str] = None
    hallucination_flags: Optional[list[str]] = None

    @field_validator("hallucination_flags", mode="before")
    @classmethod
    def _coerce_flags(cls, v):
        """Coerce flags to a list of strings; never raise.

        A plain string becomes a single-element list; list elements are coerced to
        str; any other type is stringified and wrapped so no flag is silently dropped.
        """
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(x) for x in v]
        return [str(v)]


class PathologyExtraction(LLMExtraction):
    """Top-level extraction output — one record per pathology report.

    Every field is Optional to accommodate the full range of report styles
    (synoptic, narrative, mixed) and allow partial extractions rather than
    failing entirely. report_id and cancer_type are NOT extracted by the model
    — they are injected by the pipeline from input metadata after parsing.
    """

    schema_version: str = SCHEMA_VERSION

    # Identifiers — injected post-extraction from pipeline metadata, not from LLM output
    report_id: str
    cancer_type: str


def null_skeleton() -> str:
    """Pretty-printed JSON skeleton of the model-facing schema with all fields null.

    Injected into the prompt at `<<output_skeleton>>` so the example structure can
    never drift from the schema. Nested models are instantiated explicitly so
    `patient` and `tnm_stage` render as nested null objects rather than a bare null.
    """
    return LLMExtraction(
        patient=PatientIdentification(),
        tnm_stage=TNMStage(),
    ).model_dump_json(indent=2)
