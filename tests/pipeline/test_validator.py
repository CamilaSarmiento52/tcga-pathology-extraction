import pytest

from pydantic import ValidationError

from src.pipeline.validator import (
    VALIDATOR_VERSION,
    build_correction_message,
    check_vocab,
    parse_json,
    validate_record,
    validate_schema,
)
from src.schema import LLMExtraction
from src.schema import PathologyExtraction, TNMStage


class TestParseJson:
    def test_valid_json(self):
        result = parse_json('{"report_id": "x", "cancer_type": "BRCA"}')
        assert result == {"report_id": "x", "cancer_type": "BRCA"}

    def test_strips_markdown_fence(self):
        raw = '```json\n{"report_id": "x", "cancer_type": "BRCA"}\n```'
        assert parse_json(raw)["report_id"] == "x"

    def test_not_json_returns_none(self):
        assert parse_json("I cannot extract this.") is None

    def test_empty_string_returns_none(self):
        assert parse_json("") is None

    def test_json_embedded_in_text(self):
        raw = 'Here is the result:\n{"report_id": "x", "cancer_type": "BRCA"}'
        result = parse_json(raw)
        assert result["report_id"] == "x"


class TestValidateSchema:
    def test_valid_minimal_record(self):
        data = {"report_id": "TCGA-XX-0001", "cancer_type": "BRCA"}
        result = validate_schema(data)
        assert isinstance(result, PathologyExtraction)
        assert result.schema_version == "v1.3"

    def test_forces_current_schema_version(self):
        data = {"report_id": "TCGA-XX-0001", "cancer_type": "BRCA", "schema_version": "v1.0"}
        result = validate_schema(data)
        assert result is not None
        assert result.schema_version == "v1.3"

    def test_missing_report_id_fails(self):
        result = validate_schema({"cancer_type": "BRCA"})
        assert result is None

    def test_coerces_hallucination_flags_string_to_list(self):
        data = {"report_id": "x", "cancer_type": "BRCA", "hallucination_flags": "some flag"}
        result = validate_schema(data)
        assert result.hallucination_flags == ["some flag"]



class TestCheckVocab:
    def test_valid_tnm_no_flags(self):
        record = PathologyExtraction(
            report_id="x", cancer_type="BRCA", tnm_stage=TNMStage(T="pT2", N="pN1", M="pM0")
        )
        assert check_vocab(record) == []

    def test_invalid_t_stage_flagged(self):
        record = PathologyExtraction(
            report_id="x", cancer_type="BRCA", tnm_stage=TNMStage(T="banana")
        )
        flags = check_vocab(record)
        assert any("invalid_tnm_T" in f for f in flags)

    def test_no_tnm_no_flags(self):
        record = PathologyExtraction(report_id="x", cancer_type="BRCA")
        assert check_vocab(record) == []

    def test_tx_is_valid(self):
        record = PathologyExtraction(
            report_id="x", cancer_type="BRCA", tnm_stage=TNMStage(T="TX", N="NX", M="MX")
        )
        assert check_vocab(record) == []


class TestValidatorVersion:
    def test_version_string(self):
        assert VALIDATOR_VERSION == "v1.3"


class TestBuildCorrectionMessage:
    def test_names_offending_field(self):
        try:
            LLMExtraction.model_validate({"tnm_stage": {"T": 42}})
        except ValidationError as e:
            msg = build_correction_message(e)
            assert "tnm_stage.T" in msg
            assert "schema" in msg.lower()
        else:
            pytest.fail("expected a ValidationError for tnm_stage.T = 42")


class TestCheckVocabExpanded:
    """Tests for expanded TNM regex and grade vocab added in validator v1.1."""

    def _record(self, T=None, N=None, M=None, grade=None):
        tnm = TNMStage(T=T, N=N, M=M) if (T or N or M) else None
        return PathologyExtraction(report_id="x", cancer_type="BRCA", tnm_stage=tnm, tumor_grade=grade)

    # TNM N-stage: sentinel node and micrometastasis variants
    def test_n0_sn_valid(self):
        assert check_vocab(self._record(N="pN0(sn)")) == []

    def test_n1mi_valid(self):
        assert check_vocab(self._record(N="pN1mi")) == []

    def test_n0_i_minus_valid(self):
        assert check_vocab(self._record(N="pN0(i-)")) == []

    def test_n0_i_plus_valid(self):
        assert check_vocab(self._record(N="pN0(i+)")) == []

    def test_n0_i_minus_sn_valid(self):
        assert check_vocab(self._record(N="pN0(i-)(sn)")) == []

    # TNM T-stage: multi-lesion variant
    def test_t4b_mult_valid(self):
        assert check_vocab(self._record(T="pT4b(mult)")) == []

    # Grade: expanded vocab
    def test_grade_nhg2_valid(self):
        assert check_vocab(self._record(grade="NHG2")) == []

    def test_grade_range_valid(self):
        assert check_vocab(self._record(grade="Grade 2-3")) == []

    def test_grade_low_iaslc_valid(self):
        assert check_vocab(self._record(grade="Low")) == []

    def test_grade_moderate_to_poorly_valid(self):
        assert check_vocab(self._record(grade="moderate to poorly differentiated")) == []

    # Still invalid
    def test_pn01_invalid(self):
        flags = check_vocab(self._record(N="pN01"))
        assert any("invalid_tnm_N" in f for f in flags)

    def test_garbage_grade_invalid(self):
        flags = check_vocab(self._record(grade="unknown grading"))
        assert any("unrecognised_grade" in f for f in flags)


class TestValidateRecord:
    def test_valid_json_and_schema(self):
        raw = '{"histological_diagnosis": "invasive ductal carcinoma"}'
        record, status, flags = validate_record(raw, "TCGA-XX-0001", "BRCA")
        assert record is not None
        assert status == "valid"
        assert flags == []

    def test_json_failed(self):
        record, status, _ = validate_record("not json at all", "TCGA-XX-0001", "BRCA")
        assert record is None
        assert status == "json_failed"

    def test_schema_failed(self):
        # A wrong type for a field (tnm_stage as a plain string) must fail schema validation
        record, status, _ = validate_record('{"tnm_stage": "not-an-object"}', "TCGA-XX-0001", "BRCA")
        assert record is None
        assert status == "schema_failed"

    def test_vocab_flagged_still_returns_record(self):
        raw = '{"tnm_stage": {"T": "banana", "N": null, "M": null, "edition": null, "confidence": null}}'
        record, status, flags = validate_record(raw, "TCGA-XX-0001", "BRCA")
        assert record is not None
        assert status == "vocab_flagged"
        assert len(flags) > 0
