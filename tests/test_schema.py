"""Unit tests for src/schema.py — PathologyExtraction v1.3."""

import json

import pytest
from pydantic import ValidationError

from src.schema import (
    SCHEMA_VERSION,
    LLMExtraction,
    PathologyExtraction,
    PatientIdentification,
    TNMStage,
    null_skeleton,
)


class TestSchemaVersion:
    def test_schema_version_constant(self):
        assert SCHEMA_VERSION == "v1.3"

    def test_default_schema_version_on_model(self):
        record = PathologyExtraction(report_id="TCGA-XX-0001", cancer_type="BRCA")
        assert record.schema_version == SCHEMA_VERSION


class TestTNMStage:
    def test_all_none_optional_fields_accepted(self):
        tnm = TNMStage()
        assert tnm.T is None
        assert tnm.N is None
        assert tnm.M is None
        assert tnm.edition is None
        assert tnm.confidence is None

    def test_full_tnm_parses(self):
        tnm = TNMStage(T="pT2", N="pN1", M="pM0", edition="AJCC8", confidence="stated")
        assert tnm.T == "pT2"
        assert tnm.edition == "AJCC8"

    @pytest.mark.parametrize("field", ["T", "N", "M"])
    def test_wrong_type_on_tnm_field_raises(self, field):
        with pytest.raises(ValidationError):
            TNMStage(**{field: 42})


class TestPatientIdentification:
    def test_all_none_accepted(self):
        p = PatientIdentification()
        assert p.sex is None

    def test_sex_populated(self):
        p = PatientIdentification(sex="female")
        assert p.sex == "female"


class TestPathologyExtraction:
    def test_minimal_valid_record(self):
        record = PathologyExtraction(report_id="TCGA-BH-A0B3", cancer_type="BRCA")
        assert record.report_id == "TCGA-BH-A0B3"
        assert record.cancer_type == "BRCA"
        assert record.schema_version == SCHEMA_VERSION
        assert record.tnm_stage is None
        assert record.patient is None

    def test_full_record_parses(self):
        record = PathologyExtraction(
            report_id="TCGA-BH-A0B3",
            cancer_type="BRCA",
            patient=PatientIdentification(sex="female"),
            primary_site="left breast",
            histological_diagnosis="invasive ductal carcinoma",
            histological_subtype="not otherwise specified",
            tumor_grade="Grade 2",
            tnm_stage=TNMStage(T="pT2", N="pN1", M="pM0", edition="AJCC8", confidence="stated"),
            specimen_type="mastectomy",
            extraction_notes="TNM stated explicitly in synoptic section.",
            hallucination_flags=[],
        )
        assert record.tnm_stage.T == "pT2"
        assert record.patient.sex == "female"
        assert record.hallucination_flags == []

    def test_all_optional_fields_none(self):
        record = PathologyExtraction(report_id="TCGA-XX-0001", cancer_type="LUAD")
        for field in [
            "patient",
            "primary_site",
            "histological_diagnosis",
            "histological_subtype",
            "tumor_grade",
            "tnm_stage",
            "specimen_type",
            "extraction_notes",
            "hallucination_flags",
        ]:
            assert getattr(record, field) is None

    def test_model_validate_from_dict(self):
        data = {
            "report_id": "TCGA-L9-A50W",
            "cancer_type": "LUSC",
            "tnm_stage": {"T": "pT3", "N": "pN0", "M": None},
        }
        record = PathologyExtraction.model_validate(data)
        assert record.tnm_stage.T == "pT3"
        assert record.tnm_stage.M is None

    def test_serialises_to_dict(self):
        record = PathologyExtraction(report_id="TCGA-BH-A0B3", cancer_type="BRCA")
        d = record.model_dump()
        assert d["schema_version"] == SCHEMA_VERSION
        assert "tnm_stage" in d


class TestSexEnum:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("M", "male"),
            ("f", "female"),
            ("  FEMALE ", "female"),
            ("Male", "male"),
            ("unknown", None),
            (None, None),
            ("", None),
            ("intersex", None),  # off-vocab → coerced to None, never raises
            ("n/a", None),
        ],
    )
    def test_normalises_and_coerces(self, raw, expected):
        p = PatientIdentification(sex=raw)
        assert p.sex == expected

    def test_off_vocab_sex_does_not_drop_record(self):
        # An unrecognised sex must null only that field, leaving the record valid.
        record = PathologyExtraction.model_validate(
            {"report_id": "TCGA-XX-0001", "cancer_type": "BRCA", "patient": {"sex": "intersex"}, "primary_site": "left breast"}
        )
        assert record.patient.sex is None
        assert record.primary_site == "left breast"


class TestConfidenceEnum:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("stated", "stated"),
            ("  STATED ", "stated"),
            ("Inferred", "inferred"),
            (None, None),
            ("assumed", None),  # off-vocab → None
            ("", None),
        ],
    )
    def test_normalises_and_coerces(self, raw, expected):
        tnm = TNMStage(confidence=raw)
        assert tnm.confidence == expected


class TestHallucinationFlagsCoercion:
    def test_plain_string_becomes_list(self):
        rec = LLMExtraction(hallucination_flags="one flag")
        assert rec.hallucination_flags == ["one flag"]

    def test_list_elements_coerced_to_str(self):
        rec = LLMExtraction(hallucination_flags=["a", 2, 3.5])
        assert rec.hallucination_flags == ["a", "2", "3.5"]

    def test_none_stays_none(self):
        assert LLMExtraction(hallucination_flags=None).hallucination_flags is None

    def test_non_list_non_str_is_stringified(self):
        assert LLMExtraction(hallucination_flags={"x": 1}).hallucination_flags == [str({"x": 1})]


class TestLLMExtraction:
    def test_excludes_injected_fields(self):
        fields = set(LLMExtraction.model_fields)
        assert "report_id" not in fields
        assert "cancer_type" not in fields
        assert "schema_version" not in fields

    def test_is_subset_of_pathology_extraction(self):
        assert set(LLMExtraction.model_fields).issubset(set(PathologyExtraction.model_fields))

    def test_group_b_fields_present(self):
        fields = set(LLMExtraction.model_fields)
        for f in [
            "primary_site",
            "histological_diagnosis",
            "histological_subtype",
            "tumor_grade",
            "tnm_stage",
            "specimen_type",
        ]:
            assert f in fields


class TestNullSkeleton:
    def test_renders_nested_nulls(self):
        skeleton = json.loads(null_skeleton())
        assert skeleton["patient"] == {"sex": None}
        assert skeleton["tnm_stage"]["confidence"] is None
        assert skeleton["tnm_stage"]["T"] is None

    def test_omits_injected_fields(self):
        skeleton = json.loads(null_skeleton())
        assert "schema_version" not in skeleton
        assert "report_id" not in skeleton
        assert "cancer_type" not in skeleton
