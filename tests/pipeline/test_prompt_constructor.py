import json
from pathlib import Path

import pytest
import tiktoken

from src.pipeline.prompt_constructor import (
    build_prompt,
    count_tokens,
    truncate_to_tokens,
)

TEMPLATE_PATH = Path(__file__).parent.parent.parent / "src" / "prompts" / "extraction_v1.2.txt"
TEMPLATE_PATH_V17 = Path(__file__).parent.parent.parent / "src" / "prompts" / "extraction_v1.7.txt"
PLACEHOLDER = "<<report_text>>"
TOKEN_BUDGET = 1_500  # max tokens for the template before adding report text


@pytest.fixture
def sample_record():
    return {
        "report_id": "TCGA-BH-A0B3.abc123",
        "cancer_type": "BRCA",
        "text": "SPECIMEN: Left breast mastectomy. DIAGNOSIS: Invasive ductal carcinoma, Grade 2.",
    }


@pytest.fixture
def tmp_few_shot(tmp_path):
    examples = [
        {
            "report_id": f"TCGA-XX-{i:04d}.abc",
            "text": f"Report text {i}",
            "ground_truth": {
                "schema_version": "v1.1",
                "report_id": f"TCGA-XX-{i:04d}.abc",
                "cancer_type": "BRCA",
            },
        }
        for i in range(3)
    ]
    p = tmp_path / "few_shot.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in examples))
    return p


class TestTemplateFile:
    def test_file_exists(self):
        assert TEMPLATE_PATH_V17.exists(), f"Prompt template not found at {TEMPLATE_PATH_V17}"

    def test_placeholder_present_exactly_once(self):
        template = TEMPLATE_PATH_V17.read_text(encoding="utf-8")
        count = template.count(PLACEHOLDER)
        assert count == 1, f"Expected 1 occurrence of '{PLACEHOLDER}', found {count}"

    def test_group_b_fields_mentioned(self):
        template = TEMPLATE_PATH_V17.read_text(encoding="utf-8")
        for term in [
            "primary_site",
            "histological_diagnosis",
            "histological_subtype",
            "tumor_grade",
            "tnm_stage",
            "specimen_type",
        ]:
            assert term in template, f"Group B field '{term}' not found in prompt template"

    def test_date_of_birth_not_present(self):
        template = TEMPLATE_PATH_V17.read_text(encoding="utf-8")
        assert "date_of_birth" not in template, "date_of_birth should have been removed in v1.2"

    def test_null_handling_rule_present(self):
        template = TEMPLATE_PATH_V17.read_text(encoding="utf-8")
        assert "null" in template.lower()

    def test_chain_of_thought_instruction_present(self):
        template = TEMPLATE_PATH_V17.read_text(encoding="utf-8")
        lower = template.lower()
        assert "step" in lower or "reason" in lower

    def test_template_token_count_within_budget(self):
        template = TEMPLATE_PATH_V17.read_text(encoding="utf-8")
        template_without_placeholder = template.replace(PLACEHOLDER, "").replace("<<output_skeleton>>", "")
        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(template_without_placeholder))
        assert token_count < TOKEN_BUDGET, (
            f"Template uses {token_count} tokens (budget: {TOKEN_BUDGET}). "
            "Reduce template length to leave room for report text."
        )


class TestCountTokens:
    def test_non_zero_for_text(self):
        assert count_tokens("Hello world") > 0

    def test_empty_string(self):
        assert count_tokens("") == 0


class TestTruncateToTokens:
    def test_short_text_unchanged(self):
        text = "Short text."
        result, truncated = truncate_to_tokens(text, max_tokens=100)
        assert result == text
        assert not truncated

    def test_long_text_truncated(self):
        text = " ".join(["word"] * 10000)
        result, truncated = truncate_to_tokens(text, max_tokens=100)
        assert truncated
        assert count_tokens(result) <= 100


class TestBuildPrompt:
    def test_report_text_injected(self, sample_record):
        prompt, _ = build_prompt(sample_record, TEMPLATE_PATH)
        assert "<<report_text>>" not in prompt
        assert sample_record["text"] in prompt

    def test_report_id_not_in_prompt(self, sample_record):
        prompt, _ = build_prompt(sample_record, TEMPLATE_PATH)
        assert sample_record["report_id"] not in prompt

    def test_cancer_type_not_in_prompt(self, sample_record):
        prompt, _ = build_prompt(sample_record, TEMPLATE_PATH)
        assert sample_record["cancer_type"] not in prompt

    def test_few_shot_block_included(self, sample_record, tmp_few_shot):
        prompt, _ = build_prompt(sample_record, TEMPLATE_PATH, few_shot_path=tmp_few_shot)
        assert "FEW-SHOT EXAMPLES" in prompt
        assert "Example 1" in prompt

    def test_truncation_flagged_for_long_text(self):
        long_record = {
            "report_id": "TCGA-XX-0001.abc",
            "cancer_type": "LUAD",
            "text": " ".join(["word"] * 10000),
        }
        _, truncated = build_prompt(long_record, TEMPLATE_PATH)
        assert truncated

    def test_truncation_caps_report_at_7800_tokens(self):
        long_record = {
            "report_id": "TCGA-XX-0001.abc",
            "cancer_type": "LUAD",
            "text": " ".join(["word"] * 10000),
        }
        result, truncated = truncate_to_tokens(long_record["text"], max_tokens=7800)
        assert truncated
        assert count_tokens(result) <= 7800


class TestSkeletonInjection:
    def test_v17_skeleton_placeholder_replaced(self, sample_record):
        prompt, _ = build_prompt(sample_record, TEMPLATE_PATH_V17)
        assert "<<output_skeleton>>" not in prompt
        assert '"tnm_stage"' in prompt
        assert '"confidence"' in prompt
        assert '"schema_version"' not in prompt  # injected fields excluded from skeleton

    def test_v12_without_placeholder_is_noop(self, sample_record):
        prompt, _ = build_prompt(sample_record, TEMPLATE_PATH)
        assert "<<output_skeleton>>" not in prompt
