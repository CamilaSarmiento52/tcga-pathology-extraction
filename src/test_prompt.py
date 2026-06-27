"""
Quick prompt smoke-test: run the current extraction prompt against N few-shot
records and print extracted output vs ground truth side by side.

Uses the same pipeline code paths as a real run — prompt_constructor.build_prompt
(handles the <<report_text>> / <<output_skeleton>> placeholders) and
model_caller.call_model (openai SDK, provider-prefixed model strings) — so the
smoke-test never drifts from production behaviour.

Usage:
    uv run python src/test_prompt.py                      # first 3 records
    uv run python src/test_prompt.py --n 5                # first N records
    uv run python src/test_prompt.py --all                # all records
    uv run python src/test_prompt.py --model openai:o4-mini
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

ROOT        = Path(__file__).parent.parent
PROMPT_FILE = ROOT / "src/prompts/extraction_v1.7.txt"
FEW_SHOT    = ROOT / "data/annotations/few_shot_examples_v1.4.jsonl"
MODEL       = "openai:o4-mini"
MAX_TOKENS  = 4096

# Run as a script (`uv run python src/test_prompt.py`) but still import the
# pipeline as the `src` package — put the project root on the path.
sys.path.insert(0, str(ROOT))
from src.pipeline.model_caller import call_model  # noqa: E402
from src.pipeline.prompt_constructor import build_prompt  # noqa: E402
from src.pipeline.validator import parse_json  # noqa: E402
from src.schema import PathologyExtraction  # noqa: E402

load_dotenv(ROOT / ".env")

COMPARE_FIELDS = [
    "primary_site",
    "histological_diagnosis",
    "histological_subtype",
    "tumor_grade",
    "specimen_type",
]
TNM_FIELDS = ["T", "N", "M", "edition", "confidence"]


def fmt(val) -> str:
    if val is None:
        return "null"
    return str(val)


def print_diff(extracted: dict, ground_truth: dict, record_n: int, report_id: str):
    print(f"\n{'='*70}")
    print(f"Record {record_n}: {report_id}")
    print(f"{'='*70}")

    gt_patient  = ground_truth.get("patient") or {}
    ext_patient = extracted.get("patient") or {}

    print(f"\n{'Field':<30} {'Extracted':<30} {'Ground Truth':<30}")
    print(f"{'-'*30} {'-'*30} {'-'*30}")

    print(f"{'patient.sex':<30} {fmt(ext_patient.get('sex')):<30} {fmt(gt_patient.get('sex')):<30}")

    for field in COMPARE_FIELDS:
        match = "✓" if fmt(extracted.get(field)) == fmt(ground_truth.get(field)) else "✗"
        print(f"{field:<30} {fmt(extracted.get(field)):<30} {fmt(ground_truth.get(field)):<30} {match}")

    gt_tnm  = ground_truth.get("tnm_stage") or {}
    ext_tnm = extracted.get("tnm_stage") or {}
    for sub in TNM_FIELDS:
        field = f"tnm_stage.{sub}"
        match = "✓" if fmt(ext_tnm.get(sub)) == fmt(gt_tnm.get(sub)) else "✗"
        print(f"{field:<30} {fmt(ext_tnm.get(sub)):<30} {fmt(gt_tnm.get(sub)):<30} {match}")

    flags = extracted.get("hallucination_flags")
    if flags:
        print(f"\nHallucination flags:")
        for f in flags:
            print(f"  • {f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="Number of records to test")
    parser.add_argument("--all", action="store_true", help="Test all records")
    parser.add_argument("--model", default=MODEL, help="Provider-prefixed model string (e.g. openai:o4-mini)")
    args = parser.parse_args()

    records = []
    with open(FEW_SHOT) as f:
        for line in f:
            records.append(json.loads(line))

    if not args.all:
        records = records[: args.n]

    print(f"Prompt : {PROMPT_FILE.name}")
    print(f"Model  : {args.model}")
    print(f"Records: {len(records)}")

    passed = 0
    failed = 0

    for i, record in enumerate(records, 1):
        report_id   = record["report_id"]
        cancer_type = record["cancer_type"]
        ground_truth = record["ground_truth"]

        prompt, _ = build_prompt(record, PROMPT_FILE)

        try:
            response = call_model(prompt, args.model, max_tokens=MAX_TOKENS)
            raw_dict = parse_json(response.raw_text)
            if raw_dict is None:
                raise json.JSONDecodeError("no JSON object found", response.raw_text, 0)
            raw_dict["report_id"]   = report_id
            raw_dict["cancer_type"] = cancer_type
            validated = PathologyExtraction.model_validate(raw_dict)
            extracted = validated.model_dump()
            print_diff(extracted, ground_truth, i, report_id)
            passed += 1
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"\nRecord {i} ({report_id}): SCHEMA ERROR — {e}")
            failed += 1
        except Exception as e:
            print(f"\nRecord {i} ({report_id}): ERROR — {e}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"Results: {passed} passed, {failed} failed")


if __name__ == "__main__":
    main()
