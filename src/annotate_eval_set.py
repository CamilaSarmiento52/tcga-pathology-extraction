"""
Automated annotation of the eval set using Claude via aisuite.

Usage:
    uv run python src/annotate_eval_set.py                # annotate all remaining records
    uv run python src/annotate_eval_set.py --retry-errors # re-run only previously failed records
    uv run python src/annotate_eval_set.py --limit 2      # smoke test on first N records

Reads  : data/annotations/eval_set_sampling_frame_v1.jsonl
         data/annotations/few_shot_examples_v1.3.jsonl   (pathologist-corrected few-shot)
Writes : data/annotations/eval_set_v2.jsonl  (validated, Pydantic-checked, schema v1.2)

This is the second annotation campaign: prompt extraction_v1.4 (NST/lobular convention)
with the pathologist-corrected few-shot examples (v1.3) injected for guidance. The prior
eval_set_v1 / v1.1 / v1.2 files were a single zero-shot run (prompt v1.1) plus schema
migrations; this run regenerates labels under the corrected convention.

Resumes automatically if interrupted — already-annotated records are skipped.
--retry-errors loads report_ids from the error log and runs only those.

NOTE (academic): Ground truth is LLM-generated (Claude), not human expert-annotated.
Accuracy metrics reflect agreement with Claude annotations, not clinical ground truth.
NOTE (bias): The same corrected few-shot examples guide both this ground truth and the
models evaluated against it — a shared-teacher bias to document as a known limitation.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aisuite as ai
from dotenv import load_dotenv
from pydantic import ValidationError
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent
SAMPLING_FRAME = ROOT / "data/annotations/eval_set_sampling_frame_v1.jsonl"
JSONL_OUT      = ROOT / "data/annotations/eval_set_v2.jsonl"
ERROR_LOG      = ROOT / "data/annotations/annotation_errors_v2.json"
PROMPT_FILE    = ROOT / "src/prompts/extraction_v1.4.txt"
FEW_SHOT_FILE  = ROOT / "data/annotations/few_shot_examples_v1.3.jsonl"

# ── config ──────────────────────────────────────────────────────────────────
MODEL          = "anthropic:claude-sonnet-4-6"
TEMPERATURE    = 0.0
MAX_TOKENS     = 2500
RETRY_ATTEMPTS = 2
SLEEP_BETWEEN  = 0.5   # seconds between calls — avoids rate limits

sys.path.insert(0, str(ROOT / "src"))
from schema import PathologyExtraction  # noqa: E402
from pipeline.prompt_constructor import build_prompt  # noqa: E402

load_dotenv(ROOT / ".env")


# ── helpers ──────────────────────────────────────────────────────────────────
def parse_llm_response(response_text: str) -> dict:
    """Extract JSON from LLM response — handles markdown code fences."""
    text = response_text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def load_already_annotated() -> set:
    """Return set of report_ids already saved to JSONL — enables resume."""
    if not JSONL_OUT.exists():
        return set()
    done = set()
    with open(JSONL_OUT) as f:
        for line in f:
            done.add(json.loads(line)["report_id"])
    return done


def load_records() -> list[dict]:
    records = []
    with open(SAMPLING_FRAME) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_failed_ids() -> set:
    """Return report_ids that previously failed, from the error log."""
    if not ERROR_LOG.exists():
        return set()
    with open(ERROR_LOG) as f:
        errors = json.load(f)
    return {e["report_id"] for e in errors}


def coerce_hallucination_flags(raw_dict: dict) -> dict:
    """If the model returned hallucination_flags as a plain string, wrap it in a list."""
    flags = raw_dict.get("hallucination_flags")
    if isinstance(flags, str):
        raw_dict["hallucination_flags"] = [flags]
    return raw_dict


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Re-run only the records listed in the error log",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke test: only annotate the first N remaining records",
    )
    args = parser.parse_args()

    records      = load_records()
    already_done = load_already_annotated()

    if args.retry_errors:
        failed_ids = load_failed_ids()
        if not failed_ids:
            print(f"No failed records found in {ERROR_LOG.name} — nothing to retry.")
            return
        # Only retry failed records, even if they somehow ended up in the success file
        remaining = [r for r in records if r["report_id"] in failed_ids]
        print(f"Retrying {len(remaining)} previously failed records.")
    else:
        remaining = [r for r in records if r["report_id"] not in already_done]

    if args.limit is not None:
        remaining = remaining[: args.limit]

    print(f"Total eval records : {len(records)}")
    print(f"Already annotated  : {len(already_done)}")
    print(f"Remaining          : {len(remaining)}")
    print(f"Model              : {MODEL}")
    print(f"Prompt             : {PROMPT_FILE.name}")
    print(f"Few-shot           : {FEW_SHOT_FILE.name}")
    print(f"Output             : {JSONL_OUT.name}\n")

    if not remaining:
        print(f"All records already annotated — {JSONL_OUT.name} is ready.")
        return

    client        = ai.Client()
    valid_count   = 0
    invalid_count = 0
    error_log     = []

    with open(JSONL_OUT, "a") as out_f:
        for record in tqdm(remaining, desc="Annotating"):
            report_id   = record["report_id"]
            cancer_type = record["cancer_type"]
            prompt, _   = build_prompt(record, PROMPT_FILE, FEW_SHOT_FILE)
            raw_response = None

            for attempt in range(1, RETRY_ATTEMPTS + 1):
                try:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                    raw_response = response.choices[0].message.content
                    raw_dict     = parse_llm_response(raw_response)

                    # Inject identifiers, coerce types, and validate against schema
                    raw_dict["report_id"]   = report_id
                    raw_dict["cancer_type"] = cancer_type
                    raw_dict = coerce_hallucination_flags(raw_dict)
                    validated = PathologyExtraction.model_validate(raw_dict)

                    # Write to JSONL — drop pipeline metadata fields, not needed in ground truth
                    out = validated.model_dump()
                    out.pop("extraction_notes", None)
                    out.pop("hallucination_flags", None)
                    out["annotated_at"]    = datetime.now(timezone.utc).isoformat()
                    out["annotated_by"]    = MODEL
                    out["annotation_note"] = "LLM-generated ground truth (academic use)"
                    out_f.write(json.dumps(out) + "\n")
                    out_f.flush()

                    valid_count += 1
                    break  # success — next record

                except (json.JSONDecodeError, ValidationError, Exception) as e:
                    if attempt == RETRY_ATTEMPTS:
                        invalid_count += 1
                        error_log.append({
                            "report_id": report_id,
                            "error":     str(e),
                            "response":  raw_response,
                        })
                        tqdm.write(f"  FAILED: {report_id[:50]} — {e}")

            time.sleep(SLEEP_BETWEEN)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Done.  Valid: {valid_count}  |  Failed: {invalid_count}")
    print(f"Output: {JSONL_OUT}")

    if error_log:
        with open(ERROR_LOG, "w") as f:
            json.dump(error_log, f, indent=2)
        print(f"Errors: {ERROR_LOG}")


if __name__ == "__main__":
    main()
