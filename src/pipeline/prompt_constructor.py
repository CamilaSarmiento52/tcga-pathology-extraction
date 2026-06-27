import json
from pathlib import Path
from typing import Optional

import tiktoken

from src.schema import null_skeleton

_ENCODING = tiktoken.get_encoding("cl100k_base")
MAX_REPORT_TOKENS = 7800
_REPORT_SECTION_MARKER = "PATHOLOGY REPORT:"
_SKELETON_MARKER = "<<output_skeleton>>"


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def truncate_to_tokens(text: str, max_tokens: int = MAX_REPORT_TOKENS) -> tuple[str, bool]:
    tokens = _ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return text, False
    return _ENCODING.decode(tokens[:max_tokens]), True


def format_few_shot(few_shot_path: str | Path) -> str:
    examples = []
    with open(few_shot_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gt = rec.get("ground_truth", {})
            examples.append(
                f"## Example {i}\n"
                f"Report:\n{rec['text']}\n\n"
                f"Output:\n{json.dumps(gt, indent=2)}\n"
            )
    return "\n---\n".join(examples)


def build_prompt(
    record: dict,
    template_path: str | Path,
    few_shot_path: Optional[str | Path] = None,
) -> tuple[str, bool]:
    template = Path(template_path).read_text(encoding="utf-8")

    report_text, was_truncated = truncate_to_tokens(record["text"])

    # Inject the JSON skeleton from the schema (single source of truth). No-op for
    # older templates that hardcode the skeleton and lack the placeholder.
    if _SKELETON_MARKER in template:
        template = template.replace(_SKELETON_MARKER, null_skeleton())

    prompt = template.replace("<<report_text>>", report_text)

    if few_shot_path is not None:
        few_shot_block = format_few_shot(few_shot_path)
        insertion = (
            "---\n\n"
            "FEW-SHOT EXAMPLES (for reference only — do not copy, extract from the report below):\n\n"
            f"{few_shot_block}\n\n"
        )
        prompt = prompt.replace(
            f"\n{_REPORT_SECTION_MARKER}",
            f"\n{insertion}{_REPORT_SECTION_MARKER}",
        )

    return prompt, was_truncated
