import hashlib
import re

import pandas as pd

# ── 0. Style classification ───────────────────────────────────────────────────
# TCGA reports are stored as single continuous strings with no embedded
# newlines, so line-anchored patterns (re.MULTILINE + ^) never fire.
# All five patterns use inline anchors / lookbehinds instead.
#
#   _ALLCAPS_COLON  : ALL-CAPS label + colon  — "FINAL DIAGNOSIS:", "SPECIMEN:"
#   _ALLCAPS_PERIOD : ALL-CAPS label + period  — "HISTORY.", "MACROSCOPIC."
#   _TITLECASE_COLON: Title-case label + colon — "Tumor Grade:", "T Stage:"
#   _NUMBERED_PERIOD: Numbered/lettered item   — "1.", "A.", "(a)"
#   _NUMBERED_COLON : Numbered item + colon    — "1: Specimen …"

_STYLE_ALLCAPS_COLON   = re.compile(r"(?<!\w)[A-Z][A-Z\s/\(\)-]{2,35}:")
_STYLE_ALLCAPS_PERIOD  = re.compile(r"(?<!\w)(?:[A-Z]{3,}\s+)*[A-Z]{3,}\.(?=\s)")
_STYLE_TITLECASE_COLON = re.compile(r"(?<!\w)[A-Z][A-Za-z][A-Za-z\s,\(\)/\-]{2,45}:\s")
_STYLE_NUMBERED_PERIOD = re.compile(r"(?<!\w)(?:\d+\.|\([0-9]+\)|[A-Z]\.)\s+[A-Z]")
_STYLE_NUMBERED_COLON  = re.compile(r"(?<!\w)\d+:\s+[A-Z]")


def extract_structural_signals(text: str) -> dict:
    """Return observable text-structure signals for one report.

    Returns a dict with keys:
        allcaps_colon_count, allcaps_period_count, titlecase_colon_count,
        numbered_item_count, total_structure_count, structure_density,
        prose_fraction, max_prose_run, avg_prose_run
    """
    _empty = dict(
        allcaps_colon_count=0, allcaps_period_count=0,
        titlecase_colon_count=0, numbered_item_count=0,
        total_structure_count=0, structure_density=0.0,
        prose_fraction=1.0, max_prose_run=0, avg_prose_run=0.0,
    )
    if not isinstance(text, str) or not text.strip():
        return _empty

    total_words = max(1, len(text.split()))

    allcaps_colon_count   = len(_STYLE_ALLCAPS_COLON.findall(text))
    allcaps_period_count  = len(_STYLE_ALLCAPS_PERIOD.findall(text))
    titlecase_colon_count = len(_STYLE_TITLECASE_COLON.findall(text))
    numbered_item_count   = (len(_STYLE_NUMBERED_PERIOD.findall(text)) +
                             len(_STYLE_NUMBERED_COLON.findall(text)))

    all_spans = sorted(
        set(m.start() for m in _STYLE_ALLCAPS_COLON.finditer(text))   |
        set(m.start() for m in _STYLE_ALLCAPS_PERIOD.finditer(text))  |
        set(m.start() for m in _STYLE_TITLECASE_COLON.finditer(text)) |
        set(m.start() for m in _STYLE_NUMBERED_PERIOD.finditer(text)) |
        set(m.start() for m in _STYLE_NUMBERED_COLON.finditer(text))
    )
    total_structure_count = len(all_spans)
    structure_density     = round(total_structure_count / total_words * 100, 2)

    if total_structure_count == 0:
        return dict(
            allcaps_colon_count=allcaps_colon_count,
            allcaps_period_count=allcaps_period_count,
            titlecase_colon_count=titlecase_colon_count,
            numbered_item_count=numbered_item_count,
            total_structure_count=0,
            structure_density=0.0,
            prose_fraction=1.0,
            max_prose_run=total_words,
            avg_prose_run=float(total_words),
        )

    positions = [0] + all_spans + [len(text)]
    gap_wc = [
        len(text[positions[i]:positions[i + 1]].split())
        for i in range(len(positions) - 1)
    ]
    max_prose_run  = max(gap_wc)
    avg_prose_run  = round(sum(gap_wc) / len(gap_wc), 1)
    prose_fraction = round(sum(g for g in gap_wc if g > 30) / total_words, 3)

    return dict(
        allcaps_colon_count=allcaps_colon_count,
        allcaps_period_count=allcaps_period_count,
        titlecase_colon_count=titlecase_colon_count,
        numbered_item_count=numbered_item_count,
        total_structure_count=total_structure_count,
        structure_density=structure_density,
        prose_fraction=prose_fraction,
        max_prose_run=max_prose_run,
        avg_prose_run=avg_prose_run,
    )


def classify_style_from_signals(signals: dict) -> str:
    """Map a signal dict (from extract_structural_signals) to a style label.

    Score-based decision logic:

      syn_score (0–5): rewards low prose_fraction and high structure_density
        prose_fraction < 0.5   → +2
        prose_fraction < 0.7   → +1
        structure_density ≥ 10 → +2
        structure_density ≥ 5  → +1

      nar_score (0–5): rewards few structural elements and high prose fraction
        total_structure_count < 5 → +2
        prose_fraction ≥ 0.9      → +2
        prose_fraction ≥ 0.8      → +1

      label rules:
        syn_score ≥ 3 AND nar_score = 0  →  "synoptic"
        nar_score ≥ 3 AND syn_score ≤ 1  →  "narrative"
        otherwise                        →  "mixed"
    """
    tc = signals["total_structure_count"]
    pf = signals["prose_fraction"]
    sd = signals["structure_density"]

    syn = (
        int(pf < 0.5) * 2 +
        int(pf < 0.7) +
        int(sd >= 10) * 2 +
        int(sd >= 5)
    )
    nar = (
        int(tc < 5) * 2 +
        int(pf >= 0.9) * 2 +
        int(pf >= 0.8)
    )

    if syn >= 3 and nar == 0:
        return "synoptic"
    if nar >= 3 and syn <= 1:
        return "narrative"
    return "mixed"


def classify_style(text: str) -> str:
    """Classify a single report text as 'synoptic', 'narrative', or 'mixed'."""
    return classify_style_from_signals(extract_structural_signals(text))


# ── 1. has_headers ────────────────────────────────────────────────────────────
_HEADER_RE = re.compile(r"(?<!\w)[A-Z][A-Z\s/\(\)-]{2,35}:")


def detect_headers(text: str, min_headers: int = 2) -> bool:
    if not isinstance(text, str):
        return False
    return len(_HEADER_RE.findall(text)) >= min_headers


def header_detail(text: str) -> dict:
    if not isinstance(text, str):
        return {"section_count": 0, "section_names": []}
    matches = _HEADER_RE.findall(text)
    return {
        "section_count": len(matches),
        "section_names": sorted(set(h.strip().rstrip(":") for h in matches)),
    }


# ── 2. token_bucket ───────────────────────────────────────────────────────────
# very_short (<100 tokens) can be either a true stub OR a dense synoptic
# (e.g. TCGA-A8-* BRCA reports that pack full TNM + grade into one sentence).
# Token count alone cannot distinguish the two — the floor handles exclusion.
#   very_short  < 100  tokens
#   short       100–500
#   medium      500–2,000
#   long        > 2,000
def token_bucket(n: int) -> str:
    if n < 100:
        return "very_short"
    if n < 500:
        return "short"
    if n < 2_000:
        return "medium"
    return "long"


# ── 3. ocr_noise ──────────────────────────────────────────────────────────────
_SPACED_LETTERS_RE = re.compile(r"(?<![A-Z])([A-Z] ){3,}[A-Z]")
_GARBLED_WORDS_RE  = re.compile(r"\b[A-Z]{2,}[a-z][A-Z]{2,}\b")


def detect_ocr_noise(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return (
        len(_SPACED_LETTERS_RE.findall(text)) > 0
        or len(_GARBLED_WORDS_RE.findall(text)) > 2
    )


def ocr_noise_detail(text: str) -> dict:
    if not isinstance(text, str):
        return {"spaced_letters": [], "garbled_words": []}
    return {
        "spaced_letters": [m.strip() for m in _SPACED_LETTERS_RE.findall(text)],
        "garbled_words":  _GARBLED_WORDS_RE.findall(text)[:5],
    }


# ── 4. duplicate_content ──────────────────────────────────────────────────────
def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()) if isinstance(text, str) else ""


def build_duplicate_flags(df: pd.DataFrame, text_col: str = "text") -> pd.Series:
    norm         = df[text_col].apply(_normalise)
    exact_hashes = norm.apply(lambda t: hashlib.sha256(t.encode()).hexdigest())
    near_prefix  = norm.str[:300]
    exact_dup    = exact_hashes.duplicated(keep=False)
    near_dup     = near_prefix.duplicated(keep=False) & ~exact_dup
    return exact_dup | near_dup


# ── 5. tabular_format ─────────────────────────────────────────────────────────
# Detects reports that are spreadsheet rows or database exports rather than
# clinical text. Discovered via TCGA-80-5611 (LUAD, 75 tokens) which contains
# "A description for each data field can be found in the 'Data description
# worksheet (yellow tab)" — a spreadsheet column header, not a report.
_TABULAR_RE = re.compile(
    r"data\s+description\s+worksheet|yellow\s+tab|data\s+field\s+can\s+be\s+found",
    re.IGNORECASE,
)


def detect_tabular_format(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(_TABULAR_RE.search(text))


# ── Apply all flags to a DataFrame ───────────────────────────────────────────
def apply_quality_flags(df: pd.DataFrame, token_count_col: str = "token_count") -> pd.DataFrame:
    """Add all quality and style columns to a corpus DataFrame in one call.

    Adds: has_headers, token_bucket, ocr_noise, duplicate_content,
          tabular_format, style.
    """
    df = df.copy()
    df["has_headers"]       = df["text"].apply(detect_headers)
    df["token_bucket"]      = df[token_count_col].apply(token_bucket)
    df["ocr_noise"]         = df["text"].apply(detect_ocr_noise)
    df["duplicate_content"] = build_duplicate_flags(df)
    df["tabular_format"]    = df["text"].apply(detect_tabular_format)
    df["style"]             = df["text"].apply(classify_style)
    return df
