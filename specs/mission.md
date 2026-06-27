# Mission

## Goal

Build an LLM extraction pipeline that reads free-text pathology reports and outputs structured clinical registry fields, enabling a Cancer Registry unit to replace manual extraction with an auditable, human-in-the-loop automated system.

## Problem

Manual extraction of structured fields from free-text pathology reports is inconsistent, unscalable, and produces missing or incorrect fields at national registry submission. At 1–3 minutes per report, it cannot scale. The pipeline must match or exceed manual accuracy on critical fields before any production use.

## Dataset

- **Source**: TCGA pathology reports corpus (Kefeli et al., 2024)
- **Volume**: 9,523 reports, free text, stored in `TCGA_Reports.csv` (`patient_filename`, `text`)
- **Cancer type metadata**: `tcga_patient_to_cancer_type.csv` — 11,160 rows, 33 cancer types, joined on `patient_id` extracted from `patient_filename`
- **No structured ground truth** — a manual annotation effort is required to create the evaluation set

## Pilot Scope

Three cancer types for the first iteration:
- **BRCA** — Breast invasive carcinoma (1,097 reports, highest volume)
- **LUAD** — Lung adenocarcinoma (522 reports)
- **LUSC** — Lung squamous cell carcinoma (504 reports)

## Extraction Targets

### Group A — Patient Identification
Fields required for registry linkage. In real hospital reports these appear in the report header.

| Field | Difficulty | TCGA note |
|-------|-----------|-----------|
| Sex | Low | Presence unknown — assess in EDA |
| Date of birth | — | Confirmed absent in TCGA dataset — removed from schema (v1.1) |
| Institution ID | Low | Derivable from TCGA project/site code in `patient_filename` |
| Clinical process number | Low | Maps to `patient_filename` in TCGA context |

### Group B — Tumour Characterization (Priority 1 — pilot scope)

| Field | Difficulty |
|-------|-----------|
| Primary tumor site | Low |
| Histological diagnosis | Medium |
| Histological subtype | Medium |
| Tumor grade | Medium |
| TNM stage — T | High |
| TNM stage — N | High |
| TNM stage — M | High |
| Specimen type | Low |

### Group C — Priority 2 (second iteration)
Margin status, lymphovascular invasion, lymph node involvement, tumor size, IHC/biomarkers, molecular markers.

## Success Criteria

- **Primary**: aggregate `combined_score` ≥ 0.85, with each field's component meeting its per-field threshold (Low ≥ 0.89, Medium ≥ 0.83, High ≥ 0.87), assessed on the pathologist-annotated evaluation set (`eval_set_v2.jsonl`, 180 records). `combined_score` is a weighted mean (using `FIELD_WEIGHTS`) of: completeness-aware semantic similarity for free-text fields (primary_site, specimen_type, histological_diagnosis, histological_subtype, tumor_grade) — capturing clinical equivalence so pathologist review is only triggered when genuinely needed — and exact-match F1 for TNM fields (T, N, M) — where staging codes require exact accuracy.
- **Complementary**: `weighted_f1` and `mean_similarity_complete` reported alongside `combined_score` as diagnostics.
- **Secondary**: overall hallucination rate ≤ 5%
- **Tertiary**: pipeline reliability ≥ 95% (records passing all three validation layers without manual intervention)

### Threshold rationale

The per-field thresholds (Low ≥ 0.89, Medium ≥ 0.83, High ≥ 0.87) are **design targets chosen by the project author**, not values empirically derived from data or formally signed off by a pathologist. They encode a deliberate two-axis judgement, which is why they are not monotonic in difficulty:

- **Extraction difficulty** (the Low/Medium/High labels in the field tables) pushes the required bar *down* as fields get harder: easy fields (primary site, specimen type) are held to a near-perfect 0.89; medium fields (histological diagnosis, subtype, grade) are relaxed to 0.83.
- **Clinical criticality** pulls the bar back *up* for the most consequential fields. The High-difficulty group is the TNM staging codes (T, N, M), where a wrong value is the costliest error in the schema — it drives treatment and is the headline figure reported to the national registry. Difficulty alone would place these below 0.83; criticality lifts them to 0.87.

The collision of these two axes produces the ordering **Low (0.89) > High (0.87) > Medium (0.83)**: Medium sits lowest because it is the only group that is neither the easiest to extract nor the most critical to get right.

These numbers should be read as acceptance criteria the author set, to be revisited once they can be **grounded empirically** — e.g. anchored to manual cancer-registry abstraction accuracy reported in the literature, or to inter-annotator agreement measured on the project's own evaluation set. The "Cancer Registry Manager defines accuracy thresholds / pathologists sign off" roles below describe the governance that *would* set these in a real deployment; they are simulated stakeholder roles for this case study, not a record of approvals already obtained.

## Constraints

- TCGA public dataset — not real hospital data; treat as a research/validation proxy
- No patient data should leave the local environment in a production hospital context — addressed via model selection (local Llama fallback) or GDPR-compliant API
- Every extraction must be traceable: model version, prompt version, dataset version, validation status
- Human-in-the-loop is a design requirement, not a fallback

## Stakeholders (simulated)

- Cancer Registry Manager — decision-maker, defines accuracy thresholds, runs Tier 2 monthly audit
- Pathologists — domain experts, annotate ground truth, sign off on accuracy
- Hospital direction — institutional risk, GDPR governance
- National oncology registry — external accountability, defines submission standards
