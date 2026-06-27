# Tech Stack

## Version Control

- **Git + GitHub** — public repo: `pathology-report-llm-extraction`
- Each SDD feature phase is developed on its own branch and merged via PR
- Specs (`specs/`) are versioned alongside code — a prompt or schema change gets its own commit

## Language & Runtime

- **Python 3.11+**
- Package manager: **`uv`** — fast Rust-based package manager, replaces pip + venv
- Environment variables: `python-dotenv` via `.env` file

### uv Workflow

```bash
uv init                    # initialize project (creates pyproject.toml)
uv add <package>           # add dependency
uv sync                    # install all dependencies
uv run python src/...      # run a script in the managed environment
uv run jupyter notebook    # launch notebook
```

## Project Structure

```
/
├── data/
│   ├── raw/                   # TCGA_Reports.csv, tcga_patient_to_cancer_type.csv
│   ├── processed/             # filtered corpus, JSONL datasets
│   └── annotations/           # ground truth evaluation set (JSONL)
├── specs/                     # SDD constitution and feature specs
│   ├── features/              # per-feature plan/requirements/validation docs
│   └── research/              # backlog of research notes
├── src/
│   ├── pipeline/              # 5-component extraction pipeline
│   │   ├── loader.py          # Component 1: report loader
│   │   ├── prompt_constructor.py  # Component 2
│   │   ├── model_caller.py    # Component 3: aisuite API caller
│   │   ├── validator.py       # Component 4: 3-layer validator
│   │   └── result_writer.py   # Component 5
│   ├── schema.py              # Pydantic output schema (versioned)
│   ├── prompts/               # versioned prompt templates (.txt or .jinja2)
│   └── evaluation/            # metric computation scripts
├── notebooks/
│   ├── 01_eda.ipynb           # Data Understanding (Phase 2)
│   └── 02_evaluation.ipynb    # Evaluation results and model comparison
├── mlflow/                    # MLflow tracking artifacts
├── .env                       # API keys (gitignored)
├── requirements.txt
└── README.md
```

## Core Dependencies

| Package | Purpose |
|---------|---------|
| `aisuite` | Unified multi-model interface — swap models by changing one string |
| `pydantic>=2.0` | Structured output schema, field validation, JSON enforcement |
| `mlflow` | Experiment tracking — logs model version, prompt version, metrics per run |
| `pandas` | Data manipulation, CSV/JSONL loading, join operations |
| `tiktoken` | Token counting for context window planning (OpenAI tokenizer) |
| `python-dotenv` | Load API keys from `.env` |
| `jupyter` | Interactive EDA and evaluation notebooks |
| `tqdm` | Progress bars for batch pipeline runs |
| `pytest` + `pytest-cov` | Unit/integration tests for deterministic pipeline components (dev dependency) |

## Model Access via aisuite

aisuite uses provider-prefixed model strings: `provider:model-name`

| Provider | aisuite string | Notes |
|----------|---------------|-------|
| OpenAI | `openai:gpt-4o` | **Starting model** — requires `OPENAI_API_KEY` |
| Ollama (local) | `ollama:llama3` | Local GDPR-safe fallback, requires Ollama running |

Switching models requires changing **one config value** — the pipeline is model-agnostic by design.

## Output Schema

Defined in `src/schema.py` using Pydantic v2. Versioned: `SchemaV1`, `SchemaV2`, etc.

```python
class TNMStage(BaseModel):
    T: Optional[str]
    N: Optional[str]
    M: Optional[str]
    edition: Optional[str]      # e.g. "AJCC8"
    confidence: Optional[str]

class PatientIdentification(BaseModel):
    sex: Optional[str]                  # conditionally present in TCGA reports
    institution_id: Optional[str]       # derivable from patient_filename
    clinical_process_number: Optional[str]
    # date_of_birth removed in v1.1 — confirmed absent in TCGA dataset

class PathologyExtraction(BaseModel):
    report_id: str
    cancer_type: str
    # Group A — Patient identification
    patient: Optional[PatientIdentification]
    # Group B — Tumour characterization
    primary_site: Optional[str]
    histological_diagnosis: Optional[str]
    histological_subtype: Optional[str]
    tumor_grade: Optional[str]
    tnm_stage: Optional[TNMStage]
    specimen_type: Optional[str]
    # Metadata
    extraction_notes: Optional[str]
    hallucination_flags: Optional[list[str]]
```

## Validation Pipeline (3 layers)

1. **JSON parsing** — enforce valid JSON output; retry once with correction prompt on failure
2. **Pydantic validation** — parse through `PathologyExtraction`; log validation errors
3. **Clinical vocabulary check** — TNM regex, grade controlled vocabulary, ICD-O-3 site list; flag out-of-vocabulary values as `hallucination_candidate`

## Experiment Tracking (MLflow)

- Experiment name: `cancer_registry_extraction`
- Logged per run: model name/version, prompt version, dataset version, batch date, cancer types
- Logged metrics: records processed, Tier 1 flagging rate, validation pass rate, hallucination rate, avg latency, total cost, weighted F1
- Logged artifacts: results JSONL, Tier 1 flagged records, run summary

## Testing Strategy

Two distinct layers — never confuse them:

| Layer | Tool | What it tests |
|-------|------|--------------|
| **Unit/integration tests** | `pytest` | Deterministic components: schema parsing, prompt construction, truncation logic, validator layers, data joins, result writer |
| **LLM evaluation** | Evaluation notebook + human review | Extraction accuracy, F1 per field, hallucination rate — requires annotated ground truth, cannot be automated |

Run tests with:
```bash
uv run pytest tests/ -v --cov=src
```

Each pipeline component (`loader`, `prompt_constructor`, `validator`, etc.) has a corresponding test file in `tests/`.

## Versioning Policy

- Every change to prompt template, Pydantic schema, or dataset creates a new version (`v1.0`, `v1.1`, etc.)
- Version is logged in every output record and every MLflow run
- Changing a prompt template **invalidates previous evaluation results** — re-run required

## Environment Variables

```env
OPENAI_API_KEY=...
MLFLOW_TRACKING_URI=./mlflow
```
