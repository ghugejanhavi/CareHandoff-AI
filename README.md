# CareHandoff AI

> Agentic RAG system that automatically audits hospital discharge notes for missing clinical information before care handoff.

---

## Overview

Discharge-to-handoff communication failures are responsible for an estimated 70% of medical errors at care transitions (IOM, Joint Commission). CareHandoff AI runs a 5-agent pipeline over a discharge note — comparing it against the patient's structured EHR data and clinical guidelines — and surfaces medication gaps, lab omissions, allergy conflicts, and guideline violations for clinician review before the patient leaves.

**Decision-support only** — every flagged item is reviewed and acted on by a clinician. No autonomous clinical decisions are made.

---

## Features

- **5-agent pipeline** — Planning → EHR Comparison + Guidelines (parallel) → Self-Correction → HITL Review
- **EHR cross-check** — compares discharge note against MIMIC-IV medications, labs, allergies, and procedures
- **RxNorm normalization** — live NLM API call ensures `"Metoprolol Succinate ER"` and `"metoprolol succinate"` are treated as the same drug
- **Guideline RAG** — semantic search over 17 ACC/AHA, AHRQ, and specialty society guidelines
- **Self-correction** — dedicated agent re-reads the note and dismisses false positives before the clinician sees anything
- **Adaptive HITL loop** — when a clinician dismisses a gap, the rationale re-enters the reasoning loop and the gap may be re-flagged
- **Note rewrite** — generates a corrected discharge note with every confirmed gap addressed and hallucination-guarded
- **Clinical Q&A** — chat interface for guideline lookups, powered by the same RAG index
- **React SPA** — clean 4-tab UI (Analysis → Review → Revised Note → Q&A)

---

## Architecture

```
Discharge Note + hadm_id
         │
         ▼
┌─────────────────────────────────────────┐
│  Agent 1 — Planning Agent               │
│  Reads full note → builds TaskPlan      │
│  Identifies diagnoses, meds, labs,      │
│  guideline topics, gaps to investigate  │
└──────────────────┬──────────────────────┘
                   │ TaskPlan
         ┌─────────┴──────────┐  ← parallel
         ▼                    ▼
┌──────────────────┐  ┌───────────────────────┐
│ Agent 2          │  │ Agent 3               │
│ EHR Comparison   │  │ Guidelines Agent      │
│ · Medication gaps│  │ · RAG → ChromaDB      │
│ · Lab gaps       │  │ · ACC/AHA compliance  │
│ · RxNorm lookup  │  │ · AHRQ RED Toolkit    │
│ · Dose checks    │  │ · 17 guideline topics │
└────────┬─────────┘  └──────────┬────────────┘
         └──────────┬────────────┘
                    │ All gaps merged
                    ▼
┌─────────────────────────────────────────┐
│  Agent 4 — Self-Correction Agent        │
│  Re-verifies every gap (reduces FP)     │
│  Adjusts severity by care setting       │
└──────────────────┬──────────────────────┘
                   │ VerifiedGaps
                   ▼
┌─────────────────────────────────────────┐
│  Agent 5 — HITL Orchestrator            │
│  Formats ReviewPackage for clinician    │
│  Dismissals → re-enter reasoning loop   │
└─────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | GPT-4o-mini via OpenAI / OpenRouter |
| Embeddings | `text-embedding-3-small` (standard OpenAI endpoint) |
| Vector store | ChromaDB (persistent, two collections) |
| Orchestration | LangChain + custom agents |
| Backend | FastAPI + Uvicorn |
| Frontend | React 18 + TypeScript + Vite + Tailwind CSS |
| Clinical Q&A UI | Gradio (mounted at `/qa`) |
| Deployment | Docker (multi-stage) on Render |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/roshan2110/CareHandoff_AI.git
cd CareHandoff_AI
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env — see Configuration section below

# 3. Place MIMIC-IV data files (see Data Setup below)

# 4. Run
uvicorn main:app --reload
# Open http://localhost:8000
```

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 20+ (only if rebuilding the frontend)
- OpenAI API key (`text-embedding-3-small` + `gpt-4o-mini`)
- MIMIC-IV data access (free via PhysioNet — requires CITI training)

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Key for LLM calls (GPT-4o-mini). Accepts OpenRouter keys (`sk-or-v1-...`) |
| `OPENAI_API_BASE` | No | Leave blank for standard OpenAI. Set to `https://openrouter.ai/api/v1` for OpenRouter |
| `OPENAI_EMBEDDING_KEY` | No | Standard OpenAI key (`sk-...`) for embeddings. Falls back to `OPENAI_API_KEY` if not set. **Required if using OpenRouter for LLM**, since OpenRouter does not support `/v1/embeddings` |

### Data Setup

#### MIMIC-IV files

Place files in `data/raw/mimic/` (`.csv` or `.csv.gz` both work):

| File | Table | Required |
|------|-------|----------|
| `discharge.csv` | MIMIC-IV-Note | Yes |
| `prescriptions.csv` | MIMIC-IV | Yes |
| `labevents.csv` | MIMIC-IV | Yes |
| `d_labitems.csv` | MIMIC-IV | Yes |
| `diagnoses_icd.csv` | MIMIC-IV | Yes |
| `admissions.csv` | MIMIC-IV | Recommended |
| `procedures_icd.csv` | MIMIC-IV | Recommended |
| `d_icd_diagnoses.csv` | MIMIC-IV | Recommended |

Access: [physionet.org](https://physionet.org) — free, requires CITI training (~2 hours). Approval takes 3–5 days.

#### Clinical guidelines

17 guideline documents are included in `data/raw/guidelines/` (ACC/AHA, AHRQ, ACG, ADA). No additional downloads needed.

#### Ingest into ChromaDB

```bash
python -c "
from orchestrator import AgenticRAGOrchestrator
o = AgenticRAGOrchestrator()
o.ingest_discharge_notes(limit=50)   # limit is optional
o.ingest_guidelines()
"
```

Or start the server and let it auto-ingest in the background on first launch.

---

## Usage

### Web Interface

```bash
uvicorn main:app --reload
# Open http://localhost:8000
```

| Tab | What it does |
|-----|-------------|
| **Patient Analysis** | Select a patient by `hadm_id`, choose care setting, run the full 5-agent pipeline |
| **Clinician Review** | Confirm / dismiss / escalate each flagged gap; dismissals re-enter the reasoning loop |
| **Revised Note** | Generate and validate a corrected discharge note for all confirmed gaps |
| **Clinical Q&A** | Ask guideline questions in natural language (RAG over all 17 documents) |

### REST API

Base URL: `http://localhost:8000/api`
Interactive docs: `http://localhost:8000/api/docs`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns ingestion status and chunk counts |
| `GET` | `/patients` | List available `hadm_id` values |
| `POST` | `/analyze` | Run the 5-agent pipeline |
| `GET` | `/review` | Get pending review items |
| `POST` | `/review/feedback` | Submit clinician action (confirm / dismiss / escalate) |
| `POST` | `/review/confirm-all` | Batch-confirm all pending items |
| `POST` | `/note/generate` | Generate revised discharge note |
| `GET` | `/note/validation` | Hallucination guard report for new additions |
| `POST` | `/note/save` | Persist revised note to CSV |

**Example — run an analysis:**

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"hadm_id": "12345678", "care_setting": "home"}'
```

### Programmatic Usage

```python
from orchestrator import AgenticRAGOrchestrator

orc = AgenticRAGOrchestrator()

result = orc.analyze(
    discharge_note="...",          # raw text
    hadm_id="12345678",
    care_setting="home",           # "home" | "skilled_nursing_facility" | "urgent_clinic"
)

print(result.summary())
# Analysis 20240415_103022 | hadm_id=12345678
#   Gaps: 5 total (1 critical, 2 high, 2 medium, 0 low)
#   Dismissed in self-correction: 1

# Individual agent outputs
print(result.ehr_result.medication_gaps)
print(result.guidelines_result.violations)

# Clinician feedback (triggers re-analysis on dismiss)
orc.hitl.record_feedback(
    analysis_id=result.analysis_id,
    item_id="20240415_103022_gap_0",
    action="dismiss",
    notes="Patient enrolled in warfarin clinic at receiving SNF.",
)
```

---

## Deployment

### Render (Docker)

The app is configured for one-click deploy on Render via `render.yaml`.

Embeddings are baked into the Docker image at build time so the container starts in ~5 seconds with no runtime ingestion.

**Steps:**

1. Fork / push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Blueprint → connect your repo
3. In your service's **Environment** tab, set:
   - `OPENAI_API_KEY`
   - `OPENAI_EMBEDDING_KEY` (standard `sk-...` key — required for embeddings even if using OpenRouter for LLM)
   - `OPENAI_API_BASE` (set to `https://openrouter.ai/api/v1` if using OpenRouter, otherwise leave blank)

Render passes the `OPENAI_EMBEDDING_KEY` and `OPENAI_API_KEY` values as Docker build args automatically (configured in `render.yaml`).

**Verify the deploy:**

```bash
curl https://<your-app>.onrender.com/api/health
# {"status":"ok","ingestion_complete":true,"notes_chunks":1250,"guidelines_chunks":3800}
```

### Local Docker

```bash
docker build \
  --build-arg OPENAI_EMBEDDING_KEY=sk-... \
  --build-arg OPENAI_API_KEY=sk-... \
  -t carehandoff-ai .

docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  carehandoff-ai
```

---

## Project Structure

```
CareHandoff_AI/
├── main.py                      # FastAPI app — mounts API, Gradio Q&A, React SPA
├── orchestrator.py              # Entry point: AgenticRAGOrchestrator.analyze()
├── config.py                    # All constants (model names, paths, chunk sizes)
├── requirements.txt
├── Dockerfile                   # Multi-stage: frontend → python-base → ingestion → runtime
├── render.yaml                  # Render deployment blueprint
├── .env.example
│
├── agents/
│   ├── planning_agent.py        # Agent 1 — TaskPlan via structured output
│   ├── ehr_comparison_agent.py  # Agent 2 — medication/lab/allergy/dose gaps
│   ├── guidelines_agent.py      # Agent 3 — RAG-retrieved guideline compliance
│   ├── self_correction_agent.py # Agent 4 — re-verify gaps, adjust severity
│   ├── hitl_orchestrator.py     # Agent 5 — review package + dismissal re-analysis
│   ├── note_rewrite_agent.py    # Rewrites note with confirmed gaps addressed
│   └── rewrite_validator.py     # Hallucination guard on [ADDED] blocks
│
├── api/
│   ├── routes.py                # REST endpoints (/analyze, /review, /note, /health)
│   └── state.py                 # Singleton AppState — shared across requests
│
├── rag/
│   ├── vector_store.py          # ChromaDB: two collections, section-aware chunking
│   └── retriever.py             # Domain-aware retrieval helpers per agent
│
├── qa/
│   └── clinical_qa.py           # MetaIntelligentClinicalRAG + GuardedClinicalRAG
│
├── gradio_qa_app.py             # Gradio Q&A blocks — mounted at /qa by main.py
│
├── data/
│   ├── loaders/
│   │   ├── mimic_loader.py      # MIMIC-IV CSV loader; get_patient_ehr(hadm_id)
│   │   ├── guidelines_loader.py # Loads .txt/.pdf/.html → LangChain Documents
│   │   └── rxnorm_client.py     # NLM RxNorm REST API — live drug normalization
│   └── raw/
│       ├── mimic/               # Place MIMIC-IV CSV files here (gitignored)
│       └── guidelines/          # 17 guideline documents (included)
│
└── frontend/
    ├── src/
    │   ├── pages/               # AnalysisPage, ReviewPage, NotePage, QAPage
    │   ├── lib/                 # api.ts (fetch client), types.ts, utils.ts
    │   └── App.tsx              # 4-tab layout
    └── dist/                    # Production build (served by FastAPI)
```

---

## Output Reference

```python
AnalysisResult
├── analysis_id        "20240415_103022"
├── hadm_id            "12345678"
│
├── task_plan
│   ├── identified_diagnoses         ["Heart failure", "Hypertension", ...]
│   ├── medications_to_verify        ["Furosemide", "Metoprolol", ...]
│   ├── pending_labs_to_check        ["BMP", "INR"]
│   └── critical_gaps_to_investigate [...]
│
├── ehr_result
│   ├── medication_gaps   [MedicationGap(drug_name, reason_flagged, severity)]
│   ├── lab_gaps          [LabGap(lab_name, reason_flagged, severity)]
│   ├── allergy_alerts    [AllergyAlert(drug_name, allergy, conflict_type, ...)]
│   └── dose_discrepancies[DoseDiscrepancy(drug_name, ehr_dose, note_dose, ...)]
│
├── guidelines_result
│   ├── violations        [GuidelineViolation(requirement, source, missing_element, severity)]
│   ├── compliant_areas   [str]
│   └── overall_compliant bool
│
├── correction_result
│   ├── verified_gaps     [VerifiedGap(gap_type, description, severity, action)]
│   └── dismissed_count   int
│
└── review_package
    ├── review_items      [ReviewItem(item_id, severity, status, ...)]
    ├── critical_count    int
    ├── high_count        int
    ├── medium_count      int
    └── total_gaps        int
```

---

## Care Settings

The Self-Correction Agent adjusts gap severity based on the patient's destination:

| Setting | Behaviour |
|---------|-----------|
| `home` | Patient/caregiver responsible — higher severity for medication and monitoring gaps |
| `skilled_nursing_facility` | Nursing staff manages meds — reduced severity unless high-risk drug |
| `urgent_clinic` | Physician needs complete information immediately — elevated severity across all gap types |

---

## Cost Estimate (GPT-4o-mini)

| Step | Approx. tokens | Cost |
|------|---------------|------|
| Planning Agent | ~600 | ~$0.0001 |
| EHR Comparison | ~1,500 | ~$0.0002 |
| Guidelines Agent | ~1,800 | ~$0.0003 |
| Self-Correction | ~1,200 | ~$0.0002 |
| HITL (per dismissal) | ~800 | ~$0.0001 |
| **Total per analysis** | **~6,000** | **~$0.001** |
| 1,000 analyses/day | | ~$1/day |

---

## Clinical Disclaimer

- This system is **decision-support, not autonomous** — a clinician reviews and approves every flagged item before any action is taken
- Final clinical responsibility rests with the treating and receiving physicians
- Audit logging is available via `HITLOrchestrator.get_audit_log()`
- No patient data is transmitted beyond the configured OpenAI endpoint; the ChromaDB store is local/container-bound

---

## Data Sources

| Source | Role | Access |
|--------|------|--------|
| MIMIC-IV-Note (`discharge.csv`) | Discharge summaries to audit | PhysioNet — free, CITI training required |
| MIMIC-IV structured tables | Ground-truth EHR (meds, labs, orders) | Same PhysioNet credential |
| ACC/AHA Guidelines | Cardiac/metabolic discharge standards | Public — included in repo |
| AHRQ RED Toolkit | Handoff communication requirements | Public — included in repo |
| ACG, ADA, specialty guidelines | Condition-specific discharge criteria | Public — included in repo |
| RxNorm API (NLM) | Live drug name normalization | No key required |

---

## References

- IOM: *To Err is Human* — communication failures as leading cause of medical errors
- Joint Commission Sentinel Event Alert 58: Care Coordination
- AHRQ Re-Engineered Discharge (RED) Toolkit
- ACC/AHA Hypertension Guidelines (2023)
- ACC/AHA Heart Failure Guidelines
- ADA Standards of Medical Care in Diabetes
