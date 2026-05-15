"""
Central configuration for Healthcare Agentic RAG.
All tuneable constants live here — no magic strings scattered across modules.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI / OpenRouter ───────────────────────────────────────────────────────
# LLM calls (ChatOpenAI):
#   OPENAI_API_KEY  = sk-or-v1-...  (OpenRouter key)
#   OPENAI_API_BASE = https://openrouter.ai/api/v1
#
# Embedding calls (OpenAIEmbeddings):
#   OpenRouter does NOT support /v1/embeddings — embeddings must go directly to
#   OpenAI.  Set OPENAI_EMBEDDING_KEY to a standard sk-... OpenAI key.
#   If you only have an OpenAI key (no OpenRouter), you can set:
#     OPENAI_API_KEY = sk-...  (same key for both, leave OPENAI_API_BASE unset)
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE      = os.getenv("OPENAI_API_BASE")        # None → standard OpenAI
OPENAI_EMBEDDING_KEY = os.getenv("OPENAI_EMBEDDING_KEY") or OPENAI_API_KEY

# ── LLM & Embedding ─────────────────────────────────────────────────────────
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_TEMPERATURE = 0.2

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MIMIC_DIR = os.path.join(DATA_DIR, "mimic")
GUIDELINES_DIR = os.path.join(DATA_DIR, "guidelines")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chromadb_data")
REVISED_NOTES_FILE = os.path.join(BASE_DIR, "data", "revised_notes.csv")

# ── ChromaDB Collections ─────────────────────────────────────────────────────
NOTES_COLLECTION = "discharge_notes"
GUIDELINES_COLLECTION = "clinical_guidelines"

# ── MIMIC-IV File Names ───────────────────────────────────────────────────────
# Files may be .csv or .csv.gz — loader handles both
MIMIC_FILES = {
    "discharge":     "discharge.csv",        # MIMIC-IV-Note
    "prescriptions": "prescriptions.csv",
    "labevents":     "labevents.csv",
    "d_labitems":    "d_labitems.csv",
    "diagnoses":     "diagnoses_icd.csv",
    "procedures":    "procedures_icd.csv",
    "admissions":    "admissions.csv",
    "d_icd_diagnoses": "d_icd_diagnoses.csv",
    "d_icd_procedures": "d_icd_procedures.csv",
}

# ── RxNorm API ────────────────────────────────────────────────────────────────
RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"
RXNORM_TIMEOUT_SEC = 5

# ── Ingestion limits ──────────────────────────────────────────────────────────
# Caps notes ingested at build time to stay within Render free-tier 512 MB RAM.
# 50 patients ≈ 1,250 chunks × 1,536 dims + ChromaDB overhead ≈ 200 MB.
NOTES_INGEST_LIMIT = 50

# ── RAG Retrieval ─────────────────────────────────────────────────────────────
NOTES_CHUNK_SIZE = 1200
NOTES_CHUNK_OVERLAP = 100
GUIDELINES_CHUNK_SIZE = 3000   # larger chunks → fewer vectors → less RAM
GUIDELINES_CHUNK_OVERLAP = 150
TOP_K_NOTES = 6
TOP_K_GUIDELINES = 5
RELEVANCE_THRESHOLD = 0.3
MAX_CONTEXT_TOKENS = 7000

# ── Clinical Section Headers (MIMIC-IV discharge note format) ─────────────────
MIMIC_SECTION_HEADERS = [
    "Chief Complaint:",
    "History of Present Illness:",
    "Past Medical History:",
    "Social History:",
    "Family History:",
    "Physical Exam:",
    "Pertinent Results:",
    "Brief Hospital Course:",
    "Medications on Admission:",
    "Discharge Medications:",
    "Discharge Disposition:",
    "Discharge Diagnosis:",
    "Discharge Condition:",
    "Discharge Instructions:",
    "Followup Instructions:",
    "Allergies:",
]
