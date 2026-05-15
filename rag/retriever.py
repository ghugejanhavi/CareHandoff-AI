"""
Context Retriever
Clinical-domain wrapper around HealthcareVectorStore.
Each agent calls these methods to get the exact context slice it needs.

V1 improvements:
  - parse_note_sections() for direct section extraction (no RAG needed on input note)
  - Plan-driven retrieval: accepts TaskPlan fields for patient-specific queries
  - Per-diagnosis guideline retrieval instead of generic hardcoded queries
  - Token-aware context assembly via tiktoken
"""

import logging
from typing import Dict, List, Optional

import tiktoken
from langchain_core.documents import Document

from config import MAX_CONTEXT_TOKENS
from rag.vector_store import HealthcareVectorStore, parse_note_into_sections

logger = logging.getLogger(__name__)

_ENCODER = tiktoken.encoding_for_model("gpt-4o-mini")


def truncate_to_tokens(text: str, max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    """
    Truncate text to fit within a token budget, cutting at the last sentence
    boundary before the limit.  Falls back to a hard token cut if no sentence
    boundary is found.
    """
    tokens = _ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated = _ENCODER.decode(tokens[:max_tokens])
    last_period = truncated.rfind(". ")
    last_newline = truncated.rfind("\n")
    cut = max(last_period, last_newline)
    if cut > len(truncated) // 2:
        return truncated[: cut + 1]
    return truncated


def parse_note_sections(discharge_note: str) -> Dict[str, str]:
    """
    Parse a raw discharge note into a dict of {section_name: section_text}.
    Used by agents that already have the note text and don't need RAG retrieval.
    Section names are header strings without the trailing colon
    (e.g. "Discharge Medications", "Pertinent Results").

    Sections that appear before the first known header are keyed as 'preamble'.
    """
    pairs = parse_note_into_sections(discharge_note)
    sections: Dict[str, str] = {}
    for name, text in pairs:
        if name in sections:
            sections[name] += "\n" + text
        else:
            sections[name] = text
    return sections


class ContextRetriever:
    """
    Clinically-aware retrieval layer.
    Agents depend on this interface, not directly on the vector store.
    """

    def __init__(self, vector_store: HealthcareVectorStore):
        self.store = vector_store

    # ── Note retrieval (RAG — used when the raw note isn't available) ──────

    def get_medication_sections(
        self, hadm_id: str, medications: Optional[List[str]] = None,
    ) -> List[Document]:
        """Retrieve medication-related note sections. Optionally include drug names in query."""
        query = "discharge medications prescriptions drug dose route"
        if medications:
            query += " " + " ".join(medications[:10])
        return self.store.retrieve_note_sections(query, hadm_id=hadm_id, k=4)

    def get_lab_sections(
        self, hadm_id: str, pending_labs: Optional[List[str]] = None,
    ) -> List[Document]:
        query = "pending lab results laboratory tests follow-up results"
        if pending_labs:
            query += " " + " ".join(pending_labs[:10])
        return self.store.retrieve_note_sections(query, hadm_id=hadm_id, k=3)

    def get_followup_sections(self, hadm_id: str) -> List[Document]:
        return self.store.retrieve_note_sections(
            "follow-up instructions appointment primary care physician",
            hadm_id=hadm_id, k=3,
        )

    def get_diagnosis_sections(self, hadm_id: str) -> List[Document]:
        return self.store.retrieve_note_sections(
            "discharge diagnosis principal diagnosis secondary diagnosis",
            hadm_id=hadm_id, k=3,
        )

    # ── Guidelines retrieval (plan-driven) ─────────────────────────────────

    def get_guidelines_for_diagnoses(
        self, diagnoses: List[str], topics: Optional[List[str]] = None, k_per_query: int = 3,
    ) -> List[Document]:
        """
        Issue one focused retrieval per diagnosis/topic from the TaskPlan.
        Returns deduplicated results with broader coverage than a single query.
        """
        queries = []
        for dx in diagnoses[:5]:
            queries.append(f"discharge documentation requirements for {dx}")
        for topic in (topics or [])[:3]:
            queries.append(f"clinical guideline {topic} discharge handoff")

        queries.append("medication reconciliation discharge documentation requirements")
        queries.append("follow-up instructions patient education pending results handoff")

        seen_content = set()
        unique_docs: List[Document] = []

        for query in queries:
            docs = self.store.retrieve_guidelines(query, k=k_per_query)
            for doc in docs:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    unique_docs.append(doc)

        return unique_docs

    # ── Utilities ──────────────────────────────────────────────────────────

    @staticmethod
    def docs_to_text(
        docs: List[Document],
        separator: str = "\n\n---\n\n",
        max_tokens: int = MAX_CONTEXT_TOKENS,
    ) -> str:
        """
        Flatten Documents into a single string for LLM context,
        respecting a token budget.  Includes section metadata when available.
        """
        parts = []
        for d in docs:
            source = d.metadata.get("source", "unknown")
            section = d.metadata.get("section_name", "")
            header = f"[Source: {source}"
            if section:
                header += f" | Section: {section}"
            header += "]"
            parts.append(f"{header}\n{d.page_content}")

        full = separator.join(parts)
        return truncate_to_tokens(full, max_tokens)
