"""
ChromaDB Vector Store
Manages two persistent collections:
  - discharge_notes  : chunked MIMIC-IV-Note discharge summaries
  - clinical_guidelines : chunked ACC/AHA / AHRQ guideline documents

Uses OpenAI text-embedding-3-small for all embeddings.

V1 improvements:
  - Section-aware chunking with section_name metadata
  - Deterministic chunk IDs for idempotent re-ingestion
  - Relevance-score filtering on retrieval
  - Safe re-ingestion via collection clearing
"""

import re
import hashlib
import logging
from typing import List, Optional, Tuple
import pandas as pd
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    OPENAI_EMBEDDING_KEY,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    NOTES_COLLECTION,
    GUIDELINES_COLLECTION,
    MIMIC_SECTION_HEADERS,
    NOTES_CHUNK_SIZE,
    NOTES_CHUNK_OVERLAP,
    GUIDELINES_CHUNK_SIZE,
    GUIDELINES_CHUNK_OVERLAP,
    RELEVANCE_THRESHOLD,
)

logger = logging.getLogger(__name__)

_SECTION_PATTERN = re.compile(
    r"(?=^(" + "|".join(re.escape(h) for h in MIMIC_SECTION_HEADERS) + r"))",
    re.MULTILINE,
)


def _deterministic_id(*parts: str) -> str:
    """SHA-256 based deterministic ID from arbitrary string parts."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def parse_note_into_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split a MIMIC-IV discharge note into (section_name, section_text) pairs
    by matching known clinical section headers.

    Sections that appear before the first known header are labeled 'preamble'.
    Returns a list of tuples preserving document order.
    """
    splits = _SECTION_PATTERN.split(text)

    sections: List[Tuple[str, str]] = []
    i = 0
    while i < len(splits):
        chunk = splits[i]
        if chunk in MIMIC_SECTION_HEADERS:
            header = chunk.rstrip(":")
            body = splits[i + 1] if i + 1 < len(splits) else ""
            body = body.strip()
            if body:
                sections.append((header, f"{chunk}\n{body}"))
            i += 2
        else:
            body = chunk.strip()
            if body:
                sections.append(("preamble", body))
            i += 1

    return sections


class HealthcareVectorStore:
    """
    Wraps two ChromaDB collections with domain-aware chunking strategies:
    - Discharge notes: regex-split on clinical section headers, then character-split oversized sections
    - Guidelines: recursive character split with deterministic IDs
    """

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        self._embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_EMBEDDING_KEY,
            # Explicitly force standard OpenAI endpoint.
            # langchain_openai auto-reads OPENAI_API_BASE from env if not set here,
            # which would route to OpenRouter — which does NOT support /v1/embeddings.
            openai_api_base="https://api.openai.com/v1",
            # High max_retries so transient 429 rate-limit errors during bulk
            # ingestion are retried with exponential backoff instead of crashing.
            max_retries=20,
        )
        self._notes_store: Optional[Chroma] = None
        self._guidelines_store: Optional[Chroma] = None

    # ── Collection accessors ───────────────────────────────────────────────

    @property
    def notes_store(self) -> Chroma:
        if self._notes_store is None:
            self._notes_store = Chroma(
                collection_name=NOTES_COLLECTION,
                embedding_function=self._embeddings,
                persist_directory=self.persist_dir,
            )
        return self._notes_store

    @property
    def guidelines_store(self) -> Chroma:
        if self._guidelines_store is None:
            self._guidelines_store = Chroma(
                collection_name=GUIDELINES_COLLECTION,
                embedding_function=self._embeddings,
                persist_directory=self.persist_dir,
            )
        return self._guidelines_store

    # ── Collection management ──────────────────────────────────────────────

    def clear_notes(self) -> None:
        """Delete all documents from the notes collection."""
        try:
            col = self.notes_store._collection
            count = col.count()
            if count > 0:
                ids = col.get()["ids"]
                col.delete(ids=ids)
                logger.info("Cleared %d documents from notes collection", count)
        except Exception as e:
            logger.error("Failed to clear notes collection: %s", e)

    def clear_guidelines(self) -> None:
        """Delete all documents from the guidelines collection."""
        try:
            col = self.guidelines_store._collection
            count = col.count()
            if count > 0:
                ids = col.get()["ids"]
                col.delete(ids=ids)
                logger.info("Cleared %d documents from guidelines collection", count)
        except Exception as e:
            logger.error("Failed to clear guidelines collection: %s", e)

    # ── Ingestion ──────────────────────────────────────────────────────────

    def ingest_discharge_notes(self, notes_df: pd.DataFrame, batch_size: int = 100) -> int:
        """
        Section-split and embed discharge notes from a MIMIC-IV notes DataFrame.
        Uses deterministic IDs so re-ingestion is idempotent.
        """
        if notes_df.empty:
            logger.warning("No discharge notes to ingest.")
            return 0

        self.clear_notes()

        sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=NOTES_CHUNK_SIZE,
            chunk_overlap=NOTES_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        total = 0
        batch_docs: List[Document] = []
        batch_ids: List[str] = []

        for _, row in notes_df.iterrows():
            raw_text = str(row.get("text", "")).strip()
            if not raw_text:
                continue
            hadm_id = str(row.get("hadm_id", "unknown"))
            subject_id = str(row.get("subject_id", "unknown"))

            sections = parse_note_into_sections(raw_text)

            for si, (section_name, section_text) in enumerate(sections):
                if len(section_text) > NOTES_CHUNK_SIZE:
                    sub_chunks = sub_splitter.split_text(section_text)
                else:
                    sub_chunks = [section_text]

                for ci, chunk in enumerate(sub_chunks):
                    doc_id = _deterministic_id(hadm_id, str(si), section_name, str(ci))
                    batch_docs.append(Document(
                        page_content=chunk,
                        metadata={
                            "hadm_id": hadm_id,
                            "subject_id": subject_id,
                            "section_name": section_name,
                            "chunk_index": ci,
                            "source": "mimic_discharge",
                        }
                    ))
                    batch_ids.append(doc_id)

                    if len(batch_docs) >= batch_size:
                        self.notes_store.add_documents(batch_docs, ids=batch_ids)
                        total += len(batch_docs)
                        logger.info("Ingested notes batch (total so far: %d)", total)
                        batch_docs = []
                        batch_ids = []

        if batch_docs:
            self.notes_store.add_documents(batch_docs, ids=batch_ids)
            total += len(batch_docs)

        logger.info("Total discharge note chunks ingested: %d", total)
        return total

    def ingest_guidelines(self, guideline_docs: List[Document], batch_size: int = 100) -> int:
        """
        Chunk and embed guideline documents with deterministic IDs.
        Clears existing guidelines first for idempotent re-ingestion.
        """
        if not guideline_docs:
            logger.warning("No guideline documents to ingest.")
            return 0

        self.clear_guidelines()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=GUIDELINES_CHUNK_SIZE,
            chunk_overlap=GUIDELINES_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(guideline_docs)

        chunk_ids = [
            _deterministic_id(
                c.metadata.get("source", "unknown"),
                str(i),
            )
            for i, c in enumerate(chunks)
        ]

        for start in range(0, len(chunks), batch_size):
            batch_docs = chunks[start: start + batch_size]
            batch_ids = chunk_ids[start: start + batch_size]
            self.guidelines_store.add_documents(batch_docs, ids=batch_ids)
            logger.info("Ingested guidelines batch %d-%d / %d", start, start + len(batch_docs), len(chunks))

        logger.info("Total guideline chunks ingested: %d", len(chunks))
        return len(chunks)

    # ── Retrieval ──────────────────────────────────────────────────────────

    def retrieve_note_sections(
        self,
        query: str,
        hadm_id: Optional[str] = None,
        k: int = 6,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
    ) -> List[Document]:
        """
        Semantic search over discharge note chunks with relevance filtering.
        Returns only chunks scoring above the threshold.
        """
        filter_dict = {"hadm_id": str(hadm_id)} if hadm_id else None
        try:
            results = self.notes_store.similarity_search_with_relevance_scores(
                query, k=k, filter=filter_dict,
            )
            filtered = [doc for doc, score in results if score >= relevance_threshold]
            if not filtered and results:
                filtered = [results[0][0]]
            return filtered
        except Exception as e:
            logger.error("Note retrieval failed: %s", e)
            return []

    def retrieve_guidelines(
        self,
        query: str,
        k: int = 5,
        source_filter: Optional[str] = None,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
    ) -> List[Document]:
        """
        Semantic search over clinical guideline chunks with relevance filtering.
        """
        filter_dict = {"source": source_filter} if source_filter else None
        try:
            results = self.guidelines_store.similarity_search_with_relevance_scores(
                query, k=k, filter=filter_dict,
            )
            filtered = [doc for doc, score in results if score >= relevance_threshold]
            if not filtered and results:
                filtered = [results[0][0]]
            return filtered
        except Exception as e:
            logger.error("Guidelines retrieval failed: %s", e)
            return []

    # ── Stats ──────────────────────────────────────────────────────────────

    def collection_stats(self) -> dict:
        try:
            notes_count = self.notes_store._collection.count()
        except Exception:
            notes_count = 0
        try:
            guidelines_count = self.guidelines_store._collection.count()
        except Exception:
            guidelines_count = 0
        return {
            "notes_chunks": notes_count,
            "guidelines_chunks": guidelines_count,
            "persist_dir": self.persist_dir,
        }
