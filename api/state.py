"""
Singleton application state.
Initialised once on FastAPI startup; shared across all requests.
"""
import logging
from typing import Optional

from orchestrator import AgenticRAGOrchestrator, AnalysisResult
from agents.rewrite_validator import ValidationResult
from qa.clinical_qa import MetaIntelligentClinicalRAG, GuardedClinicalRAG

logger = logging.getLogger(__name__)


class AppState:
    orchestrator: AgenticRAGOrchestrator
    current_result: Optional[AnalysisResult] = None
    current_note: str = ""
    revised_note: str = ""
    validation_result: Optional[ValidationResult] = None
    guarded_rag: Optional[GuardedClinicalRAG] = None


_state: Optional[AppState] = None


def get_state() -> AppState:
    global _state
    if _state is None:
        raise RuntimeError("AppState not initialised — call initialize_state() on startup.")
    return _state


def initialize_state() -> AppState:
    """
    Fast synchronous init — creates all objects but does NOT ingest data.
    Data ingestion (slow: ~5-15 min on cold start) is kicked off as an
    asyncio background task by main.py so the server can start and pass
    Render's health check immediately.
    """
    global _state
    logger.info("[AppState] Initialising (ingestion deferred to background)...")
    s = AppState()
    s.orchestrator = AgenticRAGOrchestrator()

    meta_rag = MetaIntelligentClinicalRAG(s.orchestrator.vector_store)
    s.guarded_rag = GuardedClinicalRAG(meta_rag)

    _state = s
    logger.info("[AppState] Ready — waiting for background ingestion.")
    return _state
