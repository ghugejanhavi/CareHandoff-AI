"""
HITL Orchestrator — Agent 5 of 5
Formats verified gaps for clinician review and manages the feedback loop.

Key design: a clinician DISMISSAL re-enters the reasoning loop with the
dismissal rationale injected as additional context — it is NOT simply filtered out.
This is the adaptive replanning behaviour required by Agentic RAG.
"""

import logging
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import LLM_MODEL, LLM_TEMPERATURE
from rag.retriever import truncate_to_tokens
from agents.self_correction_agent import VerifiedGap, SelfCorrectionResult

logger = logging.getLogger(__name__)


# ── Data models ────────────────────────────────────────────────────────────────

class ReviewItem(BaseModel):
    item_id: str
    gap_type: str
    description: str
    severity: str
    recommended_action: str
    care_setting_note: str
    status: str = "pending_review"  # pending_review | confirmed | dismissed | escalated
    clinician_notes: str = ""
    reviewed_at: Optional[str] = None

class ReviewPackage(BaseModel):
    analysis_id: str
    review_items: List[ReviewItem]
    total_gaps: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    created_at: str

class ReevaluationResult(BaseModel):
    updated_severity: str = Field(description="Revised severity after re-analysis")
    updated_recommendation: str = Field(description="Updated recommended action")
    reanalysis_notes: str = Field(description="What changed after considering clinician feedback")
    keep_flagged: bool = Field(description="True if still flagged even after clinician dismissal context")


# ── Re-analysis prompt ─────────────────────────────────────────────────────────

_REANALYSIS_SYSTEM = """You are re-analysing a flagged clinical documentation gap after a clinician dismissed it.
The clinician's rationale provides additional context that was not available during the initial analysis.
Determine whether the gap remains clinically significant given this new context."""

_REANALYSIS_HUMAN = """ORIGINAL FLAGGED GAP:
{gap_description}
Original Severity: {original_severity}
Original Recommendation: {recommended_action}

CLINICIAN DISMISSAL RATIONALE:
{dismissal_notes}

DISCHARGE NOTE EXCERPT:
{discharge_note_excerpt}

Re-evaluate: is this gap still clinically significant given the clinician's context?"""


# ── Orchestrator ───────────────────────────────────────────────────────────────

class HITLOrchestrator:
    """
    Manages the human-in-the-loop review cycle.
    Dismissals trigger a re-evaluation pass (adaptive replanning).
    """

    def __init__(self, model: str = LLM_MODEL, temperature: float = LLM_TEMPERATURE):
        self._review_packages: dict = {}  # analysis_id → ReviewPackage
        self._audit_log: List[dict] = []

        llm = ChatOpenAI(model=model, temperature=temperature)
        reanalysis_prompt = ChatPromptTemplate.from_messages([
            ("system", _REANALYSIS_SYSTEM),
            ("human", _REANALYSIS_HUMAN),
        ])
        self._reanalysis_chain = reanalysis_prompt | llm.with_structured_output(ReevaluationResult)

    # ── Package creation ───────────────────────────────────────────────────

    def create_review_package(
        self, analysis_id: str, correction_result: SelfCorrectionResult
    ) -> ReviewPackage:
        """Convert SelfCorrectionResult into a ReviewPackage sorted by severity."""
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        items = [
            ReviewItem(
                item_id=f"{analysis_id}_gap_{i}",
                gap_type=gap.gap_type,
                description=gap.description,
                severity=gap.adjusted_severity,
                recommended_action=gap.recommended_action,
                care_setting_note=gap.care_setting_note,
            )
            for i, gap in enumerate(correction_result.verified_gaps)
        ]

        items.sort(key=lambda x: severity_order.get(x.severity, 99))

        package = ReviewPackage(
            analysis_id=analysis_id,
            review_items=items,
            total_gaps=len(items),
            critical_count=sum(1 for i in items if i.severity == "critical"),
            high_count=sum(1 for i in items if i.severity == "high"),
            medium_count=sum(1 for i in items if i.severity == "medium"),
            low_count=sum(1 for i in items if i.severity == "low"),
            created_at=datetime.now().isoformat(),
        )
        self._review_packages[analysis_id] = package
        logger.info(
            "[HITLOrchestrator] Review package created: %d gaps (%d critical, %d high)",
            package.total_gaps, package.critical_count, package.high_count,
        )
        return package

    # ── Feedback recording ─────────────────────────────────────────────────

    def record_feedback(
        self,
        analysis_id: str,
        item_id: str,
        action: str,          # "confirm" | "dismiss" | "escalate"
        notes: str = "",
        discharge_note: str = "",
    ) -> dict:
        """
        Record clinician feedback.
        If action is 'dismiss', triggers a re-evaluation loop with the dismissal context.
        """
        package = self._review_packages.get(analysis_id)
        if not package:
            return {"status": "error", "message": f"Analysis {analysis_id} not found"}

        item = next((i for i in package.review_items if i.item_id == item_id), None)
        if not item:
            return {"status": "error", "message": f"Item {item_id} not found"}

        _STATUS_MAP = {"confirm": "confirmed", "dismiss": "dismissed", "escalate": "escalated"}
        item.status = _STATUS_MAP.get(action, action)
        item.clinician_notes = notes
        item.reviewed_at = datetime.now().isoformat()

        self._audit_log.append({
            "analysis_id": analysis_id,
            "item_id": item_id,
            "action": action,
            "notes": notes,
            "timestamp": item.reviewed_at,
        })

        if action == "dismiss" and notes:
            return self._reenter_reasoning_loop(item, notes, discharge_note)

        logger.info("[HITLOrchestrator] Feedback recorded: %s → %s", item_id, action)
        return {"status": action, "item_id": item_id, "message": f"Gap marked as '{action}'."}

    def _reenter_reasoning_loop(
        self, item: ReviewItem, dismissal_notes: str, discharge_note: str
    ) -> dict:
        """
        Adaptive replanning: re-analyse the dismissed gap with the clinician's context.
        This is the key HITL agentic behaviour — dismissals are NOT simply filtered out.
        """
        logger.info("[HITLOrchestrator] Dismissal detected — re-entering reasoning loop for %s", item.item_id)
        try:
            reeval: ReevaluationResult = self._reanalysis_chain.invoke({
                "gap_description": item.description,
                "original_severity": item.severity,
                "recommended_action": item.recommended_action,
                "dismissal_notes": dismissal_notes,
                "discharge_note_excerpt": truncate_to_tokens(discharge_note, 2000) if discharge_note else "(not provided)",
            })

            if reeval.keep_flagged:
                item.severity = reeval.updated_severity
                item.recommended_action = reeval.updated_recommendation
                item.status = "re_flagged_after_dismissal"
                logger.info("[HITLOrchestrator] Gap re-flagged after dismissal: %s", item.item_id)
                return {
                    "status": "re_flagged_after_dismissal",
                    "item_id": item.item_id,
                    "message": "Gap remains clinically significant despite dismissal.",
                    "reanalysis": reeval.reanalysis_notes,
                    "updated_severity": reeval.updated_severity,
                    "updated_recommendation": reeval.updated_recommendation,
                }
            else:
                item.status = "dismissed_confirmed"
                logger.info("[HITLOrchestrator] Dismissal confirmed after re-analysis: %s", item.item_id)
                return {
                    "status": "dismissed_confirmed",
                    "item_id": item.item_id,
                    "message": "Dismissal confirmed — gap is not clinically significant given context.",
                    "reanalysis": reeval.reanalysis_notes,
                }
        except Exception as e:
            logger.error("[HITLOrchestrator] Re-analysis failed: %s", e)
            item.status = "dismissed"
            return {
                "status": "dismissed",
                "item_id": item.item_id,
                "message": "Dismissed (re-analysis unavailable).",
            }

    # ── Accessors ──────────────────────────────────────────────────────────

    def get_package(self, analysis_id: str) -> Optional[ReviewPackage]:
        return self._review_packages.get(analysis_id)

    def get_audit_log(self) -> List[dict]:
        return self._audit_log
