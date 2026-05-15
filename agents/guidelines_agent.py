"""
Guidelines Agent — Agent 3 of 5
Retrieves relevant ACC/AHA and AHRQ RED Toolkit chunks from ChromaDB,
then checks whether the discharge note satisfies each requirement.

V1 improvements:
  - Per-diagnosis targeted retrieval from TaskPlan instead of 3 generic queries
  - Token-aware context assembly and truncation
"""

import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import LLM_MODEL, LLM_TEMPERATURE
from rag.retriever import ContextRetriever, truncate_to_tokens
from agents.planning_agent import TaskPlan

logger = logging.getLogger(__name__)


# ── Output schema ──────────────────────────────────────────────────────────────

class GuidelineViolation(BaseModel):
    requirement: str = Field(description="The specific guideline requirement that is missing or incomplete")
    guideline_source: str = Field(description="Which guideline this comes from (e.g. ACC/AHA, AHRQ RED)")
    missing_element: str = Field(description="What is absent from the discharge note")
    severity: str = Field(description="critical | high | medium | low")
    clinical_rationale: str = Field(description="Why this matters clinically")

class GuidelinesResult(BaseModel):
    violations: List[GuidelineViolation] = Field(
        description="Guideline requirements not met by the discharge note"
    )
    compliant_areas: List[str] = Field(
        description="Areas where the note IS compliant with guidelines"
    )
    overall_compliant: bool = Field(
        description="True only if zero violations were found"
    )
    summary: str = Field(description="One-sentence compliance summary")


# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM = """You are a clinical quality assurance specialist validating a hospital discharge note
against ACC/AHA cardiology guidelines and AHRQ Re-Engineered Discharge (RED) Toolkit requirements.

Your task:
1. Review the retrieved guideline excerpts to understand what documentation is REQUIRED.
2. Check the discharge note to determine which requirements are MET vs MISSING.
3. Only flag genuine omissions — not style preferences or minor formatting issues."""

_HUMAN = """PATIENT DIAGNOSES:
{diagnoses}

RETRIEVED GUIDELINE REQUIREMENTS (from vector store):
{guideline_context}

DISCHARGE NOTE:
{discharge_note}

PLANNING AGENT IDENTIFIED THESE GAPS TO INVESTIGATE:
{gaps_to_investigate}

Evaluate the discharge note against the guideline requirements above.
List every missing required element as a GuidelineViolation."""


# ── Agent ──────────────────────────────────────────────────────────────────────

class GuidelinesAgent:
    """Uses plan-driven RAG retrieval to check guideline compliance."""

    def __init__(
        self,
        retriever: ContextRetriever,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
    ):
        self.retriever = retriever
        llm = ChatOpenAI(model=model, temperature=temperature)
        prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])
        self.chain = prompt | llm.with_structured_output(GuidelinesResult)

    def check_compliance(
        self,
        discharge_note: str,
        task_plan: TaskPlan,
    ) -> GuidelinesResult:
        diagnoses = task_plan.identified_diagnoses
        topics = task_plan.guideline_topics

        guideline_docs = self.retriever.get_guidelines_for_diagnoses(
            diagnoses=diagnoses,
            topics=topics,
            k_per_query=3,
        )

        guideline_context = self.retriever.docs_to_text(guideline_docs[:12], max_tokens=4000)

        if not guideline_context.strip():
            logger.warning("[GuidelinesAgent] No guideline chunks retrieved — skipping compliance check.")
            return GuidelinesResult(
                violations=[],
                compliant_areas=[],
                overall_compliant=True,
                summary="No guidelines loaded — compliance check skipped. Ingest guidelines first.",
            )

        try:
            result: GuidelinesResult = self.chain.invoke({
                "diagnoses": diagnoses[:10],
                "guideline_context": guideline_context,
                "discharge_note": truncate_to_tokens(discharge_note, 5000),
                "gaps_to_investigate": task_plan.critical_gaps_to_investigate,
            })
            logger.info(
                "[GuidelinesAgent] %d violations found; compliant=%s",
                len(result.violations), result.overall_compliant,
            )
            return result
        except Exception as e:
            logger.error("[GuidelinesAgent] Failed: %s", e)
            return GuidelinesResult(
                violations=[],
                compliant_areas=[],
                overall_compliant=False,
                summary=f"Guidelines check failed: {e}",
            )
