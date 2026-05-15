"""
Planning Agent — Agent 1 of 5
Reads the full discharge note and decomposes the audit task BEFORE any retrieval.
This is the defining Agentic RAG behaviour: plan first, retrieve second.
"""

import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import LLM_MODEL, LLM_TEMPERATURE
from rag.retriever import truncate_to_tokens

logger = logging.getLogger(__name__)


# ── Output schema ──────────────────────────────────────────────────────────────

class TaskPlan(BaseModel):
    """Structured task plan produced before any retrieval begins."""
    identified_diagnoses: List[str] = Field(
        description="Primary and secondary diagnoses found in the discharge note"
    )
    medications_to_verify: List[str] = Field(
        description="Medications mentioned in the note that must be cross-checked against EHR"
    )
    pending_labs_to_check: List[str] = Field(
        description="Lab tests that appear pending or whose results should be in the note"
    )
    follow_up_requirements: List[str] = Field(
        description="Follow-up elements the note should contain (appointments, timing, PCP name)"
    )
    guideline_topics: List[str] = Field(
        description="Clinical topics to retrieve guidelines for (e.g. heart failure, hypertension)"
    )
    critical_gaps_to_investigate: List[str] = Field(
        description="Specific potential information gaps the downstream agents must investigate"
    )


# ── Agent ──────────────────────────────────────────────────────────────────────

_SYSTEM = """You are a senior clinical documentation auditor specialising in hospital discharge summaries.
Your job is to READ the discharge note carefully and produce a structured investigation plan.
Do NOT attempt to detect gaps yourself — that is done by downstream agents.
Focus only on decomposing what needs to be checked."""

_HUMAN = """DISCHARGE NOTE:
{discharge_note}

EHR CONTEXT (for reference only — do not cross-reference yet):
- Recorded diagnoses: {diagnoses}
- Medications in EHR: {med_count} drugs on record
- Lab tests completed: {lab_count} tests
- Pending labs: {pending_labs}

Produce a structured task plan that downstream agents will execute."""


class PlanningAgent:
    """
    Decomposes the discharge audit into a structured TaskPlan.
    Output is consumed by every downstream agent.
    """

    def __init__(self, model: str = LLM_MODEL, temperature: float = LLM_TEMPERATURE):
        llm = ChatOpenAI(model=model, temperature=temperature)
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("human", _HUMAN),
        ])
        self.chain = prompt | llm.with_structured_output(TaskPlan)

    def decompose(self, discharge_note: str, ehr_snapshot: dict) -> TaskPlan:
        """
        Args:
            discharge_note: raw text of the discharge summary
            ehr_snapshot:   dict from MIMICLoader.get_patient_ehr()
        Returns:
            TaskPlan with all fields populated
        """
        try:
            plan: TaskPlan = self.chain.invoke({
                "discharge_note": truncate_to_tokens(discharge_note, 6000),
                "diagnoses": ehr_snapshot.get("diagnoses", [])[:10],
                "med_count": len(ehr_snapshot.get("medications", [])),
                "lab_count": len(ehr_snapshot.get("lab_results", [])),
                "pending_labs": ehr_snapshot.get("pending_labs", []),
            })
            logger.info(
                "[PlanningAgent] Plan produced: %d diagnoses, %d gaps to investigate",
                len(plan.identified_diagnoses),
                len(plan.critical_gaps_to_investigate),
            )
            return plan
        except Exception as e:
            logger.error("[PlanningAgent] Failed: %s", e)
            return TaskPlan(
                identified_diagnoses=[],
                medications_to_verify=[],
                pending_labs_to_check=[],
                follow_up_requirements=[],
                guideline_topics=[],
                critical_gaps_to_investigate=["Planning agent failed — manual review required"],
            )
