"""
Note Rewrite Agent — Agent 6
Rewrites a discharge note to address confirmed documentation gaps.

Takes the original note, confirmed gaps from HITL review, and the EHR snapshot,
then produces a full rewritten note with [ADDED] annotations marking inserted content.
"""

import logging
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import LLM_MODEL, LLM_TEMPERATURE
from rag.retriever import truncate_to_tokens

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a clinical documentation specialist. Your task is to rewrite a hospital \
discharge summary so that it addresses every confirmed documentation gap listed below.

Rules:
1. Preserve ALL existing correct content — do not remove or alter any accurate information.
2. Insert missing medications into the "Discharge Medications:" section.
3. Insert pending lab follow-up plans into "Discharge Instructions:" or create that section if absent.
4. Insert any missing guideline-required elements (follow-up timing, weight monitoring, BP targets, etc.) \
into the most appropriate section.
5. Wrap every piece of newly inserted text with [ADDED] ... [/ADDED] markers so the clinician \
can quickly identify what changed.
6. Keep the same section structure and formatting as the original note.
7. Do NOT fabricate clinical facts. Only add information that is supported by the confirmed gaps \
and the EHR data provided.
8. Return the FULL rewritten note — not a diff or summary."""

_HUMAN_PROMPT = """\
ORIGINAL DISCHARGE NOTE:
{original_note}

CONFIRMED DOCUMENTATION GAPS TO ADDRESS:
{confirmed_gaps}

EHR DATA (for reference when inserting details):
{ehr_summary}

Rewrite the complete discharge note with the gaps addressed. Mark all additions with [ADDED]...[/ADDED]."""


class NoteRewriteAgent:
    """Rewrites a discharge note to incorporate confirmed documentation gaps."""

    def __init__(self, model: str = LLM_MODEL, temperature: float = LLM_TEMPERATURE):
        llm = ChatOpenAI(model=model, temperature=temperature)
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_PROMPT),
        ])
        self._chain = prompt | llm

    def rewrite(
        self,
        original_note: str,
        confirmed_gaps: List[str],
        ehr_snapshot: dict,
    ) -> str:
        if not confirmed_gaps:
            return original_note

        gaps_text = "\n".join(f"  {i+1}. {g}" for i, g in enumerate(confirmed_gaps))
        ehr_summary = self._format_ehr(ehr_snapshot)

        logger.info("[NoteRewriteAgent] Rewriting note with %d confirmed gaps", len(confirmed_gaps))

        result = self._chain.invoke({
            "original_note": truncate_to_tokens(original_note, 6000),
            "confirmed_gaps": gaps_text,
            "ehr_summary": truncate_to_tokens(ehr_summary, 2000),
        })

        revised = result.content.strip()
        logger.info("[NoteRewriteAgent] Rewrite complete (%d chars)", len(revised))
        return revised

    @staticmethod
    def _format_ehr(ehr: dict) -> str:
        if not ehr:
            return "(No EHR data available)"

        parts = []

        med_details = ehr.get("medications_detail", [])
        if med_details:
            med_lines = [f"  - {m.get('drug', 'Unknown')} ({m.get('route', '')}) "
                         f"[{m.get('starttime', '')} → {m.get('stoptime', '')}]" for m in med_details[:20]]
            parts.append("Medications:\n" + "\n".join(med_lines))
        else:
            meds = ehr.get("medications", [])
            if meds:
                med_lines = [f"  - {m}" for m in meds[:20]]
                parts.append("Medications:\n" + "\n".join(med_lines))

        labs = ehr.get("lab_results", [])
        if labs:
            lab_lines = [f"  - {lab.get('label', 'Unknown')}: {lab.get('value', '')} "
                         f"(flag: {lab.get('flag', 'normal')})" for lab in labs[:20]]
            parts.append("Lab Results:\n" + "\n".join(lab_lines))

        pending = ehr.get("pending_labs", [])
        if pending:
            pend_lines = [f"  - {p}" for p in pending[:10]]
            parts.append("Pending Labs:\n" + "\n".join(pend_lines))

        diags = ehr.get("diagnoses", [])
        if diags:
            diag_lines = [f"  - {d}" for d in diags[:15]]
            parts.append("Diagnoses:\n" + "\n".join(diag_lines))

        return "\n\n".join(parts) if parts else "(No structured EHR data)"
