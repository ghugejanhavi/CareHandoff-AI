"""
Rewrite Validation Agent — Post-rewrite hallucination guard.

After the NoteRewriteAgent produces a revised discharge note, this agent:
1. Extracts all [ADDED]...[/ADDED] blocks from the revised note
2. Sends each block to an LLM along with the EHR data and confirmed gaps
3. Verifies every addition is grounded in source data
4. Returns a validation report flagging any ungrounded content
"""

import re
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import LLM_MODEL, LLM_TEMPERATURE
from rag.retriever import truncate_to_tokens

logger = logging.getLogger(__name__)


class AddedBlock(BaseModel):
    index: int = Field(description="Position of this block in the note (1-indexed)")
    text: str = Field(description="The added text content")
    grounded: bool = Field(description="True if the content is supported by source data")
    source: str = Field(description="Which source supports this content (EHR, confirmed gap, or 'UNGROUNDED')")
    concern: str = Field(description="Empty if grounded; otherwise describes the hallucination risk")

class ValidationResult(BaseModel):
    validated_blocks: List[AddedBlock] = Field(description="Validation result for each [ADDED] block")
    total_additions: int = Field(description="Total number of [ADDED] blocks found")
    grounded_count: int = Field(description="Number of additions verified as grounded")
    ungrounded_count: int = Field(description="Number of additions flagged as potentially hallucinated")
    summary: str = Field(description="Overall validation summary")


_SYSTEM = """\
You are a clinical documentation auditor performing a HALLUCINATION CHECK on an AI-rewritten \
discharge note. The AI was asked to insert missing information based on confirmed documentation \
gaps and EHR data.

Your task: for EACH [ADDED] block, determine whether the inserted content is GROUNDED in the \
source data provided (EHR records and confirmed gaps). A block is grounded if:
- The drug name, dose, and route match the EHR medication records, OR
- The lab value or test name matches the EHR lab results, OR
- The clinical recommendation directly addresses a confirmed gap, OR
- The follow-up instruction is consistent with standard care for the documented diagnoses

A block is UNGROUNDED if:
- It contains a specific drug dose not present in the EHR
- It mentions a lab value not in the EHR data
- It references a follow-up date, physician name, or clinic not in the source data
- It makes a clinical claim that cannot be traced to either the EHR or confirmed gaps

Be strict: when in doubt, flag as ungrounded. Patient safety depends on this check."""

_HUMAN = """\
[ADDED] BLOCKS EXTRACTED FROM REVISED NOTE:
{added_blocks}

CONFIRMED GAPS THAT THE AI WAS ASKED TO ADDRESS:
{confirmed_gaps}

EHR DATA (ground truth):
{ehr_summary}

ORIGINAL DISCHARGE NOTE (for context):
{original_note_excerpt}

For each [ADDED] block, determine if it is grounded in the source data or potentially hallucinated."""


_ADDED_PATTERN = re.compile(r"\[ADDED\](.*?)\[/ADDED\]", re.DOTALL)


class RewriteValidator:
    """Validates that all [ADDED] content in a rewritten note is grounded in source data."""

    def __init__(self, model: str = LLM_MODEL, temperature: float = 0.1):
        llm = ChatOpenAI(model=model, temperature=temperature)
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("human", _HUMAN),
        ])
        self._chain = prompt | llm.with_structured_output(ValidationResult)

    def validate(
        self,
        revised_note: str,
        original_note: str,
        confirmed_gaps: List[str],
        ehr_snapshot: dict,
    ) -> ValidationResult:
        blocks = _ADDED_PATTERN.findall(revised_note)

        if not blocks:
            return ValidationResult(
                validated_blocks=[],
                total_additions=0,
                grounded_count=0,
                ungrounded_count=0,
                summary="No [ADDED] blocks found in revised note — nothing to validate.",
            )

        blocks_text = "\n".join(
            f"  Block {i+1}: {block.strip()}" for i, block in enumerate(blocks)
        )
        gaps_text = "\n".join(f"  {i+1}. {g}" for i, g in enumerate(confirmed_gaps))
        ehr_summary = self._format_ehr(ehr_snapshot)

        logger.info("[RewriteValidator] Validating %d [ADDED] blocks", len(blocks))

        try:
            result = self._chain.invoke({
                "added_blocks": truncate_to_tokens(blocks_text, 3000),
                "confirmed_gaps": truncate_to_tokens(gaps_text, 2000),
                "ehr_summary": truncate_to_tokens(ehr_summary, 2000),
                "original_note_excerpt": truncate_to_tokens(original_note, 2000),
            })
            logger.info(
                "[RewriteValidator] Validation complete: %d grounded, %d ungrounded out of %d",
                result.grounded_count, result.ungrounded_count, result.total_additions,
            )
            return result
        except Exception as e:
            logger.error("[RewriteValidator] Validation failed: %s", e)
            return ValidationResult(
                validated_blocks=[
                    AddedBlock(
                        index=i + 1,
                        text=block.strip(),
                        grounded=False,
                        source="VALIDATION_FAILED",
                        concern=f"Validation could not be performed: {e}",
                    )
                    for i, block in enumerate(blocks)
                ],
                total_additions=len(blocks),
                grounded_count=0,
                ungrounded_count=len(blocks),
                summary=f"Validation failed — all {len(blocks)} blocks flagged as unverified.",
            )

    @staticmethod
    def _format_ehr(ehr: dict) -> str:
        if not ehr:
            return "(No EHR data available)"

        parts = []

        med_details = ehr.get("medications_detail", [])
        if med_details:
            lines = [f"  - {m.get('drug', '?')} {m.get('dose_val_rx', '')} "
                     f"{m.get('dose_unit_rx', '')} {m.get('route', '')} "
                     f"[{m.get('starttime', '')} → {m.get('stoptime', 'ongoing')}]"
                     for m in med_details[:25]]
            parts.append("Medications:\n" + "\n".join(lines))
        else:
            meds = ehr.get("medications", [])
            if meds:
                parts.append("Medications:\n" + "\n".join(f"  - {m}" for m in meds[:25]))

        labs = ehr.get("lab_results", [])
        if labs:
            lines = [f"  - {lab.get('label', '?')}: {lab.get('value', '?')} "
                     f"(flag: {lab.get('flag', 'normal')})" for lab in labs[:20]]
            parts.append("Lab Results:\n" + "\n".join(lines))

        pending = ehr.get("pending_labs", [])
        if pending:
            parts.append("Pending Labs:\n" + "\n".join(f"  - {p}" for p in pending[:10]))

        diags = ehr.get("diagnoses", [])
        if diags:
            parts.append("Diagnoses:\n" + "\n".join(f"  - {d}" for d in diags[:15]))

        return "\n\n".join(parts) if parts else "(No structured EHR data)"
