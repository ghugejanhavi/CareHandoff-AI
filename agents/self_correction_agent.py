"""
Self-Correction Agent — Agent 4 of 5
Re-verifies every flagged gap before it reaches the clinician.
Reduces false positives by re-reading the discharge note with the specific gap in mind.
Adjusts severity based on the receiving care setting (home vs SNF vs urgent clinic).
"""

import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import LLM_MODEL, LLM_TEMPERATURE
from rag.retriever import truncate_to_tokens
from agents.ehr_comparison_agent import EHRComparisonResult
from agents.guidelines_agent import GuidelinesResult, GuidelineViolation

logger = logging.getLogger(__name__)


# ── Output schema ──────────────────────────────────────────────────────────────

class VerifiedGap(BaseModel):
    gap_type: str = Field(description="medication_gap | lab_gap | guideline_violation")
    description: str = Field(description="What is missing and why it matters")
    original_severity: str = Field(description="Severity as initially flagged")
    adjusted_severity: str = Field(description="Severity after care-setting adjustment: critical | high | medium | low")
    is_real_gap: bool = Field(description="False if re-reading shows the note actually contains this information")
    rationale: str = Field(description="Evidence from the note supporting the is_real_gap determination")
    recommended_action: str = Field(description="Specific action for the receiving clinician or discharging team")
    care_setting_note: str = Field(description="How the care setting affects urgency of this gap")

class SelfCorrectionResult(BaseModel):
    verified_gaps: List[VerifiedGap] = Field(
        description="Gaps confirmed as real after re-verification"
    )
    dismissed_count: int = Field(
        description="Number of initially flagged items dismissed as false positives"
    )
    summary: str = Field(description="Summary of verification outcome")


# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM = """You are a senior clinical informaticist performing a second-pass review of flagged documentation gaps.

For each gap:
1. Re-read the discharge note carefully with that specific gap in mind.
2. Determine if the gap is REAL (truly missing) or a FALSE POSITIVE (information is present but phrased differently).
3. Adjust severity based on the receiving care setting:
   - home: patient/caregiver is responsible — higher severity for medication gaps
   - skilled_nursing_facility (SNF): nursing staff will manage — medium severity unless high-risk drug
   - urgent_clinic: physician needs complete info immediately — high severity for all gaps
4. Provide a concrete recommended action."""

_HUMAN = """DISCHARGE NOTE:
{discharge_note}

RECEIVING CARE SETTING: {care_setting}

FLAGGED GAPS TO VERIFY:
{flagged_gaps}

Re-verify each gap. Dismiss false positives. Adjust severity for the care setting."""


# ── Agent ──────────────────────────────────────────────────────────────────────

class SelfCorrectionAgent:
    """
    Re-verifies all gaps from EHR and Guidelines agents before HITL presentation.
    Batches all gaps into a single LLM call for efficiency.
    """

    def __init__(self, model: str = LLM_MODEL, temperature: float = 0.1):
        # Lower temperature for verification — we want deterministic re-reading
        llm = ChatOpenAI(model=model, temperature=temperature)
        prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])
        self.chain = prompt | llm.with_structured_output(SelfCorrectionResult)

    def verify(
        self,
        discharge_note: str,
        ehr_result: EHRComparisonResult,
        guidelines_result: GuidelinesResult,
        care_setting: str = "home",
    ) -> SelfCorrectionResult:
        """
        Verify all flagged gaps in a single pass.

        care_setting options: "home" | "skilled_nursing_facility" | "urgent_clinic"
        """
        flagged = self._build_gap_list(ehr_result, guidelines_result)

        if not flagged:
            return SelfCorrectionResult(
                verified_gaps=[],
                dismissed_count=0,
                summary="No gaps to verify.",
            )

        try:
            result: SelfCorrectionResult = self.chain.invoke({
                "discharge_note": truncate_to_tokens(discharge_note, 6000),
                "care_setting": care_setting,
                "flagged_gaps": flagged,
            })
            logger.info(
                "[SelfCorrectionAgent] %d confirmed gaps, %d dismissed | setting=%s",
                len(result.verified_gaps), result.dismissed_count, care_setting,
            )
            return result
        except Exception as e:
            logger.error("[SelfCorrectionAgent] Verification failed: %s", e)
            # Fallback: pass all through as-is
            return self._fallback_verification(flagged, care_setting)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _build_gap_list(
        self,
        ehr_result: EHRComparisonResult,
        guidelines_result: GuidelinesResult,
    ) -> List[dict]:
        gaps = []
        for gap in ehr_result.medication_gaps:
            gaps.append({
                "type": "medication_gap",
                "description": f"{gap.drug_name} — {gap.reason_flagged}",
                "severity": gap.severity,
            })
        for gap in ehr_result.lab_gaps:
            gaps.append({
                "type": "lab_gap",
                "description": f"{gap.lab_name} — {gap.reason_flagged}",
                "severity": gap.severity,
            })
        for alert in ehr_result.allergy_alerts:
            gaps.append({
                "type": "allergy_conflict",
                "description": (
                    f"ALLERGY SAFETY ALERT: {alert.drug_name} prescribed but patient has "
                    f"documented allergy to {alert.allergy} ({alert.conflict_type}). "
                    f"{alert.explanation}"
                ),
                "severity": alert.severity,
            })
        for disc in ehr_result.dose_discrepancies:
            gaps.append({
                "type": "dose_discrepancy",
                "description": (
                    f"DOSE MISMATCH: {disc.drug_name} — EHR: {disc.ehr_dose}, "
                    f"Note: {disc.note_dose}. {disc.discrepancy}. "
                    f"Risk: {disc.clinical_risk}"
                ),
                "severity": disc.severity,
            })
        for v in guidelines_result.violations:
            gaps.append({
                "type": "guideline_violation",
                "description": f"[{v.guideline_source}] {v.requirement}: {v.missing_element}",
                "severity": v.severity,
                "clinical_rationale": v.clinical_rationale,
            })
        return gaps

    def _fallback_verification(self, flagged: List[dict], care_setting: str) -> SelfCorrectionResult:
        """If LLM call fails, pass all gaps through without modification."""
        verified = [
            VerifiedGap(
                gap_type=g["type"],
                description=g["description"],
                original_severity=g["severity"],
                adjusted_severity=g["severity"],
                is_real_gap=True,
                rationale="Auto-verified (LLM unavailable)",
                recommended_action="Review manually",
                care_setting_note=f"Care setting: {care_setting}",
            )
            for g in flagged
        ]
        return SelfCorrectionResult(
            verified_gaps=verified,
            dismissed_count=0,
            summary="Fallback verification — all gaps passed through.",
        )
