"""
EHR Comparison Agent — Agent 2 of 5
Compares the discharge note against the patient's actual MIMIC-IV EHR records.
Detects:
  - Medication omissions
  - Undocumented pending lab results
  - Allergy-drug conflicts (V2)
  - Dose/frequency discrepancies between EHR and note (V2)

Uses RxNorm normalization to match drug names across naming variants.

V2 improvements:
  - Allergy-drug cross-check against discharge medications
  - Dose/frequency verification comparing EHR med details vs note
"""

import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import LLM_MODEL, LLM_TEMPERATURE
from rag.retriever import ContextRetriever, parse_note_sections, truncate_to_tokens

logger = logging.getLogger(__name__)


# ── Output schemas ─────────────────────────────────────────────────────────────

class MedicationGap(BaseModel):
    drug_name: str = Field(description="Name of the medication as it appears in EHR")
    reason_flagged: str = Field(description="Why this medication should be in the discharge note")
    severity: str = Field(description="critical | high | medium | low")

class LabGap(BaseModel):
    lab_name: str = Field(description="Name of the pending lab test")
    reason_flagged: str = Field(description="Why this result must be documented")
    severity: str = Field(description="critical | high | medium | low")

class AllergyAlert(BaseModel):
    drug_name: str = Field(description="Discharge medication that may conflict with a documented allergy")
    allergy: str = Field(description="The documented allergy it conflicts with")
    conflict_type: str = Field(description="direct | cross-reactive | class-related")
    explanation: str = Field(description="Clinical explanation of the allergy-drug relationship")
    severity: str = Field(description="critical | high | medium | low")

class DoseDiscrepancy(BaseModel):
    drug_name: str = Field(description="Name of the medication with a discrepancy")
    ehr_dose: str = Field(description="Dose/frequency/route as recorded in EHR")
    note_dose: str = Field(description="Dose/frequency/route as written in discharge note")
    discrepancy: str = Field(description="What specifically differs")
    clinical_risk: str = Field(description="Why this discrepancy matters clinically")
    severity: str = Field(description="critical | high | medium | low")

class AllergyCheckResult(BaseModel):
    alerts: List[AllergyAlert] = Field(description="Allergy-drug conflicts found")
    summary: str = Field(description="One-sentence summary")

class DoseCheckResult(BaseModel):
    discrepancies: List[DoseDiscrepancy] = Field(description="Dose/frequency mismatches found")
    summary: str = Field(description="One-sentence summary")

class EHRComparisonResult(BaseModel):
    medication_gaps: List[MedicationGap] = Field(
        description="Medications in EHR that are missing from the discharge note"
    )
    lab_gaps: List[LabGap] = Field(
        description="Pending lab results not mentioned in discharge documentation"
    )
    allergy_alerts: List[AllergyAlert] = Field(
        default_factory=list,
        description="Discharge medications that may conflict with documented allergies",
    )
    dose_discrepancies: List[DoseDiscrepancy] = Field(
        default_factory=list,
        description="Medications where the note dose differs from the EHR dose",
    )
    summary: str = Field(description="One-sentence summary of findings")


# ── Prompts ────────────────────────────────────────────────────────────────────

_MED_SYSTEM = """You are a clinical pharmacist reviewing a hospital discharge note for medication omissions.
Your task: identify medications documented in the EHR that are absent from or inadequately described in the discharge note.
Only flag clinically significant omissions — not duplicates or formatting differences."""

_MED_HUMAN = """EHR MEDICATIONS (ground truth from hospital system):
{ehr_medications}

DISCHARGE NOTE — MEDICATIONS ON ADMISSION SECTION:
{note_admission_meds}

DISCHARGE NOTE — DISCHARGE MEDICATIONS SECTION:
{note_discharge_meds}

MEDICATIONS MENTIONED ELSEWHERE IN NOTE:
{note_medications_mentioned}

RxNorm-normalized EHR drug names (for accurate matching):
{normalized_names}

Identify medication gaps. For each gap state severity: critical (high-risk drug), high (chronic disease), medium, or low."""

_LAB_SYSTEM = """You are a clinical nurse reviewing discharge documentation for pending lab result omissions.
Pending lab results MUST be referenced in discharge instructions so the receiving provider knows to follow up.
Flag any pending results not mentioned in the discharge note."""

_LAB_HUMAN = """PENDING LAB TESTS (from EHR — results not yet available at discharge):
{pending_labs}

DISCHARGE NOTE — PERTINENT RESULTS SECTION:
{note_results_section}

DISCHARGE NOTE — DISCHARGE INSTRUCTIONS SECTION:
{note_instructions_section}

Identify which pending labs are NOT mentioned in the discharge note.
Severity: critical = culture results, critical values; high = metabolic panels; medium = routine follow-up."""

_ALLERGY_SYSTEM = """You are a clinical pharmacist specialising in drug safety.
Your task: check whether any DISCHARGE MEDICATIONS could conflict with the patient's DOCUMENTED ALLERGIES.

Consider:
- Direct conflicts (e.g., penicillin allergy → amoxicillin prescribed)
- Cross-reactivity (e.g., penicillin allergy → cephalosporin risk, sulfa allergy → certain diuretics/sulfonylureas)
- Drug-class relationships (e.g., sulfonamide allergy may affect thiazide diuretics, celecoxib, some anticonvulsants)

Only flag clinically meaningful conflicts. Do NOT flag:
- Allergies with no pharmacological relationship to any discharge medication
- Mild/theoretical risks with no clinical evidence of cross-reactivity

Severity guide:
- critical: direct allergy match or high cross-reactivity risk (e.g., penicillin allergy + amoxicillin)
- high: well-documented class cross-reactivity (e.g., penicillin allergy + 1st-gen cephalosporin)
- medium: possible but lower-probability cross-reactivity
- low: theoretical concern only"""

_ALLERGY_HUMAN = """PATIENT ALLERGIES (from discharge note):
{allergies}

DISCHARGE MEDICATIONS (from discharge note):
{discharge_meds}

EHR MEDICATION DETAILS (for reference):
{ehr_med_details}

Check every discharge medication against the patient's allergies. Report any conflicts."""

_DOSE_SYSTEM = """You are a clinical pharmacist performing medication dose reconciliation.
Compare the medication doses in the DISCHARGE NOTE against the doses in the EHR (hospital pharmacy system).

Flag discrepancies where:
- The dose amount differs (e.g., EHR says 50mg, note says 25mg)
- The frequency differs (e.g., EHR says BID, note says daily)
- The route differs (e.g., EHR says IV, note says PO) — unless the switch is clinically expected at discharge
- A dose change during hospitalization is not clearly documented

Do NOT flag:
- Medications where the note explicitly documents a dose CHANGE with rationale
- Trivial numeric formatting differences (e.g., 500.0 vs 500, 1000 vs 1,000, 0.125 vs 0.13)
- The SAME dose amount written differently (e.g., "500 mg" and "500.0 mg" are identical)
- Route changes from IV to PO that are normal for discharge transitions
- Frequency that the EHR does not specify — if the EHR record has no frequency but the note does, that is NOT a discrepancy (the note is more complete)

Severity guide:
- critical: >2x dose difference on high-risk drugs (anticoagulants, insulin, digoxin, opioids)
- high: significant dose difference on chronic medications, or wrong frequency
- medium: moderate discrepancy on lower-risk medications
- low: minor differences unlikely to cause harm"""

_DOSE_HUMAN = """EHR MEDICATION RECORDS (pharmacy system — includes dose, unit, route):
{ehr_med_details}

DISCHARGE NOTE — DISCHARGE MEDICATIONS SECTION:
{note_discharge_meds}

DISCHARGE NOTE — MEDICATIONS ON ADMISSION SECTION:
{note_admission_meds}

BRIEF HOSPITAL COURSE (for context on intentional dose changes):
{hospital_course}

Compare each medication's dose/frequency/route between the EHR records and the discharge note. Report discrepancies."""


# ── Agent ──────────────────────────────────────────────────────────────────────

class EHRComparisonAgent:
    """
    Detects medication gaps, lab gaps, allergy conflicts, and dose discrepancies
    by comparing the discharge note against structured MIMIC-IV EHR data.
    """

    def __init__(
        self,
        retriever: ContextRetriever,
        rxnorm_client=None,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
    ):
        self.retriever = retriever
        self.rxnorm = rxnorm_client

        llm = ChatOpenAI(model=model, temperature=temperature)

        med_prompt = ChatPromptTemplate.from_messages([("system", _MED_SYSTEM), ("human", _MED_HUMAN)])
        lab_prompt = ChatPromptTemplate.from_messages([("system", _LAB_SYSTEM), ("human", _LAB_HUMAN)])
        allergy_prompt = ChatPromptTemplate.from_messages([("system", _ALLERGY_SYSTEM), ("human", _ALLERGY_HUMAN)])
        dose_prompt = ChatPromptTemplate.from_messages([("system", _DOSE_SYSTEM), ("human", _DOSE_HUMAN)])

        self._med_chain = med_prompt | llm.with_structured_output(EHRComparisonResult)
        self._lab_chain = lab_prompt | llm.with_structured_output(EHRComparisonResult)
        self._allergy_chain = allergy_prompt | llm.with_structured_output(AllergyCheckResult)
        self._dose_chain = dose_prompt | llm.with_structured_output(DoseCheckResult)

    def compare(self, hadm_id: str, ehr_snapshot: dict, discharge_note: str) -> EHRComparisonResult:
        """Run medication, lab, allergy, and dose comparisons and merge results."""
        sections = parse_note_sections(discharge_note)

        med_result = self._compare_medications(ehr_snapshot, discharge_note, sections)
        lab_result = self._compare_labs(ehr_snapshot, sections)
        allergy_result = self._check_allergies(ehr_snapshot, sections)
        dose_result = self._check_doses(ehr_snapshot, sections)

        merged = EHRComparisonResult(
            medication_gaps=med_result.medication_gaps,
            lab_gaps=lab_result.lab_gaps,
            allergy_alerts=allergy_result.alerts,
            dose_discrepancies=dose_result.discrepancies,
            summary=(
                f"Found {len(med_result.medication_gaps)} medication gaps, "
                f"{len(lab_result.lab_gaps)} lab gaps, "
                f"{len(allergy_result.alerts)} allergy alerts, "
                f"and {len(dose_result.discrepancies)} dose discrepancies."
            ),
        )
        logger.info(
            "[EHRComparisonAgent] %d med gaps, %d lab gaps, %d allergy alerts, %d dose issues for hadm_id=%s",
            len(merged.medication_gaps), len(merged.lab_gaps),
            len(merged.allergy_alerts), len(merged.dose_discrepancies), hadm_id,
        )
        return merged

    # ── Internal ───────────────────────────────────────────────────────────

    def _compare_medications(
        self, ehr_snapshot: dict, discharge_note: str, sections: dict,
    ) -> EHRComparisonResult:
        ehr_meds = ehr_snapshot.get("medications", [])
        if not ehr_meds:
            return EHRComparisonResult(medication_gaps=[], lab_gaps=[], summary="No EHR medications to compare.")

        normalized = {}
        if self.rxnorm:
            try:
                normalized = self.rxnorm.normalize_drug_list(ehr_meds[:50])
            except Exception:
                pass

        admission_meds = sections.get("Medications on Admission", "(section not found in note)")
        discharge_meds = sections.get("Discharge Medications", "(section not found in note)")
        note_med_mentions = _extract_med_mentions(discharge_note)

        try:
            return self._med_chain.invoke({
                "ehr_medications": ehr_meds[:60],
                "note_admission_meds": truncate_to_tokens(admission_meds, 2000),
                "note_discharge_meds": truncate_to_tokens(discharge_meds, 2000),
                "note_medications_mentioned": note_med_mentions,
                "normalized_names": normalized or "RxNorm not available",
            })
        except Exception as e:
            logger.error("[EHRComparisonAgent] Medication comparison failed: %s", e)
            return EHRComparisonResult(medication_gaps=[], lab_gaps=[], summary="Comparison failed.")

    def _compare_labs(self, ehr_snapshot: dict, sections: dict) -> EHRComparisonResult:
        pending = ehr_snapshot.get("pending_labs", [])
        if not pending:
            return EHRComparisonResult(medication_gaps=[], lab_gaps=[], summary="No pending labs.")

        results_section = sections.get("Pertinent Results", "(section not found in note)")
        instructions_section = sections.get("Discharge Instructions", "(section not found in note)")

        try:
            return self._lab_chain.invoke({
                "pending_labs": pending[:20],
                "note_results_section": truncate_to_tokens(results_section, 2000),
                "note_instructions_section": truncate_to_tokens(instructions_section, 2000),
            })
        except Exception as e:
            logger.error("[EHRComparisonAgent] Lab comparison failed: %s", e)
            return EHRComparisonResult(medication_gaps=[], lab_gaps=[], summary="Lab comparison failed.")

    def _check_allergies(self, ehr_snapshot: dict, sections: dict) -> AllergyCheckResult:
        allergies_text = sections.get("Allergies", "")
        discharge_meds_text = sections.get("Discharge Medications", "")

        if not allergies_text.strip() or allergies_text.strip().lower() in ("nkda", "no known drug allergies", "none"):
            logger.info("[EHRComparisonAgent] No allergies documented — skipping allergy cross-check.")
            return AllergyCheckResult(alerts=[], summary="No allergies documented.")

        if not discharge_meds_text.strip():
            return AllergyCheckResult(alerts=[], summary="No discharge medications found in note.")

        med_details = ehr_snapshot.get("medications_detail", [])
        ehr_med_str = _format_med_details(med_details) if med_details else "(not available)"

        try:
            result = self._allergy_chain.invoke({
                "allergies": truncate_to_tokens(allergies_text, 500),
                "discharge_meds": truncate_to_tokens(discharge_meds_text, 2000),
                "ehr_med_details": truncate_to_tokens(ehr_med_str, 1500),
            })
            logger.info("[EHRComparisonAgent] Allergy check: %d alerts", len(result.alerts))
            return result
        except Exception as e:
            logger.error("[EHRComparisonAgent] Allergy check failed: %s", e)
            return AllergyCheckResult(alerts=[], summary=f"Allergy check failed: {e}")

    def _check_doses(self, ehr_snapshot: dict, sections: dict) -> DoseCheckResult:
        med_details = ehr_snapshot.get("medications_detail", [])
        discharge_meds_text = sections.get("Discharge Medications", "")

        if not med_details:
            logger.info("[EHRComparisonAgent] No EHR medication details — skipping dose verification.")
            return DoseCheckResult(discrepancies=[], summary="No EHR medication details available.")

        if not discharge_meds_text.strip():
            return DoseCheckResult(discrepancies=[], summary="No discharge medications found in note.")

        ehr_med_str = _format_med_details(med_details)
        admission_meds_text = sections.get("Medications on Admission", "(section not found)")
        hospital_course = sections.get("Brief Hospital Course", "(section not found)")

        try:
            result = self._dose_chain.invoke({
                "ehr_med_details": truncate_to_tokens(ehr_med_str, 2000),
                "note_discharge_meds": truncate_to_tokens(discharge_meds_text, 2000),
                "note_admission_meds": truncate_to_tokens(admission_meds_text, 1500),
                "hospital_course": truncate_to_tokens(hospital_course, 2000),
            })
            logger.info("[EHRComparisonAgent] Dose check: %d discrepancies", len(result.discrepancies))
            return result
        except Exception as e:
            logger.error("[EHRComparisonAgent] Dose check failed: %s", e)
            return DoseCheckResult(discrepancies=[], summary=f"Dose check failed: {e}")


def _extract_med_mentions(text: str) -> List[str]:
    lines = text.split("\n")
    med_keywords = {"mg", "tablet", "capsule", "patch", "injection", "daily", "twice", "oral", "iv ", "sc "}
    return [
        line.strip()
        for line in lines
        if any(kw in line.lower() for kw in med_keywords) and len(line.strip()) > 5
    ][:30]


def _clean_dose(val) -> str:
    """Convert 500.0 → '500', 0.125 → '0.125', missing → ''."""
    if val is None or val == "":
        return ""
    try:
        f = float(val)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError):
        return str(val)


def _format_med_details(med_details: list) -> str:
    lines = []
    for m in med_details[:30]:
        drug = m.get("drug", "Unknown")
        dose = _clean_dose(m.get("dose_val_rx", ""))
        unit = m.get("dose_unit_rx", "")
        route = m.get("route", "")
        start = m.get("starttime", "")
        stop = m.get("stoptime", "")
        lines.append(f"  - {drug} {dose} {unit} {route} [{start} → {stop or 'ongoing'}]")
    return "\n".join(lines)
