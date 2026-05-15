"""
CareHandoff AI — REST API Routes
All endpoints are thin wrappers around the existing pipeline logic.
Long-running LLM calls are offloaded to a thread pool via asyncio.to_thread.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.state import get_state

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response Models ──────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    hadm_id: str
    care_setting: str


class FeedbackRequest(BaseModel):
    item_id: str
    action: str          # confirm | dismiss | escalate
    notes: str = ""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _severity(obj) -> str:
    return str(getattr(obj, "severity", "low")).lower()


def _serialize_medication_gap(g) -> dict:
    return {"drug_name": g.drug_name, "severity": _severity(g), "reason_flagged": g.reason_flagged}


def _serialize_lab_gap(g) -> dict:
    return {"lab_name": g.lab_name, "severity": _severity(g), "reason_flagged": g.reason_flagged}


def _serialize_allergy(a) -> dict:
    return {
        "drug_name": a.drug_name,
        "allergy": a.allergy,
        "conflict_type": a.conflict_type,
        "severity": _severity(a),
        "explanation": a.explanation,
    }


def _serialize_dose(d) -> dict:
    return {
        "drug_name": d.drug_name,
        "ehr_dose": str(getattr(d, "ehr_dose", "")),
        "note_dose": str(getattr(d, "note_dose", "")),
        "discrepancy": str(getattr(d, "discrepancy", "")),
        "clinical_risk": str(getattr(d, "clinical_risk", "")),
        "severity": _severity(d),
    }


def _serialize_violation(v) -> dict:
    return {
        "guideline_source": str(getattr(v, "guideline_source", "")),
        "missing_element": str(getattr(v, "missing_element", "")),
        "severity": _severity(v),
    }


def _serialize_review_item(item) -> dict:
    return {
        "item_id": item.item_id,
        "severity": _severity(item),
        "gap_type": str(getattr(item, "gap_type", "")),
        "description": str(getattr(item, "description", "")),
        "recommended_action": str(getattr(item, "recommended_action", "")),
        "care_setting_note": str(getattr(item, "care_setting_note", "")),
        "status": str(getattr(item, "status", "pending_review")),
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    try:
        stats = get_state().orchestrator.vector_store.collection_stats()
    except Exception:
        stats = {}
    notes = stats.get("notes_chunks", 0)
    guidelines = stats.get("guidelines_chunks", 0)
    ingested = notes > 0 and guidelines > 0
    return {
        "status": "ok",
        "ingestion_complete": ingested,
        "notes_chunks": notes,
        "guidelines_chunks": guidelines,
    }


@router.get("/patients")
async def list_patients():
    state = get_state()
    ids = await asyncio.to_thread(
        state.orchestrator.mimic.list_available_admissions, 243
    )
    return {"patients": [str(i) for i in ids]}


@router.post("/analyze")
async def analyze(req: AnalyzeRequest):
    state = get_state()

    def _run():
        notes_df = state.orchestrator.mimic.load_discharge_notes()
        row = notes_df[notes_df["hadm_id"] == int(req.hadm_id)]
        if row.empty:
            raise ValueError("No discharge note found for this patient.")
        discharge_note = str(row.iloc[0]["text"])
        state.current_note = discharge_note

        result = state.orchestrator.analyze(
            discharge_note=discharge_note,
            hadm_id=req.hadm_id,
            care_setting=req.care_setting,
        )
        state.current_result = result
        state.revised_note = ""
        state.validation_result = None
        return discharge_note, result

    try:
        discharge_note, result = await asyncio.to_thread(_run)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {type(e).__name__}")

    ehr = result.ehr_result
    pkg = result.review_package
    gl = result.guidelines_result
    tp = result.task_plan

    return {
        "analysis_id": result.analysis_id,
        "hadm_id": req.hadm_id,
        "summary": {
            "critical": pkg.critical_count,
            "high": pkg.high_count,
            "medium": pkg.medium_count,
            "low": pkg.low_count,
            "total": pkg.total_gaps,
        },
        "diagnoses": tp.identified_diagnoses,
        "medications_reviewed": tp.medications_to_verify,
        "labs_to_follow_up": tp.pending_labs_to_check,
        "areas_of_concern": tp.critical_gaps_to_investigate,
        "medication_gaps": [_serialize_medication_gap(g) for g in ehr.medication_gaps],
        "lab_gaps": [_serialize_lab_gap(g) for g in ehr.lab_gaps],
        "allergy_alerts": [_serialize_allergy(a) for a in ehr.allergy_alerts],
        "dose_discrepancies": [_serialize_dose(d) for d in ehr.dose_discrepancies],
        "guideline_violations": [_serialize_violation(v) for v in gl.violations],
        "compliant_areas": gl.compliant_areas,
        "overall_compliant": gl.overall_compliant,
        "discharge_note": discharge_note,
    }


@router.get("/review")
async def get_review():
    state = get_state()
    if not state.current_result:
        raise HTTPException(status_code=404, detail="No analysis loaded. Run a discharge review first.")
    pkg = state.current_result.review_package
    items = [_serialize_review_item(i) for i in pkg.review_items]
    pending = sum(1 for i in items if i["status"] == "pending_review")
    return {"items": items, "total": len(items), "pending": pending, "acted": len(items) - pending}


@router.post("/review/feedback")
async def submit_feedback(req: FeedbackRequest):
    state = get_state()
    if not state.current_result:
        raise HTTPException(status_code=404, detail="No analysis loaded.")

    def _run():
        return state.orchestrator.hitl.record_feedback(
            analysis_id=state.current_result.analysis_id,
            item_id=req.item_id,
            action=req.action,
            notes=req.notes,
            discharge_note=state.current_note,
        )

    try:
        result = await asyncio.to_thread(_run)
    except Exception as e:
        logger.exception("Feedback failed")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": result.get("status", "ok"),
        "message": result.get("message", ""),
        "reanalysis": result.get("reanalysis", ""),
    }


@router.post("/review/confirm-all")
async def confirm_all():
    state = get_state()
    if not state.current_result:
        raise HTTPException(status_code=404, detail="No analysis loaded.")

    def _run():
        pkg = state.current_result.review_package
        count = 0
        for item in pkg.review_items:
            if item.status == "pending_review":
                state.orchestrator.hitl.record_feedback(
                    analysis_id=state.current_result.analysis_id,
                    item_id=item.item_id,
                    action="confirm",
                    notes="Batch confirmed by clinician.",
                    discharge_note=state.current_note,
                )
                count += 1
        return count

    count = await asyncio.to_thread(_run)
    return {"confirmed": count, "message": f"{count} items confirmed."}


@router.post("/note/generate")
async def generate_note():
    state = get_state()
    if not state.current_result:
        raise HTTPException(status_code=404, detail="No analysis loaded.")

    pkg = state.current_result.review_package
    actionable = [
        i for i in pkg.review_items
        if i.status in ("confirmed", "re_flagged_after_dismissal", "escalated")
    ]
    pending = [i for i in pkg.review_items if i.status == "pending_review"]

    if pending:
        raise HTTPException(
            status_code=400,
            detail=f"{len(pending)} items still need review. Confirm, dismiss, or escalate each one first."
        )
    if not actionable:
        raise HTTPException(status_code=400, detail="No confirmed items to address.")

    def _run():
        revised, validation = state.orchestrator.generate_revised_note(
            analysis_id=state.current_result.analysis_id,
            discharge_note=state.current_note,
            hadm_id=state.current_result.hadm_id,
        )
        state.revised_note = revised
        state.validation_result = validation
        return revised, validation

    try:
        revised, validation = await asyncio.to_thread(_run)
    except Exception as e:
        logger.exception("Note generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {type(e).__name__}")

    return {"revised_note": revised, "gaps_addressed": len(actionable)}


@router.get("/note/revised")
async def get_revised_note():
    state = get_state()
    if not state.revised_note:
        raise HTTPException(status_code=404, detail="No revised note yet.")
    return {"revised_note": state.revised_note}


@router.get("/note/validation")
async def get_validation():
    state = get_state()
    if not state.validation_result:
        raise HTTPException(status_code=404, detail="No validation report yet.")
    v = state.validation_result
    blocks = []
    for b in v.validated_blocks:
        blocks.append({
            "index": b.index,
            "text": b.text,
            "grounded": b.grounded,
            "source": str(getattr(b, "source", "")),
            "concern": getattr(b, "concern", None),
        })
    return {
        "total_additions": v.total_additions,
        "grounded_count": v.grounded_count,
        "ungrounded_count": v.ungrounded_count,
        "summary": v.summary,
        "blocks": blocks,
    }


@router.post("/note/save")
async def save_note():
    state = get_state()
    if not state.current_result:
        raise HTTPException(status_code=404, detail="No analysis loaded.")
    if not state.revised_note:
        raise HTTPException(status_code=404, detail="No revised note to save.")

    pkg = state.current_result.review_package
    gaps = [
        f"[{i.severity.upper()} | {i.gap_type}] {i.description}"
        for i in pkg.review_items
        if i.status in ("confirmed", "re_flagged_after_dismissal", "escalated")
    ]

    def _run():
        state.orchestrator.save_revised_note(
            hadm_id=str(state.current_result.hadm_id or "unknown"),
            analysis_id=state.current_result.analysis_id,
            original_note=state.current_note,
            revised_note=state.revised_note,
            gaps_addressed=gaps,
        )

    try:
        await asyncio.to_thread(_run)
    except Exception as e:
        logger.exception("Save failed")
        raise HTTPException(status_code=500, detail="Could not save the note.")

    return {"success": True, "message": "Note saved successfully."}
