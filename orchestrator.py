"""
Agentic RAG Orchestrator
Wires all agents into the pipeline.

Pipeline:
  1. PlanningAgent        — decompose task before any retrieval
  2. EHRComparisonAgent   — compare note vs EHR (meds, labs, allergies, doses)
  3. GuidelinesAgent      — check RAG-retrieved guideline compliance
  4. SelfCorrectionAgent  — re-verify all gaps, adjust severity by care setting
  5. HITLOrchestrator     — format for clinician review; handle feedback loop
  6. NoteRewriteAgent     — rewrite discharge note with confirmed gaps addressed
  7. RewriteValidator     — hallucination guard on [ADDED] content
"""

import concurrent.futures
import json
import logging
import os
from datetime import datetime
from typing import Optional, List, Tuple

import pandas as pd

from config import LLM_MODEL, REVISED_NOTES_FILE, NOTES_INGEST_LIMIT
from data.loaders.mimic_loader import MIMICLoader
from data.loaders.rxnorm_client import RxNormClient
from rag.vector_store import HealthcareVectorStore
from rag.retriever import ContextRetriever
from agents.planning_agent import PlanningAgent, TaskPlan
from agents.ehr_comparison_agent import EHRComparisonAgent, EHRComparisonResult
from agents.guidelines_agent import GuidelinesAgent, GuidelinesResult
from agents.self_correction_agent import SelfCorrectionAgent, SelfCorrectionResult
from agents.hitl_orchestrator import HITLOrchestrator, ReviewPackage
from agents.note_rewrite_agent import NoteRewriteAgent
from agents.rewrite_validator import RewriteValidator, ValidationResult

logger = logging.getLogger(__name__)


class AnalysisResult:
    """Container for one complete analysis run."""

    def __init__(
        self,
        analysis_id: str,
        hadm_id: Optional[str],
        task_plan: TaskPlan,
        ehr_result: EHRComparisonResult,
        guidelines_result: GuidelinesResult,
        correction_result: SelfCorrectionResult,
        review_package: ReviewPackage,
    ):
        self.analysis_id = analysis_id
        self.hadm_id = hadm_id
        self.task_plan = task_plan
        self.ehr_result = ehr_result
        self.guidelines_result = guidelines_result
        self.correction_result = correction_result
        self.review_package = review_package

    def summary(self) -> str:
        pkg = self.review_package
        return (
            f"Analysis {self.analysis_id} | hadm_id={self.hadm_id}\n"
            f"  Gaps: {pkg.total_gaps} total "
            f"({pkg.critical_count} critical, {pkg.high_count} high, "
            f"{pkg.medium_count} medium, {pkg.low_count} low)\n"
            f"  Dismissed in self-correction: {self.correction_result.dismissed_count}"
        )


class AgenticRAGOrchestrator:
    """
    Entry point for running a full discharge note audit.

    Usage:
        orchestrator = AgenticRAGOrchestrator()
        orchestrator.ensure_data_ingested()
        result = orchestrator.analyze(discharge_note=..., hadm_id=..., care_setting=...)
    """

    def __init__(self, model: str = LLM_MODEL):
        self._model = model

        # Infrastructure
        self.mimic = MIMICLoader()
        self.rxnorm = RxNormClient()
        self.vector_store = HealthcareVectorStore()
        self.retriever = ContextRetriever(self.vector_store)

        # Agents
        self.planning = PlanningAgent(model=model)
        self.ehr_agent = EHRComparisonAgent(
            retriever=self.retriever,
            rxnorm_client=self.rxnorm if self.rxnorm.is_available() else None,
            model=model,
        )
        self.guidelines_agent = GuidelinesAgent(retriever=self.retriever, model=model)
        self.correction_agent = SelfCorrectionAgent(model=model)
        self.hitl = HITLOrchestrator(model=model)
        self.rewrite_agent = NoteRewriteAgent(model=model)
        self.rewrite_validator = RewriteValidator(model=model)

    # ── Startup ────────────────────────────────────────────────────────────

    def ensure_data_ingested(self) -> dict:
        stats = self.vector_store.collection_stats()
        result = {"notes_ingested": False, "guidelines_ingested": False}

        if stats["notes_chunks"] == 0:
            logger.info("[Startup] Notes collection empty — ingesting discharge notes (limit=%s)...", NOTES_INGEST_LIMIT)
            self.ingest_discharge_notes(limit=NOTES_INGEST_LIMIT)
            result["notes_ingested"] = True
        else:
            logger.info("[Startup] Notes collection has %d chunks — skipping ingestion", stats["notes_chunks"])

        if stats["guidelines_chunks"] == 0:
            logger.info("[Startup] Guidelines collection empty — ingesting guidelines...")
            self.ingest_guidelines()
            result["guidelines_ingested"] = True
        else:
            logger.info("[Startup] Guidelines collection has %d chunks — skipping ingestion", stats["guidelines_chunks"])

        return result

    # ── Main pipeline ──────────────────────────────────────────────────────

    def analyze(
        self,
        discharge_note: str,
        hadm_id: Optional[str] = None,
        care_setting: str = "home",
    ) -> AnalysisResult:
        analysis_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("=" * 70)
        logger.info("START ANALYSIS  id=%s  hadm_id=%s  setting=%s", analysis_id, hadm_id, care_setting)

        if hadm_id:
            logger.info("[0/5] Loading EHR snapshot for hadm_id=%s", hadm_id)
            ehr_snapshot = self.mimic.get_patient_ehr(int(hadm_id))
        else:
            logger.warning("[0/5] No hadm_id provided — EHR snapshot will be empty")
            ehr_snapshot = {"diagnoses": [], "medications": [], "lab_results": [], "pending_labs": [], "procedures": []}

        logger.info("[1/5] PlanningAgent — decomposing task")
        task_plan = self.planning.decompose(discharge_note, ehr_snapshot)
        logger.info("      %d diagnoses, %d gaps to investigate",
                     len(task_plan.identified_diagnoses), len(task_plan.critical_gaps_to_investigate))

        logger.info("[2+3/5] EHRComparisonAgent + GuidelinesAgent — running in parallel")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            ehr_future = pool.submit(
                self.ehr_agent.compare,
                hadm_id=str(hadm_id) if hadm_id else "unknown",
                ehr_snapshot=ehr_snapshot,
                discharge_note=discharge_note,
            )
            guidelines_future = pool.submit(
                self.guidelines_agent.check_compliance,
                discharge_note,
                task_plan,
            )
            ehr_result = ehr_future.result()
            guidelines_result = guidelines_future.result()
        logger.info("      EHR: %d med gaps, %d lab gaps, %d allergy alerts, %d dose discrepancies",
                     len(ehr_result.medication_gaps), len(ehr_result.lab_gaps),
                     len(ehr_result.allergy_alerts), len(ehr_result.dose_discrepancies))
        logger.info("      Guidelines: compliant=%s, %d violations",
                     guidelines_result.overall_compliant, len(guidelines_result.violations))

        logger.info("[4/5] SelfCorrectionAgent — verifying gaps")
        correction_result = self.correction_agent.verify(
            discharge_note=discharge_note,
            ehr_result=ehr_result,
            guidelines_result=guidelines_result,
            care_setting=care_setting,
        )
        logger.info("      %d confirmed, %d dismissed",
                     len(correction_result.verified_gaps), correction_result.dismissed_count)

        logger.info("[5/5] HITLOrchestrator — creating review package")
        review_package = self.hitl.create_review_package(analysis_id, correction_result)
        logger.info("      Package ready: %d items (%d critical)", review_package.total_gaps, review_package.critical_count)

        logger.info("ANALYSIS COMPLETE  id=%s", analysis_id)

        return AnalysisResult(
            analysis_id=analysis_id,
            hadm_id=hadm_id,
            task_plan=task_plan,
            ehr_result=ehr_result,
            guidelines_result=guidelines_result,
            correction_result=correction_result,
            review_package=review_package,
        )

    # ── Note rewrite + validation ──────────────────────────────────────────

    def generate_revised_note(
        self,
        analysis_id: str,
        discharge_note: str,
        hadm_id: Optional[str] = None,
    ) -> Tuple[str, Optional[ValidationResult]]:
        """
        Collect confirmed/re-flagged gaps, rewrite the note, then validate
        every [ADDED] block against source data. Returns (revised_note, validation).
        """
        package = self.hitl.get_package(analysis_id)
        if not package:
            return "Error: analysis not found.", None

        actionable_items = [
            item for item in package.review_items
            if item.status in ("confirmed", "re_flagged_after_dismissal", "escalated")
        ]

        if not actionable_items:
            return discharge_note, None

        gap_descriptions = [
            f"[{item.severity.upper()} | {item.gap_type}] {item.description} — Action: {item.recommended_action}"
            for item in actionable_items
        ]

        ehr_snapshot = {}
        if hadm_id:
            try:
                ehr_snapshot = self.mimic.get_patient_ehr(int(hadm_id))
            except Exception:
                pass

        revised = self.rewrite_agent.rewrite(
            original_note=discharge_note,
            confirmed_gaps=gap_descriptions,
            ehr_snapshot=ehr_snapshot,
        )

        logger.info("[Orchestrator] Running hallucination guard on revised note...")
        validation = self.rewrite_validator.validate(
            revised_note=revised,
            original_note=discharge_note,
            confirmed_gaps=gap_descriptions,
            ehr_snapshot=ehr_snapshot,
        )
        logger.info(
            "[Orchestrator] Validation: %d/%d additions grounded, %d ungrounded",
            validation.grounded_count, validation.total_additions, validation.ungrounded_count,
        )

        return revised, validation

    # ── Save revised note ──────────────────────────────────────────────────

    def save_revised_note(
        self,
        hadm_id: str,
        analysis_id: str,
        original_note: str,
        revised_note: str,
        gaps_addressed: List[str],
    ) -> str:
        row = {
            "hadm_id": hadm_id,
            "analysis_id": analysis_id,
            "original_note": original_note,
            "revised_note": revised_note,
            "gaps_addressed": json.dumps(gaps_addressed),
            "revised_by": "system",
            "revised_at": datetime.now().isoformat(),
        }

        file_exists = os.path.exists(REVISED_NOTES_FILE)
        df = pd.DataFrame([row])

        os.makedirs(os.path.dirname(REVISED_NOTES_FILE), exist_ok=True)
        df.to_csv(REVISED_NOTES_FILE, mode="a", header=not file_exists, index=False)

        logger.info("[SaveRevisedNote] Saved revised note for hadm_id=%s to %s", hadm_id, REVISED_NOTES_FILE)
        return REVISED_NOTES_FILE

    # ── Data ingestion helpers ─────────────────────────────────────────────

    def ingest_discharge_notes(self, limit: Optional[int] = None) -> int:
        notes_df = self.mimic.load_discharge_notes()
        if limit:
            notes_df = notes_df.head(limit)
        return self.vector_store.ingest_discharge_notes(notes_df)

    def ingest_guidelines(self) -> int:
        from data.loaders.guidelines_loader import GuidelinesLoader
        loader = GuidelinesLoader()
        docs = loader.load_all()
        return self.vector_store.ingest_guidelines(docs)

    def data_status(self) -> dict:
        mimic_status = self.mimic.is_data_available()
        from data.loaders.guidelines_loader import GuidelinesLoader
        guidelines_available = GuidelinesLoader().is_data_available()
        store_stats = self.vector_store.collection_stats()
        return {
            "mimic_files": mimic_status,
            "guidelines_files": guidelines_available,
            "vector_store": store_stats,
            "rxnorm_api": self.rxnorm.is_available(),
        }
