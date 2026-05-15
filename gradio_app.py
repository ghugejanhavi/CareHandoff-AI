"""
Gradio Web Interface — CareHandoff AI
Tabs:
  1. Patient Analysis  — select patient, run discharge review
  2. Clinician Review  — confirm, dismiss, or escalate flagged items
  3. Revised Note      — generate, validate, and save the updated note
  4. Clinical Q&A      — ask questions about clinical documentation guidelines
"""

import logging
import gradio as gr
from typing import Optional

from orchestrator import AgenticRAGOrchestrator, AnalysisResult
from agents.rewrite_validator import ValidationResult
from qa.clinical_qa import MetaIntelligentClinicalRAG, GuardedClinicalRAG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


# ── CSS overrides ──────────────────────────────────────────────────────────────

_CSS = """
/* Base */
body, .gradio-container {
    background: #F9F8F5 !important;
    color: #111827 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, sans-serif !important;
}

/* Tab nav — blue underline only, no fill */
.tab-nav { border-bottom: 1px solid #E5E7EB !important; }
.tab-nav button {
    background: transparent !important;
    color: #6B7280 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-weight: 500 !important;
    padding: 12px 16px !important;
    margin-bottom: -1px !important;
}
.tab-nav button.selected {
    color: #2563EB !important;
    border-bottom: 2px solid #2563EB !important;
    background: transparent !important;
    font-weight: 600 !important;
}

/* Primary buttons — pill, blue fill */
button.primary {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border-radius: 9999px !important;
    border: none !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    transition: background 0.15s !important;
}
button.primary:hover { background: #1D4ED8 !important; }

/* Secondary buttons — ghost, blue border */
button.secondary {
    background: transparent !important;
    color: #2563EB !important;
    border: 1.5px solid #2563EB !important;
    border-radius: 9999px !important;
    font-weight: 500 !important;
    padding: 9px 24px !important;
}
button.secondary:hover { background: #EFF6FF !important; }

/* Input fields */
input[type="text"], textarea {
    border: 1px solid #D1D5DB !important;
    border-radius: 8px !important;
    background: #FFFFFF !important;
    color: #111827 !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.08) !important;
    outline: none !important;
}

/* Labels */
label > span, .block > label > span {
    color: #374151 !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
}

/* Helper/info text */
.gr-form .info, .info-text {
    color: #6B7280 !important;
    font-size: 0.8rem !important;
}

/* Clinical output areas — monospace, left accent */
.clinical-output textarea {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace !important;
    font-size: 12.5px !important;
    line-height: 1.65 !important;
    border-left: 3px solid #2563EB !important;
    border-radius: 0 8px 8px 0 !important;
    background: #FAFAFA !important;
    color: #111827 !important;
    padding-left: 14px !important;
}

/* Accordion */
.accordion {
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    background: #FFFFFF !important;
}

/* Save banner */
.save-banner { padding: 2px 0; }
"""


# ── App state ──────────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.orchestrator = AgenticRAGOrchestrator()
        self.current_result: Optional[AnalysisResult] = None
        self.current_note: str = ""
        self.revised_note: str = ""
        self.validation_result: Optional[ValidationResult] = None

        logger.info("[AppState] Running auto-ingest check on startup...")
        ingest_result = self.orchestrator.ensure_data_ingested()
        if ingest_result["notes_ingested"]:
            logger.info("[AppState] Discharge notes ingested on startup.")
        if ingest_result["guidelines_ingested"]:
            logger.info("[AppState] Guidelines ingested on startup.")

        meta_rag = MetaIntelligentClinicalRAG(self.orchestrator.vector_store)
        self.guarded_rag = GuardedClinicalRAG(meta_rag)
        logger.info("[AppState] Startup complete. Ready for clinician use.")

    # ── Tab 1: Patient Analysis ────────────────────────────────────────────

    def analyze_patient(self, hadm_id_str: str, care_setting: str):
        if not hadm_id_str:
            return "Please select a patient from the dropdown.", "", ""
        if not care_setting:
            return "Please select a discharge destination before running the review.", "", ""

        hadm_id_str = hadm_id_str.strip()
        try:
            hadm_id = int(hadm_id_str)
        except ValueError:
            return "Unrecognized patient ID. Please select a patient from the list.", "", ""

        try:
            notes_df = self.orchestrator.mimic.load_discharge_notes()
            row = notes_df[notes_df["hadm_id"] == hadm_id]
            if row.empty:
                return "No discharge note found for this patient.", "", ""
            discharge_note = str(row.iloc[0]["text"])
        except Exception as e:
            logger.exception("Note load failed")
            return "Could not load the discharge note. Please try again.", "", ""

        self.current_note = discharge_note

        try:
            result = self.orchestrator.analyze(
                discharge_note=discharge_note,
                hadm_id=str(hadm_id),
                care_setting=care_setting,
            )
            self.current_result = result
            self.revised_note = ""
            self.validation_result = None
            report = self._format_results(result)
            pkg = result.review_package
            status = (
                f"Review complete — {pkg.total_gaps} items flagged "
                f"({pkg.critical_count} critical, {pkg.high_count} high). "
                "Go to Clinician Review to act on these findings."
            )
            return status, report, discharge_note
        except Exception as e:
            logger.exception("Analysis failed")
            return "Review could not be completed. Please try again.", "", discharge_note

    def _format_results(self, result: AnalysisResult) -> str:
        pkg = result.review_package
        ehr = result.ehr_result
        lines = [
            "DISCHARGE DOCUMENTATION REVIEW",
            "=" * 50,
            "",
            "WHAT WE IDENTIFIED",
            "-" * 30,
            f"  Diagnoses:             {', '.join(result.task_plan.identified_diagnoses) or 'None identified'}",
            f"  Medications reviewed:  {', '.join(result.task_plan.medications_to_verify) or 'None'}",
            f"  Labs to follow up:     {', '.join(result.task_plan.pending_labs_to_check) or 'None'}",
        ]
        if result.task_plan.critical_gaps_to_investigate:
            lines.append("  Areas of concern:")
            for g in result.task_plan.critical_gaps_to_investigate:
                lines.append(f"    - {g}")

        lines += [
            "",
            "MEDICATION & LAB FINDINGS",
            "-" * 30,
            f"  Missing or undocumented medications ({len(ehr.medication_gaps)}):",
        ]
        for mg in ehr.medication_gaps:
            lines.append(f"    [{mg.severity.upper()}] {mg.drug_name} — {mg.reason_flagged}")
        if not ehr.medication_gaps:
            lines.append("    None found.")

        lines.append(f"  Lab documentation gaps ({len(ehr.lab_gaps)}):")
        for lg in ehr.lab_gaps:
            lines.append(f"    [{lg.severity.upper()}] {lg.lab_name} — {lg.reason_flagged}")
        if not ehr.lab_gaps:
            lines.append("    None found.")

        if ehr.allergy_alerts:
            lines += ["", f"ALLERGY SAFETY ALERTS ({len(ehr.allergy_alerts)})", "-" * 30]
            for a in ehr.allergy_alerts:
                lines.append(
                    f"  [{a.severity.upper()}] {a.drug_name} conflicts with documented allergy "
                    f'"{a.allergy}" ({a.conflict_type}) — {a.explanation}'
                )
        else:
            lines += ["", "ALLERGY CHECK", "-" * 30, "  No allergy conflicts detected."]

        if ehr.dose_discrepancies:
            lines += ["", f"DOSE DISCREPANCIES ({len(ehr.dose_discrepancies)})", "-" * 30]
            for d in ehr.dose_discrepancies:
                lines.append(
                    f"  [{d.severity.upper()}] {d.drug_name}: "
                    f"recorded {d.ehr_dose}, noted {d.note_dose} — {d.discrepancy}. "
                    f"Clinical risk: {d.clinical_risk}"
                )
        else:
            lines += ["", "DOSE REVIEW", "-" * 30, "  No dose discrepancies found."]

        compliant_label = "MEETS STANDARDS" if result.guidelines_result.overall_compliant else "GAPS IDENTIFIED"
        lines += [
            "",
            f"CARE STANDARDS REVIEW — {compliant_label}",
            "-" * 30,
        ]
        if result.guidelines_result.violations:
            for v in result.guidelines_result.violations:
                lines.append(f"  [{v.severity.upper()}] {v.guideline_source}: {v.missing_element}")
        else:
            lines.append("  No guideline violations found.")
        if result.guidelines_result.compliant_areas:
            lines.append(f"  Standards met: {', '.join(result.guidelines_result.compliant_areas)}")

        lines += [
            "",
            "REVIEW SUMMARY",
            "=" * 50,
            f"  Critical  : {pkg.critical_count}",
            f"  High      : {pkg.high_count}",
            f"  Medium    : {pkg.medium_count}",
            f"  Low       : {pkg.low_count}",
            "  ─────────────────",
            f"  Total     : {pkg.total_gaps}",
            "",
            "Next step: Go to the Clinician Review tab to act on these findings.",
        ]
        return "\n".join(lines)

    # ── Tab 2: Clinician Review ────────────────────────────────────────────

    def get_review_items(self) -> str:
        if not self.current_result:
            return "No review available yet.\nRun a discharge review in the Patient Analysis tab first."
        pkg = self.current_result.review_package
        if not pkg.review_items:
            return "No items to review — this patient's discharge note appears complete."

        severity_icon = {"critical": "[CRITICAL]", "high": "[HIGH]", "medium": "[MEDIUM]", "low": "[LOW]"}
        lines = [
            "FLAGGED ITEMS FOR REVIEW",
            "=" * 50,
            "",
            "For each item: confirm it needs addressing, dismiss if it doesn't apply,",
            "or escalate if it needs urgent attention.",
            "You must act on all items before a revised note can be generated.",
            "",
        ]
        for i, item in enumerate(pkg.review_items, 1):
            tag = severity_icon.get(item.severity, "[?]")
            lines += [
                f"Item {i}  {tag}  |  {item.gap_type}",
                f"  ID:           {item.item_id}",
                f"  Finding:      {item.description}",
                f"  Recommended:  {item.recommended_action}",
                f"  Care setting: {item.care_setting_note}",
                f"  Status:       {item.status}",
                "",
            ]
        return "\n".join(lines)

    def submit_feedback(self, item_id: str, action: str, notes: str) -> str:
        if not self.current_result:
            return "No review loaded. Run a discharge review first."
        if not item_id.strip():
            return "Please enter the Item ID from the flagged items list above."

        result = self.orchestrator.hitl.record_feedback(
            analysis_id=self.current_result.analysis_id,
            item_id=item_id.strip(),
            action=action,
            notes=notes,
            discharge_note=self.current_note,
        )
        status = result.get("status", "unknown")
        msg = result.get("message", "")
        reanalysis = result.get("reanalysis", "")

        output = f"Action recorded: {action.title()}\n{msg}"
        if reanalysis:
            output += f"\n\nAdditional notes:\n{reanalysis}"
        return output

    def confirm_all_gaps(self) -> str:
        if not self.current_result:
            return "No review loaded. Run a discharge review first."
        pkg = self.current_result.review_package
        count = 0
        for item in pkg.review_items:
            if item.status == "pending_review":
                self.orchestrator.hitl.record_feedback(
                    analysis_id=self.current_result.analysis_id,
                    item_id=item.item_id,
                    action="confirm",
                    notes="Batch confirmed by clinician.",
                    discharge_note=self.current_note,
                )
                count += 1
        return f"{count} items confirmed. You can now generate the revised note in the Revised Note tab."

    def finalize_and_rewrite(self) -> tuple:
        if not self.current_result:
            return "No review available. Run a discharge review first.", ""

        pkg = self.current_result.review_package
        actionable = [
            item for item in pkg.review_items
            if item.status in ("confirmed", "re_flagged_after_dismissal", "escalated")
        ]
        if not actionable:
            pending = sum(1 for item in pkg.review_items if item.status == "pending_review")
            if pending > 0:
                return (
                    f"{pending} items are still awaiting your review. "
                    "Please confirm, dismiss, or escalate each item — or use Confirm All to proceed.",
                    ""
                )
            return "No items to address. The discharge note stands as written.", ""

        try:
            revised, validation = self.orchestrator.generate_revised_note(
                analysis_id=self.current_result.analysis_id,
                discharge_note=self.current_note,
                hadm_id=self.current_result.hadm_id,
            )
            self.revised_note = revised
            self.validation_result = validation

            status = f"Revised note ready — {len(actionable)} items addressed. Go to the Revised Note tab."
            if validation and validation.ungrounded_count > 0:
                status += (
                    f" Note: {validation.ungrounded_count} of {validation.total_additions} "
                    "additions need your attention — please review the validation check."
                )
            return status, ""
        except Exception as e:
            logger.exception("Note rewrite failed")
            return "Could not generate the revised note. Please try again.", ""

    # ── Tab 3: Revised Note ────────────────────────────────────────────────

    def load_revised(self) -> str:
        if self.revised_note:
            return self.revised_note
        return (
            "Your revised note will appear here.\n\n"
            "To generate it, complete the review in the Clinician Review tab "
            "and click 'Finalize Review'."
        )

    def load_validation_report(self) -> str:
        if not self.validation_result:
            return "Validation results will appear here after the note is generated."

        v = self.validation_result
        if v.ungrounded_count == 0:
            lines = [
                "NOTE VALIDATION — ALL CLEAR",
                "=" * 50,
                "",
                "No issues found. This note is ready for handoff.",
                "",
                f"  Additions reviewed:  {v.total_additions}",
                f"  All verified:        {v.grounded_count}",
                "",
                v.summary,
            ]
        else:
            lines = [
                f"NOTE VALIDATION — {v.ungrounded_count} ITEM(S) NEED ATTENTION",
                "=" * 50,
                "",
                f"  Additions reviewed:  {v.total_additions}",
                f"  Verified:            {v.grounded_count}",
                f"  Needs attention:     {v.ungrounded_count}",
                "",
                f"Summary: {v.summary}",
                "",
            ]
            for block in v.validated_blocks:
                tag = "VERIFIED" if block.grounded else "REVIEW NEEDED"
                lines += [
                    f"--- [{tag}] ---",
                    f"  Content: {block.text[:200]}{'...' if len(block.text) > 200 else ''}",
                    f"  Source:  {block.source}",
                ]
                if block.concern:
                    lines.append(f"  Concern: {block.concern}")
                lines.append("")

        return "\n".join(lines)

    def save_revised_note_ui(self) -> str:
        """UI-facing save wrapper — returns an HTML banner, never exposes file paths."""
        if not self.current_result:
            return "<p style='color:#991B1B;padding:12px;'>No review loaded. Run a discharge review first.</p>"
        if not self.revised_note:
            return "<p style='color:#991B1B;padding:12px;'>No revised note to save. Generate one first.</p>"

        if self.validation_result and self.validation_result.ungrounded_count > 0:
            logger.warning(
                "[Save] Saving note with %d ungrounded additions — clinician accepted the risk.",
                self.validation_result.ungrounded_count,
            )

        pkg = self.current_result.review_package
        gaps_addressed = [
            f"[{item.severity.upper()} | {item.gap_type}] {item.description}"
            for item in pkg.review_items
            if item.status in ("confirmed", "re_flagged_after_dismissal", "escalated")
        ]

        try:
            self.orchestrator.save_revised_note(
                hadm_id=str(self.current_result.hadm_id or "unknown"),
                analysis_id=self.current_result.analysis_id,
                original_note=self.current_note,
                revised_note=self.revised_note,
                gaps_addressed=gaps_addressed,
            )
            return (
                "<div style='background:#F0FDF4;border:1px solid #86EFAC;border-radius:8px;"
                "padding:16px 20px;margin-top:8px;'>"
                "<div style='font-weight:700;font-size:1rem;color:#166534;'>✓ Note Saved Successfully</div>"
                "<div style='font-size:0.875rem;color:#166534;margin-top:6px;'>"
                "Your revised discharge note has been securely saved and is ready for handoff."
                "</div></div>"
            )
        except Exception as e:
            logger.exception("Save failed")
            return "<p style='color:#991B1B;padding:12px;'>Could not save the note. Please try again.</p>"

    # ── Admin helpers (internal — not exposed in UI) ───────────────────────

    def refresh_data_status(self) -> str:
        status = self.orchestrator.data_status()
        lines = ["## Data Status\n"]
        lines.append("### MIMIC-IV Files")
        for fname, present in status["mimic_files"].items():
            icon = "+" if present else "-"
            lines.append(f"  {icon} {fname}")
        lines.append("\n### Guidelines")
        icon = "+" if status["guidelines_files"] else "-"
        lines.append(f"  {icon} PDF/TXT files in data/raw/guidelines/")
        lines.append("\n### Vector Store (ChromaDB)")
        vs = status["vector_store"]
        lines.append(f"  - Discharge note chunks: {vs['notes_chunks']}")
        lines.append(f"  - Guideline chunks: {vs['guidelines_chunks']}")
        lines.append(f"  - Persist dir: {vs['persist_dir']}")
        api_status = "Available" if status["rxnorm_api"] else "Offline"
        lines.append(f"\n### RxNorm API: {api_status}")
        return "\n".join(lines)

    def ingest_notes(self, limit_str: str) -> str:
        try:
            limit = int(limit_str) if limit_str.strip() else None
            count = self.orchestrator.ingest_discharge_notes(limit=limit)
            return f"Ingested {count} discharge note chunks."
        except Exception as e:
            return f"Ingestion failed: {e}"

    def ingest_guidelines_action(self) -> str:
        try:
            count = self.orchestrator.ingest_guidelines()
            return f"Ingested {count} guideline chunks."
        except Exception as e:
            return f"Ingestion failed: {e}"


# ── Build interface ────────────────────────────────────────────────────────────

def create_interface():
    state = AppState()

    with gr.Blocks(title="CareHandoff AI", css=_CSS) as demo:

        gr.Markdown(
            "# CareHandoff AI\n"
            "*Discharge documentation intelligence for safer patient transitions.*"
        )

        # ── TAB 1: Patient Analysis ───────────────────────────────────────
        with gr.Tab("Patient Analysis"):
            gr.Markdown(
                "### Review a Patient's Discharge Note\n"
                "Select a patient and their discharge destination. "
                "We'll scan their discharge note for documentation gaps, "
                "missing medications, and safety concerns."
            )

            valid_ids = state.orchestrator.mimic.list_available_admissions(n=243)

            with gr.Row():
                hadm_id_input = gr.Dropdown(
                    choices=[str(i) for i in valid_ids],
                    label="Select Patient",
                    info=f"{len(valid_ids)} patients available",
                    filterable=True,
                    value=None,
                    scale=2,
                )
                care_setting = gr.Radio(
                    choices=[
                        ("Home", "home"),
                        ("Skilled Nursing Facility", "skilled_nursing_facility"),
                        ("Urgent Care Clinic", "urgent_clinic"),
                    ],
                    label="Where is this patient being discharged to?",
                    info="This helps us apply the right follow-up care requirements for your patient.",
                    value=None,
                    scale=2,
                )
                analyze_btn = gr.Button("Run Discharge Review", scale=1, variant="primary")

            status_bar = gr.Textbox(
                label="Status",
                interactive=False,
                max_lines=2,
                placeholder="Status will appear here once you run the review.",
            )

            analysis_output = gr.Textbox(
                label="What We Found",
                info="A summary of documentation gaps, medication concerns, and safety flags in this patient's discharge note.",
                lines=30,
                interactive=False,
                placeholder="Your results will appear here once you run the discharge review above.",
                elem_classes=["clinical-output"],
            )

            with gr.Accordion("Original Discharge Note (read-only)", open=False):
                note_display = gr.Textbox(
                    label="Discharge Note",
                    lines=16,
                    interactive=False,
                    elem_classes=["clinical-output"],
                )

            analyze_btn.click(
                fn=state.analyze_patient,
                inputs=[hadm_id_input, care_setting],
                outputs=[status_bar, analysis_output, note_display],
            )

        # ── TAB 2: Clinician Review ───────────────────────────────────────
        with gr.Tab("Clinician Review"):
            gr.Markdown(
                "### Review Flagged Items\n"
                "Review each flagged item. Confirm what needs to be addressed, "
                "dismiss anything that doesn't apply, or escalate urgent concerns."
            )

            with gr.Row():
                load_review_btn = gr.Button("Load Flagged Items", scale=1)
                confirm_all_btn = gr.Button("Confirm All", scale=1, variant="secondary")

            gr.Markdown(
                "<small style='color:#6B7280;'>"
                "Use <strong>Confirm All</strong> to confirm all flagged items at once "
                "if you've already reviewed them."
                "</small>"
            )

            review_output = gr.Textbox(
                label="Flagged Items",
                lines=20,
                interactive=False,
                placeholder="Flagged items from your discharge review will appear here.",
                elem_classes=["clinical-output"],
            )
            load_review_btn.click(fn=state.get_review_items, outputs=review_output)

            confirm_all_status = gr.Textbox(
                label="Status",
                interactive=False,
                max_lines=2,
                placeholder="Status will appear here.",
            )
            confirm_all_btn.click(fn=state.confirm_all_gaps, outputs=confirm_all_status)

            gr.Markdown("---")
            gr.Markdown("### Act on an Individual Item")

            with gr.Row():
                item_id_input = gr.Textbox(
                    label="Item ID",
                    placeholder="Copy the Item ID from the list above",
                    scale=2,
                )
                action_input = gr.Radio(
                    ["confirm", "dismiss", "escalate"],
                    label="Action",
                    value="confirm",
                    scale=1,
                )

            feedback_notes = gr.Textbox(
                label="Clinical Notes",
                placeholder="Add context — required for dismissals (e.g. Patient is already monitored at receiving facility...)",
                lines=3,
            )
            submit_btn = gr.Button("Submit", variant="secondary")
            feedback_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=4,
                placeholder="Action status will appear here.",
            )
            submit_btn.click(
                fn=state.submit_feedback,
                inputs=[item_id_input, action_input, feedback_notes],
                outputs=feedback_output,
            )

            gr.Markdown("---")
            gr.Markdown(
                "### Finalize Review\n"
                "Once you have reviewed all items, click below to generate "
                "a revised discharge note with your confirmed changes applied."
            )
            finalize_btn = gr.Button("Finalize Review", variant="primary")
            rewrite_status = gr.Textbox(
                label="Status",
                interactive=False,
                max_lines=4,
                placeholder="Generation status will appear here.",
            )
            rewrite_detail = gr.Textbox(visible=False)

            finalize_btn.click(
                fn=state.finalize_and_rewrite,
                outputs=[rewrite_status, rewrite_detail],
            )

        # ── TAB 3: Revised Note ───────────────────────────────────────────
        with gr.Tab("Revised Note"):
            gr.Markdown(
                "### Your Revised Discharge Note\n"
                "Generate a clean, updated discharge note reflecting all confirmed changes. "
                "Review the validation check before saving."
            )

            load_revised_btn = gr.Button("Generate Revised Note", variant="primary")
            revised_note_display = gr.Textbox(
                label="Your Revised Discharge Note",
                lines=28,
                interactive=False,
                placeholder="Your revised note will appear here after generation.",
                elem_classes=["clinical-output"],
            )
            load_revised_btn.click(fn=state.load_revised, outputs=revised_note_display)

            gr.Markdown("---")
            gr.Markdown(
                "### Note Validation\n"
                "A final check on your revised note before saving. "
                "Any remaining concerns are flagged here for your attention."
            )
            load_validation_btn = gr.Button("Run Validation Check", variant="secondary")
            validation_display = gr.Textbox(
                label="Validation Flags",
                lines=14,
                interactive=False,
                placeholder="Validation results will appear here after the note is generated.",
                elem_classes=["clinical-output"],
            )
            load_validation_btn.click(fn=state.load_validation_report, outputs=validation_display)

            gr.Markdown("---")
            gr.Markdown(
                "<p style='color:#374151;font-size:0.9rem;margin-bottom:8px;'>"
                "Once you're satisfied with the note and validation results, "
                "save it to complete the handoff record."
                "</p>"
            )
            save_btn = gr.Button("Save Note", variant="primary")
            save_status = gr.HTML(elem_classes=["save-banner"])
            save_btn.click(fn=state.save_revised_note_ui, outputs=save_status)

        # ── TAB 4: Clinical Q&A ───────────────────────────────────────────
        with gr.Tab("Clinical Q&A"):
            gr.Markdown(
                "### Clinical Documentation Q&A\n"
                "Ask questions about discharge documentation standards and care protocols. "
                "We'll find answers from trusted clinical guidelines.\n\n"
                "> *Answers are based on clinical documentation guidelines only. "
                "This tool does not provide personal medical advice — "
                "always apply your clinical judgment.*"
            )

            gr.ChatInterface(
                fn=state.guarded_rag.chat,
                chatbot=gr.Chatbot(
                    label="Answer",
                    height=500,
                ),
                textbox=gr.Textbox(
                    placeholder="e.g. What are the documentation requirements for heart failure discharge?",
                    container=False,
                    scale=7,
                ),
                examples=[
                    "What is the recommended documentation for heart failure discharge?",
                    "What are the guidelines for hypertension management in discharge notes?",
                    "Is it required to document medication reconciliation at discharge?",
                    "Give me a comprehensive overview of all discharge documentation requirements.",
                    "What lab values should be documented for patients with AKI?",
                    "How should COPD exacerbation be documented at discharge?",
                ],
                cache_examples=False,
            )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
