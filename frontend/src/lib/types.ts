// ── Domain models ─────────────────────────────────────────────────────────────

export interface MedicationGap {
  drug_name: string
  severity: string
  reason_flagged: string
}

export interface LabGap {
  lab_name: string
  severity: string
  reason_flagged: string
}

export interface AllergyAlert {
  drug_name: string
  allergy: string
  conflict_type: string
  severity: string
  explanation: string
}

export interface DoseDiscrepancy {
  drug_name: string
  ehr_dose: string
  note_dose: string
  discrepancy: string
  clinical_risk: string
  severity: string
}

export interface GuidelineViolation {
  guideline_source: string
  missing_element: string
  severity: string
}

export interface AnalysisSummary {
  critical: number
  high: number
  medium: number
  low: number
  total: number
}

export interface AnalysisResult {
  analysis_id: string
  hadm_id: string
  summary: AnalysisSummary
  diagnoses: string[]
  medications_reviewed: string[]
  labs_to_follow_up: string[]
  areas_of_concern: string[]
  medication_gaps: MedicationGap[]
  lab_gaps: LabGap[]
  allergy_alerts: AllergyAlert[]
  dose_discrepancies: DoseDiscrepancy[]
  guideline_violations: GuidelineViolation[]
  compliant_areas: string[]
  overall_compliant: boolean
  discharge_note: string
}

export interface ReviewItem {
  item_id: string
  severity: string
  gap_type: string
  description: string
  recommended_action: string
  care_setting_note: string
  status: string
}

export interface ReviewPackage {
  items: ReviewItem[]
  total: number
  pending: number
  acted: number
}

export interface ValidationBlock {
  index: number
  text: string
  grounded: boolean
  source: string
  concern: string | null
}

export interface ValidationResult {
  total_additions: number
  grounded_count: number
  ungrounded_count: number
  summary: string
  blocks: ValidationBlock[]
}

export type CareSetting = 'home' | 'skilled_nursing_facility' | 'urgent_clinic'
export type FeedbackAction = 'confirm' | 'dismiss' | 'escalate'
