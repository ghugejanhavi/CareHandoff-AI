import { useState, useEffect } from 'react'
import {
  Search, ChevronDown, ChevronUp, AlertTriangle,
  Activity, Pill, FlaskConical, Stethoscope,
  CheckCircle2, AlertCircle, Loader2,
} from 'lucide-react'
import { api } from '../lib/api'
import type { AnalysisResult } from '../lib/types'
import { severityBadgeClass, severityDotClass, capitalize } from '../lib/utils'

interface Props {
  onAnalysisComplete: (result: AnalysisResult) => void
}

const CARE_SETTINGS = [
  { value: 'home',                    label: 'Home' },
  { value: 'skilled_nursing_facility', label: 'Skilled Nursing Facility' },
  { value: 'urgent_clinic',           label: 'Urgent Care Clinic' },
]

// ── Shared badge ──────────────────────────────────────────────────────────────
function SeverityBadge({ severity }: { severity: string }) {
  return (
    <span
      className={`inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full border whitespace-nowrap ${severityBadgeClass(severity)}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${severityDotClass(severity)}`} />
      {capitalize(severity)}
    </span>
  )
}

// ── Collapsible findings section ──────────────────────────────────────────────
function Section({
  title, icon: Icon, count, children,
}: {
  title: string
  icon: React.ElementType
  count: number
  children: React.ReactNode
}) {
  const [open, setOpen] = useState(true)
  if (count === 0) return null
  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-5 py-4 hover:bg-gray-50 transition-colors text-left"
      >
        <div className="flex items-center gap-3">
          <Icon className="w-4 h-4 text-gray-400" />
          <span className="text-sm font-medium text-gray-900">{title}</span>
          <span className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded-full font-medium">
            {count}
          </span>
        </div>
        {open
          ? <ChevronUp className="w-4 h-4 text-gray-400" />
          : <ChevronDown className="w-4 h-4 text-gray-400" />}
      </button>
      {open && (
        <div className="border-t border-gray-100 px-5 py-4 space-y-0">
          {children}
        </div>
      )}
    </div>
  )
}

function Row({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 py-3 border-b border-gray-50 last:border-0">
      {children}
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function AnalysisPage({ onAnalysisComplete }: Props) {
  const [patients, setPatients]         = useState<string[]>([])
  const [hadmId, setHadmId]             = useState('')
  const [careSetting, setCareSetting]   = useState('')
  const [loading, setLoading]           = useState(false)
  const [loadingPts, setLoadingPts]     = useState(true)
  const [error, setError]               = useState('')
  const [result, setResult]             = useState<AnalysisResult | null>(null)

  useEffect(() => {
    api.listPatients()
      .then(r => { setPatients(r.patients) })
      .catch(() => setError('Failed to load patient list.'))
      .finally(() => setLoadingPts(false))
  }, [])

  async function handleAnalyze() {
    if (!hadmId || !careSetting) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const r = await api.analyze(hadmId, careSetting as 'home' | 'skilled_nursing_facility' | 'urgent_clinic')
      setResult(r)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Analysis failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-xl font-semibold text-gray-900">Patient Discharge Analysis</h1>
        <p className="text-sm text-gray-500 mt-1">
          Select a patient and their discharge destination. We'll scan their discharge note for documentation gaps, missing medications, and safety concerns.
        </p>
      </div>

      {/* Selection card */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1.5">
              Patient
            </label>
            {loadingPts ? (
              <div className="h-9 bg-gray-100 rounded-lg animate-pulse" />
            ) : (
              <select
                value={hadmId}
                onChange={e => setHadmId(e.target.value)}
                className="w-full h-9 px-3 text-sm bg-white border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="" disabled>Search by patient ID or scroll to select</option>
                {patients.map(p => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            )}
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1.5">
              Discharge Destination
            </label>
            <select
              value={careSetting}
              onChange={e => setCareSetting(e.target.value)}
              className="w-full h-9 px-3 text-sm bg-white border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="" disabled>Select destination</option>
              {CARE_SETTINGS.map(s => (
                <option key={s.value} value={s.value}>{s.label}</option>
              ))}
            </select>
          </div>

          <button
            onClick={handleAnalyze}
            disabled={loading || !hadmId || !careSetting}
            className="h-9 px-5 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 active:bg-blue-800 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-colors"
          >
            {loading
              ? <><Loader2 className="w-4 h-4 animate-spin" /> Analysing…</>
              : <><Search className="w-4 h-4" /> Run Analysis</>}
          </button>
        </div>

        {error && (
          <div className="mt-4 flex items-center gap-2 text-sm text-red-700 bg-red-50 rounded-lg px-4 py-3 border border-red-100">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            {error}
          </div>
        )}
      </div>

      {/* ── Results ───────────────────────────────────────────────────────────── */}
      {result && (
        <>
          {/* KPI cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { label: 'Critical', value: result.summary.critical, text: 'text-red-600',    bg: 'bg-red-50',    border: 'border-red-100' },
              { label: 'High',     value: result.summary.high,     text: 'text-orange-600', bg: 'bg-orange-50', border: 'border-orange-100' },
              { label: 'Medium',   value: result.summary.medium,   text: 'text-amber-600',  bg: 'bg-amber-50',  border: 'border-amber-100' },
              { label: 'Low',      value: result.summary.low,      text: 'text-blue-600',   bg: 'bg-blue-50',   border: 'border-blue-100' },
            ].map(k => (
              <div key={k.label} className={`rounded-xl border ${k.border} ${k.bg} p-5`}>
                <div className={`text-3xl font-bold tabular-nums ${k.text}`}>{k.value}</div>
                <div className="text-xs font-medium text-gray-500 mt-1">{k.label} Priority</div>
              </div>
            ))}
          </div>

          {/* Diagnoses + Medications */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-white rounded-xl border border-gray-200 p-5">
              <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
                Identified Diagnoses
              </h3>
              <ul className="space-y-1.5">
                {result.diagnoses.map((d, i) => (
                  <li key={i} className="text-sm text-gray-700 flex items-start gap-2">
                    <span className="w-1 h-1 rounded-full bg-gray-400 mt-2 flex-shrink-0" />
                    {d}
                  </li>
                ))}
              </ul>
            </div>

            <div className="bg-white rounded-xl border border-gray-200 p-5">
              <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
                Medications Reviewed
              </h3>
              <div className="flex flex-wrap gap-1.5">
                {result.medications_reviewed.map((m, i) => (
                  <span key={i} className="text-xs text-gray-600 bg-gray-100 px-2.5 py-1 rounded-md font-medium">
                    {m}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Findings */}
          <div className="space-y-3">
            <Section title="Medication Gaps" icon={Pill} count={result.medication_gaps.length}>
              {result.medication_gaps.map((g, i) => (
                <Row key={i}>
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">{g.drug_name}</p>
                    <p className="text-xs text-gray-500 mt-0.5">{g.reason_flagged}</p>
                  </div>
                  <SeverityBadge severity={g.severity} />
                </Row>
              ))}
            </Section>

            <Section title="Lab Gaps" icon={FlaskConical} count={result.lab_gaps.length}>
              {result.lab_gaps.map((g, i) => (
                <Row key={i}>
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-gray-900">{g.lab_name}</p>
                    <p className="text-xs text-gray-500 mt-0.5">{g.reason_flagged}</p>
                  </div>
                  <SeverityBadge severity={g.severity} />
                </Row>
              ))}
            </Section>

            <Section title="Allergy Alerts" icon={AlertTriangle} count={result.allergy_alerts.length}>
              {result.allergy_alerts.map((a, i) => (
                <Row key={i}>
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-gray-900">{a.drug_name}</p>
                    <p className="text-xs text-gray-500 mt-0.5">
                      {a.conflict_type} — {a.explanation}
                    </p>
                  </div>
                  <SeverityBadge severity={a.severity} />
                </Row>
              ))}
            </Section>

            <Section title="Dose Discrepancies" icon={Activity} count={result.dose_discrepancies.length}>
              {result.dose_discrepancies.map((d, i) => (
                <Row key={i}>
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-gray-900">{d.drug_name}</p>
                    <p className="text-xs text-gray-500 mt-0.5">
                      EHR: <span className="font-mono">{d.ehr_dose}</span>
                      {' → '}
                      Note: <span className="font-mono">{d.note_dose}</span>
                      {d.discrepancy ? ` · ${d.discrepancy}` : ''}
                    </p>
                    {d.clinical_risk && (
                      <p className="text-xs text-red-600 mt-0.5">{d.clinical_risk}</p>
                    )}
                  </div>
                  <SeverityBadge severity={d.severity} />
                </Row>
              ))}
            </Section>

            <Section title="Guideline Violations" icon={Stethoscope} count={result.guideline_violations.length}>
              {result.guideline_violations.map((v, i) => (
                <Row key={i}>
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-gray-900">{v.missing_element}</p>
                    <p className="text-xs text-gray-500 mt-0.5">{v.guideline_source}</p>
                  </div>
                  <SeverityBadge severity={v.severity} />
                </Row>
              ))}
            </Section>

            {result.compliant_areas.length > 0 && (
              <div className="bg-white rounded-xl border border-gray-200 p-5">
                <div className="flex items-center gap-2 mb-3">
                  <CheckCircle2 className="w-4 h-4 text-green-600" />
                  <h3 className="text-sm font-medium text-gray-700">Compliant Areas</h3>
                </div>
                <ul className="space-y-1.5">
                  {result.compliant_areas.map((c, i) => (
                    <li key={i} className="text-sm text-gray-600 flex items-start gap-2">
                      <span className="w-1 h-1 rounded-full bg-green-400 mt-2 flex-shrink-0" />
                      {c}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* CTA */}
          <div className="flex justify-end pt-2">
            <button
              onClick={() => onAnalysisComplete(result)}
              className="px-6 py-2.5 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors"
            >
              Proceed to Clinician Review →
            </button>
          </div>
        </>
      )}
    </div>
  )
}
