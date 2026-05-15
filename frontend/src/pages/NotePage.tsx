import { useState } from 'react'
import {
  FileText, Loader2, AlertCircle, CheckCircle2,
  Save, Download,
} from 'lucide-react'
import { api } from '../lib/api'
import type { AnalysisResult } from '../lib/types'

interface Props {
  analysisResult: AnalysisResult | null
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function NotePage({ analysisResult }: Props) {
  const [generating, setGenerating]   = useState(false)
  const [saving, setSaving]           = useState(false)
  const [revisedNote, setRevisedNote] = useState('')
  const [error, setError]             = useState('')
  const [saveSuccess, setSaveSuccess] = useState(false)

  async function handleGenerate() {
    setGenerating(true)
    setError('')
    setRevisedNote('')
    setSaveSuccess(false)
    try {
      const r = await api.generateNote()
      setRevisedNote(r.revised_note)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Generation failed.')
    } finally {
      setGenerating(false)
    }
  }

  async function handleSave() {
    setSaving(true)
    setError('')
    try {
      await api.saveNote()
      setSaveSuccess(true)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Save failed.')
    } finally {
      setSaving(false)
    }
  }

  function handleDownload() {
    const blob = new Blob([revisedNote], { type: 'text/plain' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href     = url
    a.download = `revised_note_${analysisResult?.hadm_id ?? 'unknown'}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  // ── Empty state ─────────────────────────────────────────────────────────────
  if (!analysisResult) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-gray-400 gap-3">
        <FileText className="w-10 h-10" />
        <p className="text-sm">Complete the patient analysis and clinician review steps first.</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-gray-900">Revised Discharge Note</h1>
          <p className="text-sm text-gray-500 mt-1">
            Generate an updated note that addresses the items you confirmed in your review.
          </p>
        </div>

        <div className="flex items-center gap-2 flex-shrink-0">
          {revisedNote && (
            <>
              <button
                onClick={handleDownload}
                className="flex items-center gap-1.5 px-3.5 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                Download
              </button>
              <button
                onClick={handleSave}
                disabled={saving || saveSuccess}
                className="flex items-center gap-1.5 px-3.5 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {saving
                  ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  : <Save className="w-3.5 h-3.5" />}
                {saveSuccess ? 'Saved' : 'Save Note'}
              </button>
            </>
          )}

          <button
            onClick={handleGenerate}
            disabled={generating}
            className="flex items-center gap-1.5 px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {generating
              ? <><Loader2 className="w-4 h-4 animate-spin" /> Generating…</>
              : <><FileText className="w-4 h-4" /> {revisedNote ? 'Regenerate' : 'Generate Revised Note'}</>}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 text-sm text-red-700 bg-red-50 rounded-lg px-4 py-3 border border-red-100">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          {error}
        </div>
      )}

      {/* Save success banner */}
      {saveSuccess && (
        <div className="flex items-center gap-2 text-sm text-green-700 bg-green-50 rounded-lg px-4 py-3 border border-green-100">
          <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
          Note saved successfully.
        </div>
      )}

      {/* Note viewer */}
      {revisedNote && (
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden flex flex-col">
          <div className="px-5 py-3.5 border-b border-gray-100 flex items-center gap-2">
            <FileText className="w-4 h-4 text-gray-400" />
            <span className="text-sm font-medium text-gray-700">Revised Discharge Note</span>
          </div>
          <pre className="flex-1 px-5 py-4 text-xs leading-relaxed text-gray-800 whitespace-pre-wrap font-mono-clinical overflow-y-auto max-h-[640px]">
            {revisedNote}
          </pre>
        </div>
      )}
    </div>
  )
}
