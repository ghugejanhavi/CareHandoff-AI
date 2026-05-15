import { useState, useEffect, useCallback } from 'react'
import {
  CheckCircle2, XCircle, ArrowUpCircle,
  Loader2, AlertCircle, ClipboardList,
} from 'lucide-react'
import { api } from '../lib/api'
import type { AnalysisResult, ReviewItem } from '../lib/types'
import {
  severityBadgeClass, severityDotClass,
  statusBadgeClass, statusLabel, capitalize,
} from '../lib/utils'

interface Props {
  analysisResult: AnalysisResult | null
  onProceedToNote: () => void
}

// ── Single review card ────────────────────────────────────────────────────────
function ReviewCard({
  item,
  busy,
  onAction,
}: {
  item: ReviewItem
  busy: boolean
  onAction: (id: string, action: 'confirm' | 'dismiss' | 'escalate') => void
}) {
  const isPending = item.status === 'pending_review'

  const ACTIONS = [
    {
      key: 'confirm'  as const,
      Icon: CheckCircle2,
      cls: 'text-green-700 border-green-200 hover:bg-green-50',
    },
    {
      key: 'dismiss'  as const,
      Icon: XCircle,
      cls: 'text-gray-500 border-gray-200 hover:bg-gray-50',
    },
    {
      key: 'escalate' as const,
      Icon: ArrowUpCircle,
      cls: 'text-purple-700 border-purple-200 hover:bg-purple-50',
    },
  ]

  return (
    <div
      className={`bg-white rounded-xl border transition-all ${
        isPending ? 'border-gray-200' : 'border-gray-100 opacity-80'
      } p-5`}
    >
      <div className="flex items-start justify-between gap-4">
        {/* Left: content */}
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2 mb-2">
            <span
              className={`inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full border ${severityBadgeClass(item.severity)}`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${severityDotClass(item.severity)}`} />
              {capitalize(item.severity)}
            </span>
            <span className="text-xs text-gray-400 font-medium">{item.gap_type}</span>
            <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${statusBadgeClass(item.status)}`}>
              {statusLabel(item.status)}
            </span>
          </div>

          <p className="text-sm font-medium text-gray-900 leading-snug">
            {item.description}
          </p>

          {item.recommended_action && (
            <p className="text-xs text-gray-500 mt-1.5 flex items-start gap-1">
              <span className="text-gray-400 font-bold">→</span>
              {item.recommended_action}
            </p>
          )}

          {item.care_setting_note && (
            <p className="text-xs text-blue-600 mt-1 italic">{item.care_setting_note}</p>
          )}
        </div>

        {/* Right: action buttons (pending only) */}
        {isPending && (
          <div className="flex items-center gap-1.5 flex-shrink-0">
            {ACTIONS.map(({ key, Icon, cls }) => (
              <button
                key={key}
                onClick={() => onAction(item.item_id, key)}
                disabled={busy}
                className={`flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium rounded-lg border bg-white ${cls} disabled:opacity-40 disabled:cursor-not-allowed transition-colors`}
              >
                {busy
                  ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  : <Icon className="w-3.5 h-3.5" />}
                {capitalize(key)}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function ReviewPage({ analysisResult, onProceedToNote }: Props) {
  const [items, setItems]               = useState<ReviewItem[]>([])
  const [loading, setLoading]           = useState(true)
  const [error, setError]               = useState('')
  const [actingOn, setActingOn]         = useState<string | null>(null)
  const [confirmingAll, setConfirmingAll] = useState(false)

  const loadReview = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const r = await api.getReview()
      setItems(r.items)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load review.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (analysisResult) loadReview()
    else setLoading(false)
  }, [analysisResult, loadReview])

  async function handleAction(itemId: string, action: 'confirm' | 'dismiss' | 'escalate') {
    setActingOn(itemId)
    try {
      await api.submitFeedback(itemId, action)
      await loadReview()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Action failed.')
    } finally {
      setActingOn(null)
    }
  }

  async function handleConfirmAll() {
    setConfirmingAll(true)
    try {
      await api.confirmAll()
      await loadReview()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Confirm all failed.')
    } finally {
      setConfirmingAll(false)
    }
  }

  // ── Empty state ─────────────────────────────────────────────────────────────
  if (!analysisResult) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-gray-400 gap-3">
        <ClipboardList className="w-10 h-10" />
        <p className="text-sm">Run a patient analysis first to see review items.</p>
      </div>
    )
  }

  const pending  = items.filter(i => i.status === 'pending_review').length
  const acted    = items.length - pending
  const progress = items.length > 0 ? Math.round((acted / items.length) * 100) : 0
  const allDone  = items.length > 0 && pending === 0

  return (
    <div className="space-y-6">
      {/* Header row */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-gray-900">Clinician Review</h1>
          <p className="text-sm text-gray-500 mt-1">
            Review each flagged item. Confirm what needs to be addressed, dismiss anything that doesn't apply, or escalate urgent concerns.
          </p>
        </div>

        <div className="flex items-center gap-2 flex-shrink-0">
          <button
            onClick={handleConfirmAll}
            disabled={confirmingAll || pending === 0}
            className="flex items-center gap-1.5 px-3.5 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {confirmingAll
              ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
              : <CheckCircle2 className="w-3.5 h-3.5" />}
            Confirm All
          </button>

          <button
            onClick={onProceedToNote}
            disabled={!allDone}
            title={!allDone ? `${pending} item(s) still pending` : undefined}
            className="flex items-center gap-1.5 px-3.5 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Generate Note →
          </button>
        </div>
      </div>

      {/* Progress bar */}
      <div className="bg-white rounded-xl border border-gray-200 p-5">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-gray-600">Review Progress</span>
          <span className="text-xs text-gray-400 tabular-nums">
            {acted} / {items.length} reviewed
          </span>
        </div>
        <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-600 rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
        {allDone && (
          <p className="text-xs text-green-700 mt-2 flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5" />
            All items reviewed — ready to generate the revised note.
          </p>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 text-sm text-red-700 bg-red-50 rounded-lg px-4 py-3 border border-red-100">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          {error}
        </div>
      )}

      {/* Items */}
      {loading ? (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="w-6 h-6 animate-spin text-blue-600" />
        </div>
      ) : (
        <div className="space-y-3">
          {items.map(item => (
            <ReviewCard
              key={item.item_id}
              item={item}
              busy={actingOn === item.item_id}
              onAction={handleAction}
            />
          ))}
        </div>
      )}
    </div>
  )
}
