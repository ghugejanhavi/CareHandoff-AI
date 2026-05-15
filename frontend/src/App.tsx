import { useState } from 'react'
import { Activity, ClipboardCheck, FileText, MessageSquare, Heart } from 'lucide-react'
import AnalysisPage from './pages/AnalysisPage'
import ReviewPage from './pages/ReviewPage'
import NotePage from './pages/NotePage'
import QAPage from './pages/QAPage'
import type { AnalysisResult } from './lib/types'

const TABS = [
  { id: 'analysis', label: 'Patient Analysis', icon: Activity },
  { id: 'review',   label: 'Clinician Review', icon: ClipboardCheck },
  { id: 'note',     label: 'Revised Note',     icon: FileText },
  { id: 'qa',       label: 'Clinical Q&A',     icon: MessageSquare },
] as const

type TabId = (typeof TABS)[number]['id']

export default function App() {
  const [tab, setTab] = useState<TabId>('analysis')
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)

  function handleAnalysisComplete(r: AnalysisResult) {
    setAnalysisResult(r)
    setTab('review')
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center shadow-sm">
              <Heart className="w-4 h-4 text-white" strokeWidth={2.5} />
            </div>
            <div className="leading-none">
              <span className="text-gray-900 font-semibold text-sm tracking-tight">
                CareHandoff AI
              </span>
              <p className="text-xs text-gray-400 font-normal mt-0.5">
                Discharge documentation intelligence for safer patient transitions.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {analysisResult && (
              <span className="text-xs text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
                Patient{' '}
                <span className="font-medium text-gray-700">{analysisResult.hadm_id}</span>
              </span>
            )}
            <span className="inline-flex items-center gap-1.5 text-xs text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
              Ready
            </span>
          </div>
        </div>

        {/* Tab bar */}
        <div className="max-w-7xl mx-auto px-6 border-t border-gray-100">
          <nav className="flex" role="tablist">
            {TABS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                role="tab"
                aria-selected={tab === id}
                onClick={() => setTab(id)}
                className={[
                  'flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors focus:outline-none',
                  tab === id
                    ? 'border-blue-600 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300',
                ].join(' ')}
              >
                <Icon className="w-3.5 h-3.5" />
                {label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Page content — all tabs stay mounted to preserve state across switches */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-8">
        <div className={tab !== 'analysis' ? 'hidden' : ''}>
          <AnalysisPage onAnalysisComplete={handleAnalysisComplete} />
        </div>
        <div className={tab !== 'review' ? 'hidden' : ''}>
          <ReviewPage
            analysisResult={analysisResult}
            onProceedToNote={() => setTab('note')}
          />
        </div>
        <div className={tab !== 'note' ? 'hidden' : ''}>
          <NotePage analysisResult={analysisResult} />
        </div>
        <div className={tab !== 'qa' ? 'hidden' : ''}>
          <QAPage />
        </div>
      </main>
    </div>
  )
}
