import { Shield, BookOpen, Zap } from 'lucide-react'

const FEATURES = [
  {
    icon: BookOpen,
    title: 'Guideline-grounded',
    desc: 'Answers drawn from ACC/AHA, JNC, GOLD, and other clinical practice guidelines.',
  },
  {
    icon: Shield,
    title: 'Clinically scoped',
    desc: 'Only answers documentation and guideline questions — off-topic queries are declined.',
  },
  {
    icon: Zap,
    title: 'Adaptive retrieval',
    desc: 'Automatically selects the best approach to find the most relevant guidance for your question.',
  },
]

export default function QAPage() {
  return (
    <div className="space-y-5">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-gray-900">Clinical Q&amp;A</h1>
        <p className="text-sm text-gray-500 mt-1">
          Ask questions about discharge documentation requirements and clinical guidelines.
        </p>
      </div>

      {/* Feature pills */}
      <div className="flex flex-wrap gap-3">
        {FEATURES.map(({ icon: Icon, title, desc }) => (
          <div
            key={title}
            className="flex items-start gap-3 bg-white border border-gray-200 rounded-xl px-4 py-3 flex-1 min-w-[200px]"
          >
            <div className="w-7 h-7 bg-blue-50 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5">
              <Icon className="w-3.5 h-3.5 text-blue-600" />
            </div>
            <div>
              <p className="text-xs font-semibold text-gray-800">{title}</p>
              <p className="text-xs text-gray-500 mt-0.5 leading-relaxed">{desc}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Gradio iframe */}
      <div
        className="bg-white rounded-xl border border-gray-200 overflow-hidden"
        style={{ height: 'calc(100vh - 310px)', minHeight: '480px' }}
      >
        <iframe
          src="/qa/"
          className="w-full h-full border-0"
          title="Clinical Documentation Q&A"
          allow="clipboard-write"
        />
      </div>
    </div>
  )
}
