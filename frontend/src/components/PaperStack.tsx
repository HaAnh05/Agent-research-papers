import { Check, FileText, Sparkles } from 'lucide-react'
import type { TraceEvent } from '../types'

const stackNodes = [
  { node: 'router', label: 'Router' },
  { node: 'search_papers', label: 'ArXiv search' },
  { node: 'eval_search', label: 'Relevance' },
  { node: 'refine_query', label: 'Refine query' },
  { node: 'read_paper', label: 'PDF parsing' },
  { node: 'web_enrich', label: 'Enrichment' },
  { node: 'write_notes', label: 'PMRL notes' },
  { node: 'compare_benchmark', label: 'Benchmark' },
  { node: 'final_report', label: 'Final report' },
]

function stateForNode(node: string, events: TraceEvent[], currentNode?: string | null): 'complete' | 'active' | 'idle' {
  const matching = events.filter((event) => event.node === node)
  if (matching.some((event) => event.status === 'completed')) return 'complete'
  if (currentNode === node || matching.some((event) => event.status === 'running')) return 'active'
  return 'idle'
}

export function PaperStack({ events = [], currentNode, compact = false }: { events?: TraceEvent[]; currentNode?: string | null; compact?: boolean }) {
  const completed = events.filter((event) => event.status === 'completed').length
  const active = Boolean(currentNode) || events.some((event) => event.status === 'running')

  return (
    <div className={`paper-stack-wrap ${compact ? 'paper-stack-wrap--compact' : ''}`}>
      <div className={`paper-stack ${active ? 'is-live' : ''}`} aria-hidden="true">
        <div className="paper-sheet paper-sheet--back" />
        <div className="paper-sheet paper-sheet--middle" />
        <div className="paper-sheet paper-sheet--front">
          <div className="paper-sheet__line paper-sheet__line--long" />
          <div className="paper-sheet__line" />
          <div className="paper-sheet__line paper-sheet__line--short" />
          <div className="paper-sheet__seal"><FileText size={18} strokeWidth={1.6} /></div>
        </div>
        <Sparkles className="paper-stack__sparkle paper-stack__sparkle--one" size={15} aria-hidden="true" />
        <Sparkles className="paper-stack__sparkle paper-stack__sparkle--two" size={11} aria-hidden="true" />
      </div>
      <div className="paper-stack-meta">
        <span className="paper-stack-meta__title">Living paper stack</span>
        <span className="paper-stack-meta__count">{completed}/{stackNodes.length} node đã sáng</span>
      </div>
      <div className="stack-node-list" aria-label="Các node workflow">
        {stackNodes.map(({ node, label }) => {
          const state = stateForNode(node, events, currentNode)
          return <span key={node} className={`stack-node stack-node--${state}`} title={label}><span className="stack-node__dot">{state === 'complete' ? <Check size={9} strokeWidth={3} /> : null}</span><span className="stack-node__label">{label}</span></span>
        })}
      </div>
    </div>
  )
}

