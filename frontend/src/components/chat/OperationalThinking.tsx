/*
 * This is a small, controlled adaptation of Beautiful UI's Thinking and Task
 * Rows patterns (https://www.beautifului.dev/r/registry.json). Raw lifecycle
 * events are folded into one row per node so the reader sees operational
 * progress, never model reasoning or a demo timer. See frontend/NOTICE.
 */
import { useEffect, useId, useMemo, useState } from 'react'
import { AlertTriangle, Check, ChevronDown, ChevronRight, Circle, LoaderCircle, Sparkles } from 'lucide-react'
import type { TraceEvent } from '../../types'
import { ArtifactLinks } from './ArtifactLinks'
import type { ChatArtifact } from './contracts'

type SemanticStep = 'Steps' | 'Search' | 'Reading' | 'Tools' | 'Route'
const TABS: SemanticStep[] = ['Steps', 'Search', 'Reading', 'Tools', 'Route']

function semanticStep(node: string): SemanticStep {
  if (node === 'router' || node === 'refine_query') return 'Route'
  if (node === 'search_papers' || node === 'eval_search') return 'Search'
  if (node === 'read_paper') return 'Reading'
  if (node === 'web_enrich' || node === 'write_notes') return 'Tools'
  return 'Steps'
}

function formatDetails(details: TraceEvent['details']): string {
  if (!details) return ''
  if (typeof details === 'string') return details
  return Object.entries(details).map(([key, value]) => {
    if (value === null || value === undefined || value === '') return ''
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return `${key}: ${value}`
    try { return `${key}: ${JSON.stringify(value)}` } catch { return `${key}: unavailable` }
  }).filter(Boolean).join(' · ')
}

function eventArtifacts(event: TraceEvent): ChatArtifact[] {
  return (event.links ?? []).map((link, index) => ({ id: `${event.seq}-${index}`, label: link.label, href: link.href, kind: event.kind }))
}

function statusIcon(status: TraceEvent['status']) {
  if (status === 'completed') return <span className="thinking-step__icon thinking-step__icon--complete"><Check size={12} /></span>
  if (status === 'error') return <span className="thinking-step__icon thinking-step__icon--error"><AlertTriangle size={12} /></span>
  if (status === 'running') return <span className="thinking-step__icon thinking-step__icon--running"><LoaderCircle size={13} className="spin" /></span>
  return <span className="thinking-step__icon"><Circle size={8} /></span>
}

export function OperationalThinking({ events, running = false, currentNode }: { events: TraceEvent[]; running?: boolean; currentNode?: string | null }) {
  const [open, setOpen] = useState(running)
  const [activeTab, setActiveTab] = useState<SemanticStep>('Steps')
  const [manuallyToggled, setManuallyToggled] = useState(false)
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const bodyId = useId()
  const steps = useMemo(() => {
    const byNode = new Map<string, { event: TraceEvent; attempts: number }>()
    for (const event of [...events].sort((a, b) => a.seq - b.seq)) {
      const previous = byNode.get(event.node)
      byNode.set(event.node, {
        event,
        attempts: (previous?.attempts ?? 0) + (event.type === 'step.started' ? 1 : 0),
      })
    }
    return [...byNode.values()].map((step) => ({ ...step, attempts: Math.max(1, step.attempts) }))
  }, [events])
  const visibleSteps = activeTab === 'Steps' ? steps : steps.filter((step) => semanticStep(step.event.node) === activeTab)

  useEffect(() => {
    if (running) {
      setOpen(true)
      setManuallyToggled(false)
    } else if (!manuallyToggled) {
      setOpen(false)
    }
  }, [running])

  if (!steps.length) return null
  return (
    <section className={`operational-thinking ${open ? 'is-open' : 'is-closed'}`} aria-label="Thinking">
      <div className="operational-thinking__head">
        <div className="operational-thinking__identity"><span className={`operational-thinking__orb ${running ? 'is-running' : ''}`} aria-hidden="true">{running ? <LoaderCircle size={14} className="spin" /> : <Sparkles size={14} />}</span><div><strong>Thinking</strong><span>{running ? (currentNode ? `Đang xử lý ${currentNode}` : 'Workflow đang xử lý') : `${events.filter((event) => event.type === 'step.started').length || steps.length} lượt vận hành`}</span></div></div>
        <button type="button" className="operational-thinking__toggle" onClick={() => { setManuallyToggled(true); setOpen((value) => !value) }} aria-expanded={open} aria-controls={bodyId}>{open ? 'Thu gọn' : 'Mở'}{open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}</button>
      </div>
      <div id={bodyId} className="operational-thinking__body" hidden={!open}>
        <div className="operational-thinking__tabs" role="tablist" aria-label="Các góc nhìn Thinking">{TABS.map((tab) => <button type="button" key={tab} role="tab" aria-selected={activeTab === tab} className={activeTab === tab ? 'is-active' : ''} onClick={() => setActiveTab(tab)}>{tab}</button>)}</div>
        <ol className="thinking-steps" aria-live="polite">
          {visibleSteps.map(({ event, attempts }) => {
            const key = event.node
            const isExpanded = expanded.has(key)
            const details = formatDetails(event.details)
            const artifacts = eventArtifacts(event)
            const hasDetails = Boolean(details || artifacts.length)
            return <li className={`thinking-step thinking-step--${event.status}`} key={key} data-attempts={attempts}>{statusIcon(event.status)}<div className="thinking-step__content"><div className="thinking-step__topline"><span className="thinking-step__semantic">{semanticStep(event.node)}</span><strong>{event.label || event.node}</strong>{attempts > 1 ? <span className="thinking-step__attempts">×{attempts}</span> : null}</div><p>{event.summary || event.node}</p>{hasDetails ? <><button type="button" className="thinking-step__details-toggle" onClick={() => setExpanded((previous) => { const next = new Set(previous); if (next.has(key)) next.delete(key); else next.add(key); return next })} aria-expanded={isExpanded}>{isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />} {isExpanded ? 'Ẩn chi tiết' : 'Xem chi tiết'}</button>{isExpanded ? <div className="thinking-step__details">{details ? <p>{details}</p> : null}<ArtifactLinks artifacts={artifacts} /></div> : null}</> : null}</div></li>
          })}
        </ol>
        {!visibleSteps.length ? <p className="operational-thinking__empty">Chưa có event cho mục này.</p> : null}
      </div>
    </section>
  )
}
