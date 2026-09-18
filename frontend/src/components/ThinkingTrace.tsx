/*
 * ThinkingTrace is adapted for Research Scout from the Thinking component
 * pattern published by Beautiful UI (https://www.beautifului.dev/).
 * Used under the MIT License; see NOTICE for attribution and license text.
 *
 * This component intentionally renders only operational events supplied by the
 * server. It does not invent a timer, prompt, model response, or hidden chain
 * of thought in the browser.
 */
import { useEffect, useMemo, useState } from 'react'
import { AlertCircle, Check, ChevronDown, ChevronRight, Circle, ExternalLink, LoaderCircle, Radio, Sparkles } from 'lucide-react'
import type { TraceEvent, TraceStatus } from '../types'

const NODE_LABELS: Record<string, string> = {
  router: 'Router',
  search_papers: 'ArXiv search',
  eval_search: 'Relevance evaluation',
  refine_query: 'Refined query',
  read_paper: 'PDF parsing',
  web_enrich: 'GitHub & BibTeX enrichment',
  write_notes: 'PMRL notes',
  compare_benchmark: 'Benchmark comparison',
  final_report: 'Final report',
  error_handler: 'Error handling',
}

function statusLabel(status: TraceStatus): string {
  if (status === 'running') return 'đang chạy'
  if (status === 'completed') return 'hoàn tất'
  if (status === 'error') return 'có lỗi'
  if (status === 'skipped') return 'bỏ qua'
  return 'chờ xử lý'
}

function formatTime(timestamp: string): string {
  const date = new Date(timestamp)
  if (Number.isNaN(date.getTime())) return ''
  return new Intl.DateTimeFormat('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(date)
}

function formatDetails(details: TraceEvent['details']): string {
  if (!details) return ''
  if (typeof details === 'string') return details
  return Object.entries(details).map(([key, value]) => {
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return `${key}: ${value}`
    try { return `${key}: ${JSON.stringify(value)}` } catch { return `${key}: unavailable` }
  }).join('\n')
}

function EventIcon({ status }: { status: TraceStatus }) {
  if (status === 'completed') return <span className="trace-icon trace-icon--complete"><Check size={14} strokeWidth={2.5} /></span>
  if (status === 'running') return <span className="trace-icon trace-icon--running"><LoaderCircle size={15} className="spin" /></span>
  if (status === 'error') return <span className="trace-icon trace-icon--error"><AlertCircle size={15} /></span>
  if (status === 'skipped') return <span className="trace-icon trace-icon--skipped"><ChevronRight size={15} /></span>
  return <span className="trace-icon trace-icon--pending"><Circle size={10} /></span>
}

export function ThinkingTrace({ events, running = false, currentNode }: { events: TraceEvent[]; running?: boolean; currentNode?: string | null }) {
  const [open, setOpen] = useState(running)
  const [expanded, setExpanded] = useState<Set<number>>(new Set())
  const orderedEvents = useMemo(() => [...events].sort((a, b) => a.seq - b.seq), [events])

  useEffect(() => {
    if (running) setOpen(true)
  }, [running])

  useEffect(() => {
    if (!running) return
    const active = orderedEvents.find((event) => event.status === 'running')
    if (active) setExpanded((previous) => new Set(previous).add(active.seq))
  }, [orderedEvents, running])

  const latest = orderedEvents.at(-1)
  const headline = running ? (currentNode ? NODE_LABELS[currentNode] ?? currentNode : 'Đang điều phối workflow…') : latest?.summary || 'Trace workflow'

  return (
    <section className={`thinking-trace ${open ? 'is-open' : 'is-closed'}`} aria-label="Trace workflow">
      <div className="thinking-trace__head">
        <div className="thinking-trace__identity">
          <div className={`thinking-trace__orb ${running ? 'is-running' : ''}`} aria-hidden="true">{running ? <Radio size={16} /> : <Sparkles size={16} />}</div>
          <div>
            <div className="thinking-trace__title">{running ? 'Đang nghiên cứu' : 'Research trace'}</div>
            <div className="thinking-trace__headline" aria-live="polite">{headline}</div>
          </div>
        </div>
        <button type="button" className="trace-toggle" onClick={() => setOpen((value) => !value)} aria-expanded={open} aria-controls="thinking-trace-body">
          {open ? 'Thu gọn' : 'Mở trace'} {open ? <ChevronDown size={16} aria-hidden="true" /> : <ChevronRight size={16} aria-hidden="true" />}
        </button>
      </div>
      <div id="thinking-trace-body" className="thinking-trace__body" hidden={!open}>
        <div className="trace-progress" aria-hidden="true"><span style={{ width: `${latest?.progress ?? (events.length ? 0 : 0)}%` }} /></div>
        {orderedEvents.length ? (
          <ol className="trace-list" aria-live="polite">
            {orderedEvents.map((event) => {
              const isExpanded = expanded.has(event.seq)
              const hasDetails = Boolean(event.details || event.links?.length)
              return (
                <li key={`${event.seq}-${event.node}-${event.type}`} className={`trace-item trace-item--${event.status}`}>
                  <div className="trace-item__rail"><EventIcon status={event.status} /></div>
                  <div className="trace-item__content">
                    <div className="trace-item__topline">
                      <span className="trace-item__label">{event.label || NODE_LABELS[event.node] || event.node}</span>
                      <span className="trace-item__time">{formatTime(event.timestamp)}</span>
                    </div>
                    <p className="trace-item__summary">{event.summary || NODE_LABELS[event.node] || event.node}</p>
                    <div className="trace-item__meta"><code>{event.node}</code><span>·</span><span>{statusLabel(event.status)}</span>{event.progress > 0 ? <span>{Math.round(event.progress)}%</span> : null}</div>
                    {hasDetails ? (
                      <>
                        <button type="button" className="trace-detail-toggle" onClick={() => setExpanded((previous) => {
                          const next = new Set(previous)
                          if (next.has(event.seq)) next.delete(event.seq)
                          else next.add(event.seq)
                          return next
                        })} aria-expanded={isExpanded}>
                          {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />} {isExpanded ? 'Ẩn chi tiết' : 'Xem chi tiết'}
                        </button>
                        {isExpanded ? <div className="trace-item__details">
                          {event.details ? <p>{formatDetails(event.details)}</p> : null}
                          {event.links?.length ? <div className="trace-links">{event.links.map((link) => <a key={`${link.href}-${link.label}`} href={link.href} target="_blank" rel="noreferrer"><ExternalLink size={12} />{link.label}</a>)}</div> : null}
                        </div> : null}
                      </>
                    ) : null}
                  </div>
                </li>
              )
            })}
          </ol>
        ) : <div className="trace-empty"><Sparkles size={16} aria-hidden="true" /> Các node sẽ xuất hiện khi workflow bắt đầu.</div>}
      </div>
    </section>
  )
}
