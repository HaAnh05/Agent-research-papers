/*
 * Research Inspector
 *
 * The visual language of this workspace is adapted from the MIT-licensed
 * Beautiful UI primitives (https://www.beautifului.dev/).  This file keeps
 * the adapters controlled by real Research Scout DTOs instead of importing
 * the demo timers/data from those primitives.  See frontend/NOTICE.
 */
import { useEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent, type MouseEvent } from 'react'
import {
  AlertCircle,
  ArrowDown,
  ArrowUpRight,
  BookOpen,
  Check,
  CheckCircle2,
  ChevronDown,
  Circle,
  Clipboard,
  ClipboardCheck,
  Clock3,
  Code2,
  Download,
  ExternalLink,
  FileCode2,
  FileText,
  Filter,
  Github,
  Hash,
  Info,
  Library,
  Link2,
  LoaderCircle,
  MessageSquareQuote,
  Network,
  PanelRight,
  Search,
  Sparkles,
  Tags,
  Terminal,
  Wrench,
  X,
  type LucideIcon,
} from 'lucide-react'
import { MarkdownContent, titleFromMarkdown } from '../MarkdownContent'
import { downloadReport } from '../../lib/api'
import {
  readBenchmark,
  readBibtex,
  readPaperId,
  readPaperNotes,
  readPaperPdfUrl,
  readPaperRepos,
  readReportMarkdown,
  readResultPapers,
  type PaperSource,
  type PMRLNotes,
  type RunResult,
  type RunSnapshot,
  type TraceEvent,
  type TraceStatus,
} from '../../types'
import './inspector.css'

export interface InspectorWorkspaceProps {
  snapshot: RunSnapshot | null
  events: TraceEvent[]
  onAskSelection?: (text: string) => void
}

type InspectorTab = 'run' | 'papers' | 'pmrl' | 'compare' | 'report' | 'trace'
type PaperFilter = 'all' | 'pmrl' | 'repository' | 'bibtex'
type CopyState = 'idle' | 'copied' | 'error'
type DownloadState = 'idle' | 'downloading' | 'downloaded' | 'error' | 'unavailable'

const TABS: Array<{ id: InspectorTab; label: string; icon: LucideIcon }> = [
  { id: 'run', label: 'Run', icon: Network },
  { id: 'papers', label: 'Papers', icon: Library },
  { id: 'pmrl', label: 'PMRL', icon: BookOpen },
  { id: 'compare', label: 'Compare', icon: Network },
  { id: 'report', label: 'Report', icon: FileText },
  { id: 'trace', label: 'Trace', icon: Terminal },
]

const NODE_ORDER = [
  'router',
  'search_papers',
  'eval_search',
  'refine_query',
  'read_paper',
  'web_enrich',
  'write_notes',
  'compare_benchmark',
  'final_report',
  'error_handler',
]

const NODE_META: Record<string, { label: string; subtitle: string }> = {
  router: { label: 'Router', subtitle: 'understand request' },
  search_papers: { label: 'ArXiv search', subtitle: 'retrieve candidates' },
  eval_search: { label: 'Relevance', subtitle: 'evaluate candidates' },
  refine_query: { label: 'Refine query', subtitle: 'retry search branch' },
  read_paper: { label: 'PDF parsing', subtitle: 'read selected papers' },
  web_enrich: { label: 'Enrichment', subtitle: 'GitHub and BibTeX' },
  write_notes: { label: 'PMRL notes', subtitle: 'structure findings' },
  compare_benchmark: { label: 'Benchmark', subtitle: 'compare papers' },
  final_report: { label: 'Final report', subtitle: 'compose Markdown' },
  error_handler: { label: 'Error handler', subtitle: 'surface failure' },
}

/** The graph topology mirrors graph.py; statuses are filled only from events. */
const GRAPH_EDGES: Array<{ from: string; to: string; label?: string }> = [
  { from: 'router', to: 'search_papers', label: 'search' },
  { from: 'router', to: 'read_paper', label: 'direct paper' },
  { from: 'router', to: 'error_handler', label: 'error' },
  { from: 'search_papers', to: 'eval_search' },
  { from: 'eval_search', to: 'refine_query', label: 'retry' },
  { from: 'eval_search', to: 'read_paper', label: 'relevant' },
  { from: 'eval_search', to: 'error_handler', label: 'exhausted' },
  { from: 'refine_query', to: 'search_papers', label: 'loop' },
  { from: 'read_paper', to: 'web_enrich' },
  { from: 'web_enrich', to: 'write_notes' },
  { from: 'write_notes', to: 'compare_benchmark', label: '2+ papers' },
  { from: 'write_notes', to: 'final_report', label: 'report' },
  { from: 'write_notes', to: 'error_handler', label: 'error' },
  { from: 'compare_benchmark', to: 'final_report' },
]

const FLOW_POSITIONS: Record<string, { x: number; y: number }> = {
  router: { x: 50, y: 7 },
  search_papers: { x: 50, y: 21 },
  eval_search: { x: 50, y: 36 },
  refine_query: { x: 83, y: 36 },
  read_paper: { x: 50, y: 51 },
  web_enrich: { x: 50, y: 65 },
  write_notes: { x: 50, y: 79 },
  compare_benchmark: { x: 18, y: 94 },
  final_report: { x: 50, y: 94 },
  error_handler: { x: 84, y: 65 },
}

const FLOW_PATHS: Record<string, string> = {
  'router>search_papers': 'M50 10 L50 19',
  'router>read_paper': 'M50 10 C63 22 63 35 50 48',
  'router>error_handler': 'M50 10 C72 22 84 39 84 62',
  'search_papers>eval_search': 'M50 24 L50 33',
  'eval_search>refine_query': 'M54 36 C65 35 74 35 80 36',
  'eval_search>read_paper': 'M50 39 L50 48',
  'eval_search>error_handler': 'M54 37 C68 42 78 52 84 62',
  'refine_query>search_papers': 'M80 39 C74 49 62 49 54 23',
  'read_paper>web_enrich': 'M50 54 L50 62',
  'web_enrich>write_notes': 'M50 68 L50 76',
  'write_notes>compare_benchmark': 'M47 81 C39 86 26 90 18 91',
  'write_notes>final_report': 'M50 82 L50 91',
  'write_notes>error_handler': 'M54 81 C68 80 78 75 84 68',
  'compare_benchmark>final_report': 'M21 95 C31 99 42 98 50 95',
}

const NODE_LABELS: Record<string, string> = Object.fromEntries(
  Object.entries(NODE_META).map(([node, meta]) => [node, meta.label]),
)

function resultFrom(snapshot: RunSnapshot | null): RunResult | null {
  return snapshot?.result ?? null
}

function eventOrder(events: TraceEvent[]): TraceEvent[] {
  return [...events].sort((a, b) => a.seq - b.seq || a.timestamp.localeCompare(b.timestamp))
}

function latestEventsByNode(events: TraceEvent[]): Map<string, TraceEvent> {
  const latest = new Map<string, TraceEvent>()
  for (const event of eventOrder(events)) latest.set(event.node, event)
  return latest
}

function statusForNode(node: string, latest: Map<string, TraceEvent>): TraceStatus {
  const event = latest.get(node)
  if (!event) return 'pending'
  if (event.status === 'failed') return 'error'
  return event.status
}

function statusLabel(status: TraceStatus): string {
  switch (status) {
    case 'running': return 'đang chạy'
    case 'retrying': return 'đang retry'
    case 'completed': return 'hoàn tất'
    case 'error':
    case 'failed': return 'có lỗi'
    case 'skipped': return 'bỏ qua'
    default: return 'chưa chạy'
  }
}

function statusIcon(status: TraceStatus) {
  if (status === 'completed') return <Check size={13} strokeWidth={2.8} aria-hidden="true" />
  if (status === 'running' || status === 'retrying') return <LoaderCircle size={13} className="inspector-spin" aria-hidden="true" />
  if (status === 'error' || status === 'failed') return <AlertCircle size={13} aria-hidden="true" />
  if (status === 'skipped') return <ChevronDown size={13} aria-hidden="true" />
  return <Circle size={8} aria-hidden="true" />
}

function formatDuration(value?: number | null): string {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return ''
  if (value < 1000) return `${Math.round(value)} ms`
  const seconds = value / 1000
  if (seconds < 60) return `${seconds.toFixed(seconds >= 10 ? 0 : 1)} s`
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
}

function formatTimestamp(value?: string): string {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return new Intl.DateTimeFormat('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(date)
}

function detailRecord(event?: TraceEvent): Record<string, unknown> {
  return event?.details && typeof event.details === 'object' ? event.details : {}
}

function detailText(event?: TraceEvent, key?: string): string {
  if (!event || !key) return ''
  const value = detailRecord(event)[key]
  return typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean' ? String(value) : ''
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function eventDetailsText(event: TraceEvent): string {
  if (!event.details) return ''
  if (typeof event.details === 'string') return event.details
  return Object.entries(event.details).map(([key, value]) => {
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return `${key}: ${value}`
    return `${key}: ${safeJson(value)}`
  }).join('\n')
}

function paperKey(paper: PaperSource, index: number): string {
  return readPaperId(paper) ?? `${paper.title}-${index}`
}

function paperHref(paper: PaperSource): string | undefined {
  const id = readPaperId(paper)?.replace(/^arxiv:/i, '')
  if (id) return `https://arxiv.org/abs/${encodeURIComponent(id)}`
  return readPaperPdfUrl(paper)
}

function paperSourceLabel(paper: PaperSource): string {
  const source = paper.sourceType ?? paper.source_type
  return source === 'local_pdf' ? 'Local PDF' : 'ArXiv'
}

function repoCount(paper: PaperSource): number {
  return readPaperRepos(paper).length
}

function CopyButton({ value, label = 'Sao chép' }: { value: string; label?: string }) {
  const [state, setState] = useState<CopyState>('idle')

  const handleCopy = async () => {
    if (!value.trim()) return
    try {
      if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable')
      await navigator.clipboard.writeText(value)
      setState('copied')
    } catch {
      setState('error')
    }
  }

  return (
    <button type="button" className="inspector-action" onClick={() => void handleCopy()} disabled={!value.trim()}>
      {state === 'copied' ? <ClipboardCheck size={13} aria-hidden="true" /> : state === 'error' ? <AlertCircle size={13} aria-hidden="true" /> : <Clipboard size={13} aria-hidden="true" />}
      {state === 'copied' ? 'Đã sao chép' : state === 'error' ? 'Không thể sao chép' : label}
    </button>
  )
}

function CodeBlock({ code, language = 'text', filename }: { code: string; language?: string; filename?: string }) {
  const lines = code.split('\n')
  return (
    <div className="inspector-code-block">
      <div className="inspector-code-block__head">
        <span><Code2 size={13} aria-hidden="true" />{filename ?? language}</span>
        <CopyButton value={code} label="Copy" />
      </div>
      <pre aria-label={filename ?? `${language} code`}><code>{lines.map((line, index) => <span className="inspector-code-line" key={`${index}-${line}`}><span className="inspector-code-line__number" aria-hidden="true">{String(index + 1).padStart(2, '0')}</span><span>{line || ' '}</span></span>)}</code></pre>
    </div>
  )
}

function InspectorLoading({ label }: { label: string }) {
  return <div className="inspector-loading" role="status" aria-live="polite"><LoaderCircle size={16} className="inspector-spin" aria-hidden="true" /><span>{label}</span></div>
}

function SectionHeader({ eyebrow, title, count, action }: { eyebrow?: string; title: string; count?: number; action?: React.ReactNode }) {
  return (
    <div className="inspector-section-head">
      <div>
        {eyebrow ? <span className="inspector-eyebrow">{eyebrow}</span> : null}
        <h3>{title}{typeof count === 'number' ? <span className="inspector-count">{count}</span> : null}</h3>
      </div>
      {action ? <div className="inspector-section-head__action">{action}</div> : null}
    </div>
  )
}

function StatusPill({ status }: { status: string }) {
  const label = status === 'success' ? 'Hoàn tất' : status === 'degraded' ? 'Có cảnh báo' : status === 'error' ? 'Có lỗi' : status === 'running' ? 'Đang chạy' : 'Chờ chạy'
  return <span className={`inspector-status inspector-status--${status}`}><span className="inspector-status__dot" aria-hidden="true" />{label}</span>
}

function TaskRows({ snapshot, events }: { snapshot: RunSnapshot | null; events: TraceEvent[] }) {
  const latest = useMemo(() => latestEventsByNode(events), [events])
  const nodes = useMemo(() => {
    const unknown = [...new Set(events.map((event) => event.node).filter((node) => node && !NODE_META[node]))]
    return [...NODE_ORDER, ...unknown]
  }, [events])

  if (!snapshot && !events.length) return <InspectorLoading label="Chưa có event từ workflow." />

  return (
    <div className="inspector-task-rows" role="list" aria-label="Trạng thái các node LangGraph">
      {nodes.map((node) => {
        const event = latest.get(node)
        const status = statusForNode(node, latest)
        const detail = detailRecord(event)
        const metric = detailText(event, 'resultCount') || detailText(event, 'selectedCount') || detailText(event, 'paperCount')
        const metricLabel = detailText(event, 'resultCount') ? 'nguồn' : detailText(event, 'selectedCount') ? 'đã chọn' : detailText(event, 'paperCount') ? 'paper' : ''
        return (
          <div className={`inspector-task-row inspector-task-row--${status}`} key={node} role="listitem">
            <span className="inspector-task-row__state" aria-label={statusLabel(status)}>{statusIcon(status)}</span>
            <div className="inspector-task-row__main">
              <div className="inspector-task-row__title"><strong>{NODE_LABELS[node] ?? node}</strong><span>{statusLabel(status)}</span></div>
              <p>{event?.summary || NODE_META[node]?.subtitle || 'node chưa phát event'}</p>
              <div className="inspector-task-row__meta">
                <code>{node}</code>
                {metric ? <span>{metric} {metricLabel}</span> : null}
                {event?.retry ? <span>attempt {event.retry.current}{event.retry.max ? `/${event.retry.max}` : ''}</span> : null}
                {event?.durationMs != null ? <span><Clock3 size={11} aria-hidden="true" />{formatDuration(event.durationMs)}</span> : null}
                {event?.timestamp ? <time dateTime={event.timestamp}>{formatTimestamp(event.timestamp)}</time> : null}
              </div>
              {Object.keys(detail).length && event?.status === 'error' ? <details className="inspector-technical-detail"><summary>Chi tiết kỹ thuật</summary><CodeBlock code={eventDetailsText(event)} language="event" /></details> : null}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function ToolChips({ events }: { events: TraceEvent[] }) {
  const chips = useMemo(() => {
    const byTool = new Map<string, { label: string; event: TraceEvent }>()
    const completedStages: Record<string, string[]> = {
      search_papers: ['ArXiv API'],
      read_paper: ['PDF parser'],
      web_enrich: ['GitHub search', 'BibTeX generator'],
      write_notes: ['LLM · PMRL'],
      final_report: ['LLM · report'],
    }
    eventOrder(events).forEach((event) => {
      if (event.type === 'tool.started' || event.type === 'tool.completed') {
        byTool.set(event.node || event.label || 'tool', { label: event.label || event.node, event })
      } else if (event.type === 'step.completed') {
        // These nodes synchronously invoke the named integration. A completed
        // node proves the stage ran; no per-request count or timing is claimed.
        for (const label of completedStages[event.node] ?? []) byTool.set(label, { label, event })
      }
    })
    return [...byTool.entries()].map(([key, value]) => ({ key, ...value }))
  }, [events])

  return (
    <div className="inspector-tool-section">
      <div className="inspector-chip-row" aria-label="Các integration stage đã thực thi">
        {chips.map(({ key, label, event }) => <span className={`inspector-tool-chip inspector-tool-chip--${event.status}`} key={key}><Wrench size={12} aria-hidden="true" /><span>{label}</span><small>{statusLabel(event.status)}</small></span>)}
      </div>
      {!chips.length ? <p className="inspector-muted">Chưa có integration stage hoàn tất.</p> : null}
    </div>
  )
}

function ThinkingNarrative({ snapshot, events }: { snapshot: RunSnapshot | null; events: TraceEvent[] }) {
  const [open, setOpen] = useState(snapshot?.status === 'running')
  useEffect(() => {
    if (snapshot?.status === 'running') setOpen(true)
  }, [snapshot?.runId, snapshot?.status])
  const steps = useMemo(() => eventOrder(events).filter((event) => event.type.startsWith('step.') || event.type === 'run.started' || event.type === 'run.completed' || event.type === 'run.failed'), [events])
  const visibleSteps = steps.slice(-8)
  const headline = snapshot?.status === 'running' ? (snapshot.currentNode ? NODE_LABELS[snapshot.currentNode] ?? snapshot.currentNode : 'Đang điều phối workflow') : steps.at(-1)?.summary || 'Chưa có operational trace'

  return (
    <section className={`inspector-thinking ${open ? 'is-open' : ''}`} aria-label="Operational thinking">
      <div className="inspector-thinking__head">
        <div className="inspector-thinking__identity"><span className={`inspector-thinking__mark ${snapshot?.status === 'running' ? 'is-live' : ''}`}><Sparkles size={15} aria-hidden="true" /></span><div><span className="inspector-eyebrow">Thinking</span><strong>{headline}</strong></div></div>
        <button type="button" className="inspector-quiet-button" onClick={() => setOpen((value) => !value)} aria-expanded={open}>{open ? 'Thu gọn' : 'Mở'}<ChevronDown size={14} aria-hidden="true" /></button>
      </div>
      {open ? <div className="inspector-thinking__body">
        {visibleSteps.length ? <ol className="inspector-thinking__list" aria-live="polite">{visibleSteps.map((event) => <li key={`${event.seq}-${event.type}`}><span className={`inspector-thinking__step inspector-thinking__step--${event.status}`}>{statusIcon(event.status)}</span><div><strong>{event.label || NODE_LABELS[event.node] || event.node}</strong><p>{event.summary || 'Operational update'}</p></div><time dateTime={event.timestamp}>{formatTimestamp(event.timestamp)}</time></li>)}</ol> : <InspectorLoading label="Đang chờ semantic workflow event." />}
        <p className="inspector-thinking__note">Đây là tóm tắt vận hành từ event công khai, không phải chain-of-thought.</p>
      </div> : null}
    </section>
  )
}

function RunArtifacts({ result }: { result: RunResult | null }) {
  const papers = result?.papers ?? []
  const pmrlCount = papers.filter((paper) => Boolean(readPaperNotes(paper))).length
  const report = readReportMarkdown(result)
  const artifacts = [
    { label: 'Papers', value: papers.length, icon: Library },
    { label: 'PMRL', value: pmrlCount, icon: BookOpen },
    { label: 'Compare', value: result?.benchmark ? 1 : 0, icon: Network },
    { label: 'Report', value: report ? 1 : 0, icon: FileText },
  ]
  return <div className="inspector-artifact-strip" aria-label="Research artifacts">{artifacts.map(({ label, value, icon: Icon }) => <div key={label} className={value ? 'has-value' : ''}><Icon size={14} aria-hidden="true" /><span><strong>{value}</strong>{label}</span></div>)}</div>
}

function RunPanel({ snapshot, events }: { snapshot: RunSnapshot | null; events: TraceEvent[] }) {
  if (!snapshot) return <InspectorLoading label="Đang chờ research run…" />
  const result = resultFrom(snapshot)
  const latest = eventOrder(events).at(-1)
  const errorEvent = eventOrder(events).reverse().find((event) => event.status === 'error' || event.status === 'failed')
  const complete = snapshot.status === 'success' || snapshot.status === 'degraded'
  const error = snapshot.error ?? snapshot.errorMessage ?? errorEvent?.summary

  return <div className="inspector-panel inspector-panel--run">
    <div className="inspector-run-summary"><div><span className="inspector-eyebrow">Research run</span><h3>{snapshot.input.query || 'Direct paper analysis'}</h3><div className="inspector-run-meta"><code>{snapshot.runId}</code><span>{snapshot.provider}</span><span>{snapshot.model}</span></div></div><StatusPill status={snapshot.status} /></div>
    <div className="inspector-run-progress" aria-live="polite"><span>{complete ? 'Workflow hoàn tất' : snapshot.status === 'error' ? 'Workflow dừng do lỗi' : `${events.filter((event) => event.type === 'step.completed').length} node hoàn tất`}</span><span>{latest ? `Cập nhật ${formatTimestamp(latest.timestamp)}` : 'Chưa có event'}</span></div>
    <ThinkingNarrative snapshot={snapshot} events={events} />
    <SectionHeader eyebrow="Execution" title="Task rows" />
    <TaskRows snapshot={snapshot} events={events} />
    <SectionHeader eyebrow="Integrations" title="Tool chips" />
    <ToolChips events={events} />
    <SectionHeader eyebrow="Artifacts" title="Research output" />
    <RunArtifacts result={result} />
    {error ? <div className="inspector-run-error" role="alert"><AlertCircle size={16} aria-hidden="true" /><div><strong>Workflow cần chú ý</strong><p>{error}</p><details><summary>Chi tiết technical event</summary>{errorEvent ? <CodeBlock code={eventDetailsText(errorEvent) || errorEvent.summary} language="event" /> : <p className="inspector-muted">Server không phát chi tiết thêm.</p>}</details></div></div> : null}
  </div>
}

function SearchField({ value, onChange, placeholder }: { value: string; onChange: (value: string) => void; placeholder: string }) {
  return <label className="inspector-search"><Search size={14} aria-hidden="true" /><span className="sr-only">{placeholder}</span><input value={value} onChange={(event) => onChange(event.target.value)} placeholder={placeholder} type="search" /></label>
}

function PaperLink({ paper }: { paper: PaperSource }) {
  const href = paperHref(paper)
  if (!href) return null
  return <a className="inspector-inline-link" href={href} target="_blank" rel="noreferrer"><ExternalLink size={12} aria-hidden="true" />Mở nguồn</a>
}

function PaperContextCard({ paper }: { paper: PaperSource }) {
  const notes = readPaperNotes(paper)
  const source = readPaperId(paper) ?? paper.title
  return <div className="inspector-context-card inspector-selectable"><div className="inspector-context-card__head"><span className="inspector-eyebrow">Context card</span><span className="inspector-context-card__source"><Link2 size={11} aria-hidden="true" />Source: {source}</span></div><p>{paper.summary || 'Paper đã được chọn cho phiên này; backend không trả về đoạn trích section-level.'}</p>{notes ? <div className="inspector-context-card__foot"><span>PMRL đã sẵn sàng</span><span>Provenance: paper-level</span></div> : null}</div>
}

function PaperDetail({ paper }: { paper: PaperSource }) {
  const repos = readPaperRepos(paper)
  return <div className="inspector-paper-detail"><div className="inspector-paper-detail__heading"><div><span className="inspector-eyebrow">Selected paper</span><h4>{paper.title}</h4></div><PaperLink paper={paper} /></div><div className="inspector-paper-detail__meta"><code>{readPaperId(paper) ?? paperSourceLabel(paper)}</code>{paper.published ? <span>{paper.published}</span> : null}{paper.authors?.length ? <span>{paper.authors.slice(0, 3).join(', ')}{paper.authors.length > 3 ? ' +' : ''}</span> : null}</div><PaperContextCard paper={paper} />{repos.length ? <div className="inspector-repository-list"><span className="inspector-eyebrow"><Github size={12} aria-hidden="true" /> Related repository</span>{repos.map((repo) => <a key={repo.url} href={repo.url} target="_blank" rel="noreferrer"><span>{repo.name}</span><ArrowUpRight size={12} aria-hidden="true" /></a>)}</div> : null}</div>
}

function PapersPanel({ result }: { result: RunResult | null }) {
  const papers = result?.papers ?? []
  const [filter, setFilter] = useState<PaperFilter>('all')
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState<string | null>(papers.length ? paperKey(papers[0], 0) : null)
  useEffect(() => {
    if (!papers.some((paper, index) => paperKey(paper, index) === selected)) setSelected(papers.length ? paperKey(papers[0], 0) : null)
  }, [papers, selected])
  const visible = useMemo(() => papers.filter((paper, index) => {
    const haystack = [paper.title, readPaperId(paper), paper.summary, ...(paper.authors ?? [])].filter(Boolean).join(' ').toLowerCase()
    if (search.trim() && !haystack.includes(search.trim().toLowerCase())) return false
    if (filter === 'pmrl' && !readPaperNotes(paper)) return false
    if (filter === 'repository' && !repoCount(paper)) return false
    if (filter === 'bibtex' && !paper.bibtex) return false
    return Boolean(paperKey(paper, index))
  }), [filter, papers, search])
  const activePaper = papers.find((paper, index) => paperKey(paper, index) === selected)

  if (!papers.length) return <div className="inspector-empty"><Library size={19} aria-hidden="true" /><h3>Chưa có paper được chọn</h3><p>Papers sẽ xuất hiện khi Router và các node đọc paper hoàn tất.</p></div>

  return <div className="inspector-panel inspector-panel--papers"><div className="inspector-panel-toolbar"><SearchField value={search} onChange={setSearch} placeholder="Tìm trong papers…" /><div className="inspector-filter-row" role="group" aria-label="Lọc papers">{([{ id: 'all', label: 'Tất cả' }, { id: 'pmrl', label: 'PMRL' }, { id: 'repository', label: 'Repository match' }, { id: 'bibtex', label: 'BibTeX' }] as Array<{ id: PaperFilter; label: string }>).map((item) => <button type="button" key={item.id} className={filter === item.id ? 'is-active' : ''} onClick={() => setFilter(item.id)}>{item.label}</button>)}</div></div><SectionHeader eyebrow="Records table · session data" title="Selected papers" count={visible.length} /><div className="inspector-records-wrap"><table className="inspector-records"><caption className="sr-only">Selected papers in this research run</caption><thead><tr><th scope="col">Paper</th><th scope="col">Source</th><th scope="col">Artifacts</th><th scope="col"><span className="sr-only">Open</span></th></tr></thead><tbody>{visible.map((paper) => { const index = papers.indexOf(paper); const key = paperKey(paper, index); const notes = Boolean(readPaperNotes(paper)); const repos = repoCount(paper); return <tr key={key} className={selected === key ? 'is-selected' : ''}><th scope="row"><button type="button" className="inspector-record-button" onClick={() => setSelected(key)} aria-pressed={selected === key}><span>{paper.title}</span><small>{readPaperId(paper) ?? 'Local PDF'}</small></button></th><td><span className="inspector-tag">{paperSourceLabel(paper)}</span></td><td><div className="inspector-record-artifacts">{notes ? <span title="PMRL available"><BookOpen size={12} aria-hidden="true" /> PMRL</span> : null}{repos ? <span title="Related repository"><Github size={12} aria-hidden="true" /> Repo</span> : null}{paper.bibtex ? <span title="BibTeX available"><FileCode2 size={12} aria-hidden="true" /> Bib</span> : null}{!notes && !repos && !paper.bibtex ? <span className="inspector-muted">—</span> : null}</div></td><td><PaperLink paper={paper} /></td></tr> })}</tbody></table>{!visible.length ? <div className="inspector-empty inspector-empty--compact"><Filter size={17} aria-hidden="true" />Không có paper khớp bộ lọc.</div> : null}</div>{activePaper ? <PaperDetail paper={activePaper} /> : null}<CitationsBlock papers={papers} /> </div>
}

function CitationsBlock({ papers }: { papers: PaperSource[] }) {
  const bibtex = papers.map((paper) => paper.bibtex).filter((entry): entry is string => Boolean(entry?.trim())).join('\n\n')
  if (!bibtex) return null
  return <section className="inspector-citations"><SectionHeader eyebrow="Code block" title="BibTeX" action={<CopyButton value={bibtex} label="Copy all" />} /><CodeBlock code={bibtex} language="bibtex" filename="references.bib" /></section>
}

function PMRLField({ label, value, tone }: { label: string; value: string; tone: string }) {
  return <section className={`inspector-pmrl-field inspector-pmrl-field--${tone} inspector-selectable`}><span>{label}</span><p>{value || 'Backend chưa trả về mục này.'}</p></section>
}

function PMRLCard({ paper, notes, index }: { paper: PaperSource; notes: PMRLNotes; index: number }) {
  return <article className="inspector-pmrl-card"><div className="inspector-pmrl-card__head"><span className="inspector-index">{String(index + 1).padStart(2, '0')}</span><div><span className="inspector-eyebrow">Paper</span><h4>{paper.title}</h4><small>{readPaperId(paper) ?? paperSourceLabel(paper)}</small></div><PaperLink paper={paper} /></div><div className="inspector-pmrl-grid"><PMRLField label="P · Problem" value={notes.problem} tone="problem" /><PMRLField label="M · Method" value={notes.method} tone="method" /><PMRLField label="R · Result" value={notes.result} tone="result" /><PMRLField label="L · Limitation" value={notes.limitation} tone="limitation" /></div></article>
}

function PMRLPanel({ result }: { result: RunResult | null }) {
  const papers = (result?.papers ?? []).filter((paper) => Boolean(readPaperNotes(paper)))
  if (!papers.length) return <div className="inspector-empty"><BookOpen size={19} aria-hidden="true" /><h3>PMRL chưa sẵn sàng</h3><p>Notes chỉ hiển thị khi node `write_notes` trả về dữ liệu có cấu trúc.</p></div>
  return <div className="inspector-panel inspector-panel--pmrl"><SectionHeader eyebrow="Structured paper notes" title="Problem · Method · Result · Limitation" count={papers.length} /><p className="inspector-panel-intro">PMRL là artifact từ backend; chọn một đoạn văn để hỏi Research Agent nếu shell cung cấp callback.</p><div className="inspector-pmrl-list">{papers.map((paper, index) => <PMRLCard key={`${readPaperId(paper) ?? paper.title}-${index}`} paper={paper} notes={readPaperNotes(paper) as PMRLNotes} index={index} />)}</div></div>
}

function MarkdownArtifact({ content, empty, selectable = true }: { content: string; empty: string; selectable?: boolean }) {
  if (!content.trim()) return <div className="inspector-empty inspector-empty--inline"><FileText size={18} aria-hidden="true" /><p>{empty}</p></div>
  return <div className={selectable ? 'inspector-markdown inspector-selectable' : 'inspector-markdown'}><MarkdownContent content={content} /></div>
}

function ComparePanel({ result }: { result: RunResult | null }) {
  const benchmark = readBenchmark(result)
  return <div className="inspector-panel inspector-panel--compare"><SectionHeader eyebrow="Backend artifact" title="Benchmark matrix" count={benchmark ? 1 : 0} /><MarkdownArtifact content={benchmark} empty="Benchmark chỉ xuất hiện khi workflow tạo ma trận so sánh." /></div>
}

function ReportPanel({ snapshot, result }: { snapshot: RunSnapshot | null; result: RunResult | null }) {
  const markdown = readReportMarkdown(result)
  const reportId = result && typeof result === 'object' ? result.reportId ?? result.report_id : null
  const [downloadState, setDownloadState] = useState<DownloadState>('idle')
  const title = titleFromMarkdown(markdown, snapshot?.input.query || 'Research report')

  const handleDownload = async () => {
    if (!reportId) {
      setDownloadState('unavailable')
      return
    }
    setDownloadState('downloading')
    try {
      const blob = await downloadReport(reportId)
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `${reportId.replace(/\.md$/i, '')}.md`
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.setTimeout(() => URL.revokeObjectURL(url), 0)
      setDownloadState('downloaded')
    } catch {
      setDownloadState('error')
    }
  }

  return <div className="inspector-panel inspector-panel--report"><div className="inspector-report-head"><div><span className="inspector-eyebrow">Research report</span><h3>{title}</h3>{reportId ? <code>{reportId}</code> : <small>Report endpoint chưa trả về report ID.</small>}</div><div className="inspector-report-actions"><CopyButton value={markdown} label="Copy Markdown" /><button type="button" className="inspector-action" onClick={() => void handleDownload()} disabled={!markdown.trim() || downloadState === 'downloading'} title={!reportId ? 'Chưa có report ID từ backend' : undefined}><Download size={13} aria-hidden="true" />{downloadState === 'downloading' ? 'Đang tải…' : downloadState === 'downloaded' ? 'Đã tải' : downloadState === 'error' ? 'Tải lỗi' : downloadState === 'unavailable' ? 'Thiếu report ID' : 'Download'}</button></div></div><MarkdownArtifact content={markdown} empty={snapshot?.status === 'running' ? 'Report sẽ xuất hiện sau node final_report.' : 'Chưa có report Markdown cho run này.'} /></div>
}

function edgeKey(edge: { from: string; to: string }): string {
  return `${edge.from}>${edge.to}`
}

function edgeObserved(edge: { from: string; to: string }, first: Map<string, TraceEvent>, latest: Map<string, TraceEvent>): boolean {
  const source = latest.get(edge.from)
  const target = first.get(edge.to)
  return Boolean(source && target && target.seq > source.seq)
}

function FlowNode({ node, status }: { node: string; status: TraceStatus }) {
  const position = FLOW_POSITIONS[node]
  if (!position) return null
  const style = { '--flow-x': `${position.x}%`, '--flow-y': `${position.y}%` } as CSSProperties
  const meta = NODE_META[node] ?? { label: node, subtitle: 'event node' }
  return <div className={`inspector-flow-node inspector-flow-node--${status}`} style={style}><span className="inspector-flow-node__state" aria-label={statusLabel(status)}>{statusIcon(status)}</span><strong>{meta.label}</strong><small>{meta.subtitle}</small></div>
}

function Flowchart({ events }: { events: TraceEvent[] }) {
  const latest = useMemo(() => latestEventsByNode(events), [events])
  const first = useMemo(() => {
    const result = new Map<string, TraceEvent>()
    for (const event of eventOrder(events)) if (!result.has(event.node)) result.set(event.node, event)
    return result
  }, [events])
  return <div className="inspector-flowchart"><div className="inspector-flowchart__legend"><span><i className="is-observed" />Observed event</span><span><i className="is-running" />Current / running</span><span><i className="is-unvisited" />Not visited</span></div><div className="inspector-flowchart__canvas" role="img" aria-label="LangGraph workflow topology and observed node states"><svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><defs><marker id="inspector-arrow" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 z" /></marker></defs>{GRAPH_EDGES.map((edge) => { const observed = edgeObserved(edge, first, latest); const path = FLOW_PATHS[edgeKey(edge)]; return path ? <path key={edgeKey(edge)} className={observed ? 'is-observed' : ''} d={path} markerEnd="url(#inspector-arrow)" /> : null })}</svg>{Object.keys(FLOW_POSITIONS).map((node) => <FlowNode key={node} node={node} status={statusForNode(node, latest)} />)}</div><div className="inspector-edge-ledger" aria-label="Graph edges">{GRAPH_EDGES.map((edge) => { const observed = edgeObserved(edge, first, latest); return <div className={observed ? 'is-observed' : ''} key={edgeKey(edge)}><span>{NODE_LABELS[edge.from] ?? edge.from}</span><ArrowDown size={11} aria-hidden="true" /><span>{NODE_LABELS[edge.to] ?? edge.to}</span>{edge.label ? <small>{edge.label}</small> : null}</div> })}</div></div>
}

function TracePanel({ events }: { events: TraceEvent[] }) {
  const ordered = eventOrder(events)
  if (!ordered.length) return <div className="inspector-empty"><Terminal size={19} aria-hidden="true" /><h3>Trace chưa có event</h3><p>Flowchart sẽ hiển thị topology ngay khi run phát operational events.</p></div>
  return <div className="inspector-panel inspector-panel--trace"><SectionHeader eyebrow="Flowchart · graph.py" title="LangGraph trace" count={ordered.length} /><p className="inspector-panel-intro">Topology là graph tĩnh đã kiểm tra; màu trạng thái bên dưới chỉ phản ánh event server đã phát.</p><Flowchart events={ordered} /><section className="inspector-event-ledger"><SectionHeader eyebrow="Append-only events" title="Event ledger" />{ordered.map((event) => <details className={`inspector-event-row inspector-event-row--${event.status}`} key={`${event.seq}-${event.node}-${event.type}`}><summary><span className="inspector-event-row__seq">#{event.seq}</span><span className="inspector-event-row__icon">{statusIcon(event.status)}</span><span className="inspector-event-row__label">{event.label || NODE_LABELS[event.node] || event.node}</span><code>{event.node}</code><time dateTime={event.timestamp}>{formatTimestamp(event.timestamp)}</time></summary><div className="inspector-event-row__body"><p>{event.summary || 'Không có summary.'}</p>{event.details ? <CodeBlock code={eventDetailsText(event)} language="event" /> : null}{event.links?.length ? <div className="inspector-link-row">{event.links.map((link) => <a key={`${link.href}-${link.label}`} href={link.href} target="_blank" rel="noreferrer"><Link2 size={11} aria-hidden="true" />{link.label}</a>)}</div> : null}</div></details>)}</section></div>
}

function SelectionActions({ text, onAskSelection, onClear }: { text: string; onAskSelection?: (text: string) => void; onClear: () => void }) {
  if (!onAskSelection || !text) return null
  const preview = text.length > 150 ? `${text.slice(0, 150).trim()}…` : text
  return <div className="inspector-selection-actions" role="toolbar" aria-label="Actions for selected research text"><MessageSquareQuote size={15} aria-hidden="true" /><span title={text}>“{preview}”</span><button type="button" onClick={() => { onAskSelection(text); onClear() }}>Hỏi về đoạn chọn</button><button type="button" className="inspector-selection-actions__clear" onClick={onClear} aria-label="Bỏ chọn đoạn văn"><X size={14} aria-hidden="true" /></button></div>
}

export function InspectorWorkspace({ snapshot, events, onAskSelection }: InspectorWorkspaceProps) {
  const [activeTab, setActiveTab] = useState<InspectorTab>('run')
  useEffect(() => {
    const openTab = (event: Event) => {
      const tab = (event as CustomEvent<string>).detail
      if (TABS.some((item) => item.id === tab)) setActiveTab(tab as InspectorTab)
    }
    window.addEventListener('research-scout:workspace', openTab)
    return () => window.removeEventListener('research-scout:workspace', openTab)
  }, [])
  const [selectedText, setSelectedText] = useState('')
  const workspaceRef = useRef<HTMLElement>(null)
  const result = resultFrom(snapshot)
  const papers = result?.papers ?? []
  const pmrlCount = papers.filter((paper) => Boolean(readPaperNotes(paper))).length

  const tabCount = (tab: InspectorTab): number | undefined => {
    if (tab === 'papers') return papers.length
    if (tab === 'pmrl') return pmrlCount
    if (tab === 'compare') return result?.benchmark ? 1 : 0
    if (tab === 'report') return readReportMarkdown(result) ? 1 : 0
    if (tab === 'trace') return events.length
    return undefined
  }

  const captureSelection = (event: MouseEvent<HTMLElement> | KeyboardEvent<HTMLElement>) => {
    if (!onAskSelection || !workspaceRef.current) return
    const selection = window.getSelection()
    const anchor = selection?.anchorNode
    const anchorElement = anchor instanceof Element ? anchor : anchor?.parentElement
    if (!selection || !anchorElement || !workspaceRef.current.contains(anchorElement) || !anchorElement.closest('.inspector-selectable')) return
    const value = selection.toString().trim()
    setSelectedText(value.length >= 2 ? value : '')
    void event
  }

  const handleTabKey = (event: KeyboardEvent<HTMLButtonElement>, tab: InspectorTab) => {
    const currentIndex = TABS.findIndex((item) => item.id === tab)
    let nextIndex: number | undefined
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') nextIndex = (currentIndex + 1) % TABS.length
    if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') nextIndex = (currentIndex - 1 + TABS.length) % TABS.length
    if (event.key === 'Home') nextIndex = 0
    if (event.key === 'End') nextIndex = TABS.length - 1
    if (nextIndex == null) return
    event.preventDefault()
    const next = TABS[nextIndex]
    setActiveTab(next.id)
    document.getElementById(`inspector-tab-${next.id}`)?.focus()
  }

  return <section className="inspector-workspace" ref={workspaceRef} onMouseUp={captureSelection} onKeyUp={captureSelection} aria-label="Research workspace">
    <header className="inspector-workspace__header"><div className="inspector-workspace__title"><span className="inspector-workspace__glyph"><PanelRight size={15} aria-hidden="true" /></span><div><span className="inspector-eyebrow">Research workspace</span><h2>{snapshot?.input.query || 'Phiên nghiên cứu'}</h2></div></div>{snapshot ? <StatusPill status={snapshot.status} /> : <span className="inspector-muted">No active run</span>}</header>
    <div className="inspector-tabs" role="tablist" aria-label="Research artifacts and run details">{TABS.map(({ id, label, icon: Icon }) => <button type="button" key={id} id={`inspector-tab-${id}`} role="tab" tabIndex={activeTab === id ? 0 : -1} aria-selected={activeTab === id} aria-controls={`inspector-panel-${id}`} className={activeTab === id ? 'is-active' : ''} onClick={() => setActiveTab(id)} onKeyDown={(event) => handleTabKey(event, id)}><Icon size={14} aria-hidden="true" /><span>{label}</span>{tabCount(id) ? <b>{tabCount(id)}</b> : null}</button>)}</div>
    <div className="inspector-workspace__body">{activeTab === 'run' ? <div id="inspector-panel-run" role="tabpanel" aria-labelledby="inspector-tab-run" tabIndex={0}><RunPanel snapshot={snapshot} events={events} /></div> : null}{activeTab === 'papers' ? <div id="inspector-panel-papers" role="tabpanel" aria-labelledby="inspector-tab-papers" tabIndex={0}><PapersPanel result={result} /></div> : null}{activeTab === 'pmrl' ? <div id="inspector-panel-pmrl" role="tabpanel" aria-labelledby="inspector-tab-pmrl" tabIndex={0}><PMRLPanel result={result} /></div> : null}{activeTab === 'compare' ? <div id="inspector-panel-compare" role="tabpanel" aria-labelledby="inspector-tab-compare" tabIndex={0}><ComparePanel result={result} /></div> : null}{activeTab === 'report' ? <div id="inspector-panel-report" role="tabpanel" aria-labelledby="inspector-tab-report" tabIndex={0}><ReportPanel snapshot={snapshot} result={result} /></div> : null}{activeTab === 'trace' ? <div id="inspector-panel-trace" role="tabpanel" aria-labelledby="inspector-tab-trace" tabIndex={0}><TracePanel events={events} /></div> : null}</div>
    <SelectionActions text={selectedText} onAskSelection={onAskSelection} onClear={() => setSelectedText('')} />
  </section>
}
