import type { AnswerSummary, GraphNode, RunSnapshot, RunView, SimilarityGraph, StepView, SummaryBlock, TraceEvent } from '../types'

/**
 * The six stages of a paper analysis; used to show waiting rows before their events arrive.
 * Stages 2-5 run at the same time: notes then the quick summary form one chain, beside the graph and the BibTeX lookup.
 */
export const STAGES = [
  { stage: 1, node: 'read_paper', label: 'Đọc toàn văn', description: 'Trích xuất bảng và tài liệu tham khảo' },
  { stage: 2, node: 'write_notes', label: 'Ghi chú có cấu trúc', description: 'Rút vấn đề, phương pháp, kết quả, hạn chế', parallelGroup: 'analysis' },
  { stage: 3, node: 'summarize', label: 'Tóm tắt nhanh', description: 'Viết câu trả lời ngắn từ ghi chú, không chờ bản đồ', parallelGroup: 'analysis' },
  { stage: 4, node: 'citation_graph', label: 'Dựng bản đồ liên quan', description: 'Tìm bài nền tảng và bài kế thừa', parallelGroup: 'analysis' },
  { stage: 5, node: 'web_enrich', label: 'Tìm mã nguồn và BibTeX', description: 'Tìm repository chính thức và tạo BibTeX', parallelGroup: 'analysis' },
  { stage: 6, node: 'final_report', label: 'Viết báo cáo đầy đủ', description: 'Viết các mục và ghép report.md, chạy nền' },
] as const

export function emptyRun(runId: string): RunView {
  return { runId, status: 'running', steps: [], answerText: '', lastSeq: 0, reportReady: false }
}

const parseMs = (iso?: string | null): number | undefined => {
  if (!iso) return undefined
  const ms = Date.parse(iso)
  return Number.isFinite(ms) ? ms : undefined
}

function upsertStep(steps: StepView[], ev: TraceEvent, startedMs?: number): StepView[] {
  const index = steps.findIndex((s) => s.node === ev.node)
  const previous = index >= 0 ? steps[index] : undefined
  let status = previous?.status ?? 'running'
  if (ev.type === 'step.completed') {
    status = ev.status === 'error' || ev.status === 'failed' ? 'error' : ev.status === 'skipped' ? 'skipped' : 'completed'
  } else if (ev.type === 'step.started' && status !== 'completed') {
    status = 'running'
  }
  const next: StepView = {
    key: ev.node,
    node: ev.node,
    label: ev.label || previous?.label || ev.node,
    description: ev.description ?? previous?.description,
    stage: ev.stage ?? previous?.stage,
    parallelGroup: ev.parallelGroup ?? previous?.parallelGroup,
    status,
    facts: { ...(previous?.facts ?? {}), ...(ev.facts ?? {}) },
    durationMs: ev.durationMs ?? previous?.durationMs,
    // The timer counts from the first event of the step (normally step.started).
    startedMs: previous?.startedMs ?? startedMs,
  }
  if (index < 0) return [...steps, next]
  const copy = steps.slice()
  copy[index] = next
  return copy
}

/** The partial or final result carried by `answer.summary` and `run.completed`. */
function applyAnswer(run: RunView, ev: TraceEvent): Partial<RunView> {
  if (!ev.answerSummary && !ev.citationGraph && ev.reportReady == null) return {}
  const summary = normalizeSummary(ev.answerSummary) ?? run.summary
  const incoming = normalizeGraph(ev.citationGraph?.similarity)
  // Keep the previous graph object when the same graph arrives again, so the panel does not re-layout.
  const graph = incoming && run.graph && incoming.nodes.length === run.graph.nodes.length ? run.graph : incoming ?? run.graph
  const reportReady = ev.reportReady ?? run.reportReady
  return {
    summary,
    graph,
    reportReady,
    result: {
      ...run.result,
      answerSummary: ev.answerSummary ?? run.result?.answerSummary,
      citationGraph: ev.citationGraph ?? run.result?.citationGraph,
      bibtex: ev.bibtex ?? run.result?.bibtex,
      reportReady,
    },
  }
}

/**
 * Idempotent: an event whose seq is not newer than the last applied one is dropped.
 * `receivedMs` is the client time a live event arrived; the step and run timers count from it.
 * A replayed event (snapshot) has none and falls back to its server timestamp.
 */
export function applyEvent(run: RunView, ev: TraceEvent, receivedMs?: number): RunView {
  if (ev.seq > 0 && ev.seq <= run.lastSeq) return run
  const at = receivedMs ?? parseMs(ev.timestamp)
  const next: RunView = { ...run, lastSeq: Math.max(run.lastSeq, ev.seq), startedMs: run.startedMs ?? at }
  switch (ev.type) {
    case 'step.started':
    case 'step.updated':
    case 'step.completed':
      next.steps = upsertStep(run.steps, ev, at)
      break
    case 'assistant.delta':
      next.answerText = run.answerText + (ev.delta ?? '')
      break
    case 'answer.summary':
      Object.assign(next, applyAnswer(run, ev))
      break
    case 'run.completed':
      Object.assign(next, applyAnswer(run, ev))
      next.status = 'success'
      next.endedMs = at ?? run.endedMs
      next.steps = run.steps.map((s) => (s.status === 'running' ? { ...s, status: 'completed' } : s))
      break
    case 'run.failed':
      next.status = 'error'
      next.endedMs = at ?? run.endedMs
      next.error = ev.summary || ev.details || next.error || 'Run thất bại.'
      next.steps = run.steps.map((s) => (s.status === 'running' ? { ...s, status: 'error' } : s))
      break
    default:
      break
  }
  return next
}

/** Rebuild a view from a snapshot (first SSE frame, reconnect, or the final GET). */
export function runFromSnapshot(snap: RunSnapshot): RunView {
  let run = emptyRun(snap.runId)
  const events = snap.traceEvents.slice().sort((a, b) => a.seq - b.seq)
  for (const ev of events) run = applyEvent(run, ev)
  run = { ...run, lastSeq: Math.max(run.lastSeq, snap.seq) }
  run.startedMs = run.startedMs ?? parseMs(snap.startedAt)
  if (snap.status && snap.status !== 'idle') run.status = snap.status
  const terminal = run.status !== 'running'
  if (terminal) {
    run.endedMs = run.endedMs ?? parseMs(snap.updatedAt)
    if (run.status === 'error') run.error = run.error ?? snap.error ?? 'Run thất bại.'
  }
  const result = snap.result
  const answer = snap.answer ?? result?.answer
  if (typeof answer === 'string' && answer) run.answerText = answer
  if (result) {
    run.result = result
    run.summary = normalizeSummary(result.answerSummary) ?? undefined
    run.graph = normalizeGraph(result.citationGraph?.similarity) ?? undefined
    // An older backend has no reportReady: a report id then means the report exists.
    run.reportReady = result.reportReady ?? !!result.reportId
  }
  return run
}

// ---- derived state -----------------------------------------------------------------

/** A follow-up answered from the existing report: the router chose `followup` and the run has an `answer_followup` step. */
export function isQuick(run: RunView): boolean {
  return run.steps.some((s) => s.node === 'answer_followup' || (s.node === 'router' && s.facts.intent === 'followup'))
}

export function followupSources(run: RunView): string[] {
  const sources = run.steps.find((s) => s.node === 'answer_followup')?.facts.sources
  return Array.isArray(sources) ? sources.filter((x): x is string => typeof x === 'string') : []
}

/** Rows to show: seen steps without a stage (router, search, ...) then stages 1-5, waiting ones included while running. */
export function displaySteps(run: RunView): StepView[] {
  // The router row only helps in topic runs (search); for a direct paper run it is noise.
  const seen = run.steps.filter((s) => s.stage == null && (s.node !== 'router' || s.facts.intent === 'search'))
  const staged = new Map(run.steps.filter((s) => s.stage != null).map((s) => [s.stage as number, s]))
  const full = run.steps.some((s) => s.node !== 'router' && s.node !== 'error_handler')
  const rows: StepView[] = [...seen]
  for (const def of STAGES) {
    const step = staged.get(def.stage)
    if (step) rows.push(step)
    else if (full && run.status === 'running') {
      rows.push({
        key: def.node,
        node: def.node,
        label: def.label,
        description: def.description,
        stage: def.stage,
        parallelGroup: 'parallelGroup' in def ? def.parallelGroup : undefined,
        status: 'wait',
        facts: {},
      })
    }
  }
  return rows
}

export function runSeconds(run: RunView): number | undefined {
  const sec = run.summary?.durationSec
  if (typeof sec === 'number' && sec >= 0) return sec
  if (run.startedMs != null && run.endedMs != null) return Math.max(0, Math.round((run.endedMs - run.startedMs) / 1000))
  return undefined
}

// ---- tolerant normalizers ---------------------------------------------------------

const isRecord = (v: unknown): v is Record<string, unknown> => v !== null && typeof v === 'object' && !Array.isArray(v)
const str = (v: unknown): string | undefined => (typeof v === 'string' && v.trim() ? v : undefined)
const num = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined)
const pick = (r: Record<string, unknown>, ...keys: string[]) => keys.map((k) => r[k]).find((v) => v !== undefined && v !== null)

export function normalizeSummary(raw: unknown): AnswerSummary | null {
  if (!isRecord(raw)) return null
  const blocks = Array.isArray(raw.blocks)
    ? (raw.blocks.filter((b) => isRecord(b) && typeof b.type === 'string') as SummaryBlock[])
    : []
  const summary = raw as AnswerSummary
  return { ...summary, lead: summary.lead || summary.tldr, blocks }
}

function normalizeNode(raw: unknown): GraphNode | null {
  if (!isRecord(raw)) return null
  const title = str(raw.title)
  if (!title) return null
  const arxiv = str(pick(raw, 'arxivId', 'arxiv_id'))
  return {
    id: str(raw.id) ?? arxiv ?? title,
    arxivId: arxiv ?? null,
    title,
    year: num(raw.year) ?? null,
    citedByCount: num(pick(raw, 'citedByCount', 'cited_by_count')) ?? null,
    firstAuthor: str(pick(raw, 'firstAuthor', 'first_author')) ?? null,
    abstract: str(raw.abstract) ?? null,
    similarity: num(raw.similarity) ?? null,
    role: str(raw.role),
    url: str(raw.url) ?? null,
    support: num(raw.support) ?? null,
  }
}

/** Accepts camelCase (API) and snake_case (state channel). Returns null when there is nothing to draw. */
export function normalizeGraph(raw: unknown): SimilarityGraph | null {
  if (!isRecord(raw)) return null
  const list = (v: unknown): GraphNode[] =>
    Array.isArray(v) ? (v.map(normalizeNode).filter(Boolean) as GraphNode[]) : []
  const nodes = list(raw.nodes)
  if (!nodes.length) return null
  const ids = new Set(nodes.map((n) => n.id))
  const edges = (Array.isArray(raw.edges) ? raw.edges : [])
    .filter(isRecord)
    .map((e) => ({ source: String(e.source ?? ''), target: String(e.target ?? ''), weight: num(e.weight) ?? 0.1 }))
    .filter((e) => ids.has(e.source) && ids.has(e.target))
  return {
    nodes,
    edges,
    priorWorks: list(pick(raw, 'priorWorks', 'prior_works')),
    derivativeWorks: list(pick(raw, 'derivativeWorks', 'derivative_works')),
    stats: isRecord(raw.stats) ? raw.stats : undefined,
  }
}
