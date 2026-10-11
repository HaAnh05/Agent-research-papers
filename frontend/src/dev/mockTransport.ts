// Dev-only: replays recorded-style SSE sequences so the UI can be reviewed without the backend.
// Enabled by `?mock=1` on the vite dev server (or a build with VITE_MOCK=1). See src/lib/transport.ts.
import { ApiError, type RunFrame, type Transport } from '../lib/api'
import { STAGES } from '../lib/run'
import type { AnswerSummary, LlmCall, RunResult, RunSnapshot, TraceEvent, TraceFacts } from '../types'
import {
  CHATCOT_BIBTEX,
  CHATCOT_ID,
  CHATCOT_SUMMARY,
  MOCK_REPORT,
  MULTI_BIBTEX,
  MULTI_IDS,
  MULTI_SUMMARY,
  QUICK_ANSWER,
  REFLECT_BIBTEX,
  REFLECT_ID,
  REFLECT_SUMMARY,
  TABLE_ANSWER,
  mockGraph,
} from './mockData'

type Draft = Omit<TraceEvent, 'runId' | 'seq' | 'timestamp'>
interface Scheduled {
  at: number
  ev: Draft
}
interface MockRun {
  failed?: boolean
  id: string
  startedAt: number
  script: Scheduled[]
  result: RunResult
}

const slow = (() => {
  const raw = typeof location !== 'undefined' ? Number(new URLSearchParams(location.search).get('slow')) : NaN
  return Number.isFinite(raw) && raw > 0 ? raw : 1
})()

const runs = new Map<string, MockRun>()
let runCounter = 0
let hasFullRun = false

const stageDef = (node: string) => STAGES.find((s) => s.node === node)

function step(node: string, label: string, from: number, to: number, facts: TraceFacts, durationMs: number): Scheduled[] {
  const def = stageDef(node)
  const base = {
    node,
    label,
    kind: 'workflow',
    description: def?.description ?? null,
    stage: def?.stage ?? null,
    parallelGroup: def && 'parallelGroup' in def ? def.parallelGroup : null,
  }
  return [
    { at: from, ev: { ...base, type: 'step.started', status: 'running', summary: '' } },
    { at: to, ev: { ...base, type: 'step.completed', status: 'completed', summary: '', facts, durationMs } },
  ]
}

/** The summary as `summarize` makes it: no prior works, no graph size and no report id yet. */
function earlySummary(final: AnswerSummary): AnswerSummary {
  return {
    ...final,
    blocks: (final.blocks ?? []).filter((b) => b.type !== 'foundations'),
    actions: { ...final.actions, reportId: null, graphNodes: 0 },
  }
}

/**
 * One full run, in the order the real backend sends it:
 * write_notes -> summarize (answer.summary, report button grey) runs beside citation_graph and web_enrich;
 * the graph arrives after the summary (second answer.summary with prior works and the graph);
 * final_report then writes report.md; run.completed carries the final summary (report button black).
 */
function fullScript(multi: boolean, result: RunResult): Scheduled[] {
  const s: Scheduled[] = []
  s.push(...step('router', 'Phân tích yêu cầu', 0, 300, { intent: multi ? 'search' : 'direct_read' }, 3000))
  s[0].ev.description = 'Xác định đầu vào là bài báo hay chủ đề'
  s[1].ev.description = 'Xác định đầu vào là bài báo hay chủ đề'
  let t = 300
  if (multi) {
    const search = step('search_papers', 'Tìm bài trên arXiv', t, t + 700, { resultCount: 40 }, 2000)
    const evalS = step('eval_search', 'Đánh giá độ liên quan', t + 700, t + 1300, { selectedCount: 5 }, 9000)
    search.concat(evalS).forEach((x) => (x.ev.description = x.ev.node === 'search_papers' ? 'Tìm ứng viên trên arXiv' : 'Chọn bài phù hợp nhất'))
    s.push(...search, ...evalS)
    t += 1300
  }
  s.push(...step('read_paper', 'Đọc toàn văn', t, t + 1500, { paperCount: multi ? 5 : 1, tableCount: multi ? 61 : 17, referenceCount: multi ? 142 : 29, refsTotal: multi ? 142 : 29, refsResolved: multi ? 131 : 27, source: 'ar5iv' }, multi ? 22000 : 5000))
  t += 1600
  const calls = (kind: string, n: number, ms: number, tokens: [number, number, number]): LlmCall[] =>
    Array.from({ length: n }, (_, i) => ({ kind, paperId: multi ? `paper-${i + 1}` : undefined, ms: ms - i * 4000, inputTokens: tokens[0], outputTokens: tokens[1], reasoningTokens: tokens[2] }))
  const n = multi ? 5 : 1
  const final = result.answerSummary as AnswerSummary
  const early = earlySummary(final)
  const answer = (summary: AnswerSummary, reportReady: boolean, extra: Partial<RunResult> = {}): Partial<Draft> => ({
    node: 'summarize',
    label: 'Tóm tắt nhanh',
    status: reportReady ? 'completed' : 'running',
    summary: '',
    answerSummary: summary,
    reportReady,
    bibtex: result.bibtex,
    citationGraph: extra.citationGraph ?? null,
  })
  const tNotes = t + 3600
  const tSummary = tNotes + 1500
  const tGraph = tSummary + 3000
  const tReport = tGraph + 2500
  s.push(...step('write_notes', 'Ghi chú có cấu trúc', t, tNotes, { insightCount: multi ? 14 : 3, evalSettings: { proposed: multi ? 19 : 4, kept: multi ? 19 : 4, dropped: 0 }, llmCalls: calls('pmrl', n, 96000, [9400, 3100, 9400]) }, 96000))
  s.push(...step('web_enrich', 'Tìm mã nguồn và BibTeX', t, t + 1200, { repositoryCount: 0, bibtexCount: multi ? 5 : 1 }, 1000))
  s.push(...step('summarize', 'Tóm tắt nhanh', tNotes, tSummary, { keyResultCount: multi ? 3 : 3, resultRowCount: multi ? 0 : 4, llmCalls: calls('overview', 1, 74000, [5200, 1800, 6100]).slice(0, 1) }, 74000))
  // The quick summary and its lead go out right when summarize completes.
  const lead = early.lead ?? ''
  s.push({ at: tSummary - 10, ev: { type: 'assistant.delta', node: 'summarize', label: 'Tóm tắt nhanh', delta: lead } })
  s.push({ at: tSummary - 5, ev: { type: 'answer.summary', ...answer(early, false) } as Draft })
  s.push(...step('citation_graph', 'Dựng bản đồ liên quan', t, tGraph, { seeds: 22, pool: 300, nodeCount: 40, graphStatus: 'ok' }, 91000))
  s.push({ at: tGraph + 5, ev: { type: 'answer.summary', ...answer(final, false, { citationGraph: result.citationGraph }), } as Draft })
  s.push(...step('final_report', 'Viết báo cáo đầy đủ', tGraph + 10, tReport, { reportAvailable: true, sectionCount: 5, llmCalls: calls('section', n, 420000, [11800, 5200, 15300]) }, 420000))
  s.push({ at: tReport + 100, ev: { type: 'assistant.completed', node: 'final_report', label: 'Hoàn tất', status: 'completed' } })
  s.push({ at: tReport + 110, ev: { type: 'run.completed', node: 'final_report', label: 'Hoàn tất', status: 'completed', summary: '', answerSummary: final, reportReady: true, bibtex: result.bibtex, citationGraph: result.citationGraph ?? null } })
  return s
}

function quickScript(text: string, question: string): Scheduled[] {
  const s: Scheduled[] = step('router', 'Phân tích yêu cầu', 0, 300, { intent: 'followup' }, 1000)
  s.forEach((x) => (x.ev.description = 'Xác định đầu vào là bài báo hay chủ đề'))
  const sources = /table/i.test(question) ? ['Table 17'] : ['Mục Insight', 'Table 15', 'Table 16']
  const answerStep = step('answer_followup', 'Trả lời từ báo cáo đã có', 400, 400 + text.length * 3, { sources }, 6000)
  answerStep.forEach((x) => (x.ev.description = 'Trả lời câu hỏi tiếp theo bằng báo cáo và ghi chú đã có'))
  s.push(answerStep[0])
  let at = 600
  for (let i = 0; i < text.length; i += 14) {
    s.push({ at, ev: { type: 'assistant.delta', node: 'followup', label: 'Trả lời', delta: text.slice(i, i + 14) } })
    at += 35
  }
  s.push({ ...answerStep[1], at: at + 20 })
  s.push({ at: at + 50, ev: { type: 'assistant.completed', node: 'followup', label: 'Trả lời', status: 'completed' } })
  s.push({ at: at + 100, ev: { type: 'run.completed', node: 'followup', label: 'Hoàn tất', status: 'completed' } })
  return s
}

function buildRun(text: string, arxivField?: string): MockRun {
  const id = `mock-run-${++runCounter}`
  const arxiv = arxivField ?? text.match(/(\d{4}\.\d{4,5})/)?.[1]
  const startedAt = Date.now()
  const reportBase: RunResult = {
    report: MOCK_REPORT,
    answer: null,
  }

  if (/\b(fail|lỗi)\b/i.test(text)) {
    const script = [...step('router', 'Phân tích yêu cầu', 0, 300, { intent: 'direct_read' }, 1000), ...step('read_paper', 'Đọc toàn văn', 300, 900, {}, 1000).slice(0, 1)]
    script.push({ at: 1500, ev: { type: 'run.failed', node: 'read_paper', label: 'Đọc toàn văn', status: 'error', summary: 'Không tải được bản HTML hoặc PDF của bài này.' } })
    return { id, startedAt, script, result: {}, failed: true }
  }
  if (!arxiv && hasFullRun) {
    const answer = /table/i.test(text) ? TABLE_ANSWER : QUICK_ANSWER
    return { id, startedAt, script: quickScript(answer, text), result: { answer, answerSummary: null, citationGraph: null } }
  }
  hasFullRun = true

  const full = (multi: boolean, result: RunResult): MockRun => {
    const done: RunResult = { ...result, reportReady: true }
    return { id, startedAt, script: fullScript(multi, done), result: done }
  }
  if (!arxiv) {
    return full(true, { ...reportBase, reportId: MULTI_SUMMARY.actions?.reportId, bibtex: MULTI_BIBTEX, answerSummary: MULTI_SUMMARY, citationGraph: { similarity: mockGraph(MULTI_IDS) } })
  }
  if (arxiv === REFLECT_ID) {
    return full(false, { ...reportBase, papers: [{ paperId: 'arxiv:2605.05737', arxivId: REFLECT_ID, repository: { url: 'https://github.com/example/reflect', source: 'github_search' } }], reportId: REFLECT_SUMMARY.actions?.reportId, bibtex: REFLECT_BIBTEX, answerSummary: REFLECT_SUMMARY, citationGraph: { similarity: mockGraph([REFLECT_ID]) } })
  }
  // Any other arXiv id (for example "Phân tích bài này" on a graph node): a short single-paper answer.
  const graph = mockGraph([arxiv])
  const node = graph.nodes.find((n) => n.arxivId === arxiv)
  const summary = {
    ...CHATCOT_SUMMARY,
    paper: { ...CHATCOT_SUMMARY.paper, arxivId: arxiv, title: node?.title ?? CHATCOT_SUMMARY.paper?.title, published: node?.year ? String(node.year) : undefined },
    lead: arxiv === CHATCOT_ID ? CHATCOT_SUMMARY.lead : `${node?.title ?? 'Bài này'} là một bài trên bản đồ liên quan; bản tóm tắt dưới đây là dữ liệu mẫu.`,
  }
  return full(false, { ...reportBase, reportId: CHATCOT_SUMMARY.actions?.reportId, bibtex: CHATCOT_BIBTEX, answerSummary: summary, citationGraph: { similarity: graph } })
}

const scaled = (ms: number) => ms * slow
const toEvents = (run: MockRun): TraceEvent[] =>
  run.script
    .map((item, i) => ({ item, i }))
    .sort((a, b) => a.item.at - b.item.at || a.i - b.i)
    .map(({ item }, k) => ({ ...item.ev, runId: run.id, seq: k + 1, timestamp: new Date(run.startedAt + scaled(item.at)).toISOString() }) as TraceEvent)

const atOf = (run: MockRun, seq: number) => scaled(run.script.slice().sort((a, b) => a.at - b.at)[seq - 1].at)

function snapshot(run: MockRun): RunSnapshot {
  const elapsed = Date.now() - run.startedAt
  const all = toEvents(run)
  const due = all.filter((e) => atOf(run, e.seq) <= elapsed)
  const done = due.length === all.length
  const answer = done ? run.result.answer ?? null : null
  // A running snapshot carries the partial result of the last answer.summary that is due.
  const lastAnswer = [...due].reverse().find((e) => e.type === 'answer.summary')
  const partial: RunResult | undefined = lastAnswer
    ? { answerSummary: lastAnswer.answerSummary, citationGraph: lastAnswer.citationGraph, bibtex: lastAnswer.bibtex, reportReady: false, papers: run.result.papers }
    : undefined
  return {
    runId: run.id,
    threadId: 'mock-thread',
    status: done ? (run.failed ? 'error' : 'success') : 'running',
    answer,
    seq: due.length ? due[due.length - 1].seq : 0,
    traceEvents: due,
    error: done && run.failed ? 'Không tải được bản HTML hoặc PDF của bài này.' : null,
    result: done ? run.result : partial,
    startedAt: new Date(run.startedAt).toISOString(),
    updatedAt: new Date().toISOString(),
  }
}

export const mockTransport: Transport = {
  async createThread() {
    return 'mock-thread'
  },

  async sendMessage(_threadId, text, _file, arxivId) {
    const active = [...runs.values()].some((r) => snapshot(r).status === 'running')
    if (active) throw new ApiError('A research workflow is already running', 409)
    const run = buildRun(text, arxivId)
    runs.set(run.id, run)
    return snapshot(run)
  },

  async getRun(runId) {
    const run = runs.get(runId)
    if (!run) throw new ApiError('Run not found', 404)
    return snapshot(run)
  },

  subscribe(runId, afterSeq, onFrame: (frame: RunFrame) => void) {
    const run = runs.get(runId)
    if (!run) return () => undefined
    const snap = snapshot(run)
    const timers: number[] = []
    queueMicrotask(() => onFrame({ type: 'snapshot', data: snap }))
    const cursor = Math.max(afterSeq, snap.seq)
    const elapsed = Date.now() - run.startedAt
    toEvents(run)
      .filter((e) => e.seq > cursor)
      .forEach((e) => {
        timers.push(window.setTimeout(() => onFrame({ type: e.type, data: e }), Math.max(0, atOf(run, e.seq) - elapsed)))
      })
    return () => timers.forEach((id) => window.clearTimeout(id))
  },

  async downloadReport() {
    return new Blob([MOCK_REPORT], { type: 'text/markdown' })
  },
}
