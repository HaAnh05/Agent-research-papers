import type { RunSnapshot, TraceEvent, TraceFacts } from '../types'

export type StageId = 'understand' | 'discover' | 'read' | 'enrich' | 'notes' | 'compare' | 'report'
export type StageStatus = 'pending' | 'running' | 'completed' | 'skipped' | 'failed'
export type RouteKind = 'unknown' | 'search' | 'direct_read' | 'direct_compare' | 'direct_answer'
export type ViewPhase = 'running' | 'failed' | 'completed' | 'no_research' | 'missing_report'

export interface UiStage {
  id: StageId
  label: string
  status: StageStatus
  detail?: string
}

export interface UiActivity {
  key: string
  timestamp: string
  text: string
}

export interface UiRun {
  phase: ViewPhase
  /** A terminal event has arrived, but the authoritative final snapshot is still being fetched. */
  finalizing: boolean
  route: RouteKind
  stages: UiStage[]
  activity: UiActivity[]
  counts: { candidates?: number; selected?: number; pdfsRead?: number; repositories?: number; bibtexEntries?: number }
  currentStage: StageId | null
  headline: string
  error: string | null
  reportReady: boolean
  comparisonReady: boolean
  warnings: string[]
  events: TraceEvent[]
}

const STAGE_LABELS: Record<StageId, string> = {
  understand: 'Understand request',
  discover: 'Discover relevant papers',
  read: 'Read source PDFs',
  enrich: 'Enrich sources',
  notes: 'Write structured notes',
  compare: 'Compare selected papers',
  report: 'Generate final report',
}

const STAGE_COPY: Record<StageId, string> = {
  understand: 'Understanding your request…',
  discover: 'Finding and ranking papers…',
  read: 'Reading source PDFs…',
  enrich: 'Checking related repositories and citations…',
  notes: 'Writing structured notes…',
  compare: 'Comparing selected papers…',
  report: 'Preparing the research report…',
}

const NODE_STAGE: Record<string, StageId | undefined> = {
  router: 'understand',
  search_papers: 'discover',
  eval_search: 'discover',
  refine_query: 'discover',
  read_paper: 'read',
  web_enrich: 'enrich',
  write_notes: 'notes',
  compare_benchmark: 'compare',
  final_report: 'report',
}

export function stageForNode(node: string | null | undefined): StageId | null {
  return node ? NODE_STAGE[node] ?? null : null
}

/** Snapshots include history; sequence is the only source of ordering truth. */
export function orderedRunEvents(snapshot: RunSnapshot, incoming: TraceEvent[] = []): TraceEvent[] {
  const bySeq = new Map<number, TraceEvent>()
  const snapshotSequences = new Set<number>()
  for (const event of snapshot.traceEvents) {
    if (event.runId !== snapshot.runId || !Number.isInteger(event.seq) || event.seq <= 0) continue
    snapshotSequences.add(event.seq)
    bySeq.set(event.seq, event)
  }
  for (const event of incoming) {
    if (event.runId !== snapshot.runId || !Number.isInteger(event.seq) || event.seq <= 0) continue
    // A snapshot is authoritative for every event it carries. In particular,
    // a stale SSE frame with the same sequence must never replace the
    // snapshot's route/count/status evidence. Frames newer than the snapshot
    // cursor remain eligible while the stream is live.
    if (snapshotSequences.has(event.seq)) continue
    if (event.seq <= (snapshot.seq ?? 0)) continue
    bySeq.set(event.seq, event)
  }
  return [...bySeq.values()].sort((a, b) => a.seq - b.seq)
}

function factsOf(event: TraceEvent): TraceFacts | null {
  return event.type === 'step.updated' ? event.facts ?? null : null
}

function routeFromDetails(event: TraceEvent): RouteKind | null {
  if (event.node !== 'router' || typeof event.details !== 'string') return null
  const intent = event.details.match(/(?:^| · )intent: (search|direct_read|direct_compare|direct_answer)(?: · |$)/)?.[1]
  return (intent as RouteKind | undefined) ?? null
}

function validBenchmark(value: string | null | undefined): boolean {
  const text = value?.trim() ?? ''
  return Boolean(text) && !/^\*?\(?unable to|^\*?\(?không thể|^error\b/i.test(text)
}

const DEFAULT_FAILURE_COPY: Record<StageId, string> = {
  understand: 'Could not understand the request. Check the request and start again.',
  discover: 'Could not find suitable papers. Edit the request or start again.',
  read: 'Could not read source PDFs. Check the source files or network, then start again.',
  enrich: 'Could not enrich sources. Related repositories or citations may be unavailable. Start again.',
  notes: 'Could not write structured notes. The language-model provider did not respond. Start again.',
  compare: 'Could not compare selected papers. The language-model provider did not respond. Start again.',
  report: 'Could not prepare the research report. The language-model provider did not respond. Start again.',
}

function failureStage(lastFailureNode: string | null, snapshot: RunSnapshot, events: TraceEvent[]): StageId {
  const directNode = lastFailureNode === 'error_handler' ? snapshot.currentNode : lastFailureNode
  return stageForNode(directNode) ??
    stageForNode([...events].reverse().find((event) => event.node !== 'error_handler' && stageForNode(event.node))?.node) ??
    'understand'
}

/**
 * The API normally provides stable public English failure copy. A few graph
 * guards still return Vietnamese or provider exception wording; keep those
 * details out of the English UI while retaining short, concrete fixture/API
 * messages when they are already safe to display.
 */
function publicFailureCopy(rawMessage: string | null | undefined, stage: StageId): string {
  const message = rawMessage?.trim() ?? ''
  if (/arxiv\s+không trả về bài báo nào|không có bài báo nào trong danh sách được chọn|no papers met the selection criteria/i.test(message)) {
    return 'No papers met the selection criteria. Try a more specific topic or enter an arXiv ID directly, then start again.'
  }
  if (/tất cả các bài báo đều gặp lỗi khi đọc\/tải pdf|all (?:selected|supplied) papers? (?:failed|could not).*pdf|all source pdfs failed/i.test(message)) {
    return 'Could not read source PDFs. All supplied sources failed to load. Check the source files or network, then start again.'
  }
  if (/error generating final report|final report (?:is )?unavailable/i.test(message)) {
    return DEFAULT_FAILURE_COPY.report
  }
  if (/benchmark comparison unavailable|(?:could not|unable to|failed to) generate benchmark/i.test(message)) {
    return DEFAULT_FAILURE_COPY.compare
  }
  if (/direct answer provider unavailable|could not answer this request/i.test(message)) {
    return 'Could not answer this request. The language-model provider did not respond. Start again.'
  }

  // Vietnamese graph/provider messages are not suitable public copy. This
  // intentionally catches accented text after the deterministic guards above.
  if (/[À-ỹ]/u.test(message)) return DEFAULT_FAILURE_COPY[stage]
  if (!message) return DEFAULT_FAILURE_COPY[stage]

  // Avoid forwarding obvious stack traces, paths, credentials, or oversized
  // exception payloads. The API's short public messages and useful test/API
  // causes remain visible here.
  if (message.length > 240 || /traceback|stack trace|exception|(?:api[_-]?key|token|secret)\s*[=:]|(?:[A-Za-z]:\\|\/home\/|\/Users\/|\/tmp\/)/i.test(message)) {
    return DEFAULT_FAILURE_COPY[stage]
  }
  return message
}

function activityText(event: TraceEvent): string | null {
  const stage = stageForNode(event.node)
  const facts = factsOf(event)
  if (event.node === 'refine_query' && facts?.retryCount !== undefined) return `Search query refined · ${facts.retryCount} ${facts.retryCount === 1 ? 'time' : 'times'}`
  if (event.type === 'step.started' && stage) return `${STAGE_LABELS[stage]} started`
  if (event.type === 'step.updated' && facts) {
    if (event.node === 'router' && facts.intent) return `Request routed · ${facts.intent.replace('_', ' ')}`
    if (event.node === 'search_papers' && facts.resultCount !== undefined) return `Search returned ${facts.resultCount} ${facts.resultCount === 1 ? 'paper' : 'papers'}`
    if (event.node === 'eval_search' && facts.selectedCount !== undefined) return `Selected ${facts.selectedCount} ${facts.selectedCount === 1 ? 'paper' : 'papers'}`
    if (event.node === 'read_paper' && facts.paperCount !== undefined) return `Read ${facts.paperCount} ${facts.paperCount === 1 ? 'PDF' : 'PDFs'}`
    if (event.node === 'web_enrich') return 'Source enrichment updated'
    if (event.node === 'write_notes') return facts.noteFallbackPaperIds?.length ? `Structured notes unavailable for ${facts.noteFallbackPaperIds.length} paper(s)` : 'Structured notes finalized'
    if (event.node === 'compare_benchmark') return facts.comparisonAvailable ? 'Comparison finalized' : 'Comparison unavailable'
    if (event.node === 'final_report' && facts.reportAvailable) return 'Report artifact prepared'
  }
  if (event.type === 'step.completed' && stage) return `${STAGE_LABELS[stage]} task finished`
  if (event.type === 'run.completed') return 'Research run completed'
  if (event.type === 'run.failed') return `${stage ? STAGE_LABELS[stage] : 'Research run'} failed`
  return null
}

export function deriveUiRun(snapshot: RunSnapshot, incoming: TraceEvent[] = []): UiRun {
  const events = orderedRunEvents(snapshot, incoming)
  const seen = new Set<string>()
  const counts: UiRun['counts'] = {}
  const warnings: string[] = []
  let route: RouteKind = 'unknown'
  let comparisonFlag: boolean | undefined
  let compareSeen = false
  let notesFallbackCount = 0
  let lastReadCount: number | undefined
  let lastFailureNode: string | null = null
  let routerTime = snapshot.startedAt ?? snapshot.createdAt ?? ''
  let readTime = ''

  for (const event of events) {
    seen.add(event.node)
    if (event.node === 'router') routerTime = event.timestamp
    if (event.node === 'read_paper') readTime = event.timestamp
    if (event.node === 'compare_benchmark') compareSeen = true
    if (event.type === 'run.failed' || event.status === 'error' || event.status === 'failed') lastFailureNode = event.node
    const facts = factsOf(event)
    if (facts?.intent) route = facts.intent
    else if (route === 'unknown') route = routeFromDetails(event) ?? route
    if (facts?.resultCount !== undefined) counts.candidates = facts.resultCount
    if (facts?.selectedCount !== undefined) counts.selected = facts.selectedCount
    if (event.node === 'read_paper' && facts?.paperCount !== undefined) { counts.pdfsRead = facts.paperCount; lastReadCount = facts.paperCount }
    if (facts?.repositoryCount !== undefined) counts.repositories = facts.repositoryCount
    if (facts?.bibtexCount !== undefined) counts.bibtexEntries = facts.bibtexCount
    if (facts?.failedPdfCount) warnings.push(`${facts.failedPdfCount} PDF ${facts.failedPdfCount === 1 ? 'could not be read' : 'could not be read'}.`)
    if (facts?.noteFallbackPaperIds?.length) notesFallbackCount = facts.noteFallbackPaperIds.length
    if (event.node === 'compare_benchmark' && facts?.comparisonAvailable !== undefined) comparisonFlag = facts.comparisonAvailable
  }

  if (route === 'unknown') {
    if (seen.has('direct_answer')) route = 'direct_answer'
    else if (seen.has('search_papers') || seen.has('eval_search') || seen.has('refine_query')) route = 'search'
    else if (seen.has('read_paper')) route = 'direct_read'
  }
  if (notesFallbackCount) warnings.push(`Structured notes were unavailable for ${notesFallbackCount} paper(s).`)
  if (snapshot.status === 'degraded' && warnings.length === 0) warnings.push('The research completed with warnings; some upstream data may be missing.')

  const report = snapshot.result?.report ?? snapshot.result?.finalReport ?? snapshot.result?.final_report ?? ''
  const reportReady = Boolean(report.trim()) && !/^# Error Generating Final Report\b/i.test(report.trim())
  const benchmark = snapshot.result?.benchmark ?? snapshot.result?.benchmarkMatrix ?? snapshot.result?.benchmark_matrix
  const comparisonReady = comparisonFlag === true && validBenchmark(benchmark)
  const terminalSuccess = snapshot.status === 'success' || snapshot.status === 'degraded'
  const terminalEvent = [...events].reverse().find((event) => event.type === 'run.completed' || event.type === 'run.failed')
  const finalizing = snapshot.status === 'running' && terminalEvent?.type === 'run.completed'
  const phase: ViewPhase = snapshot.status === 'error' ? 'failed'
    : terminalSuccess && route === 'direct_answer' ? 'no_research'
    : terminalSuccess && !reportReady ? 'missing_report'
    : terminalSuccess && reportReady ? 'completed'
    : 'running'

  // A paper count alone does not prove that comparison was requested or that
  // the optional benchmark node ran. Keep the row only when routing or a real
  // compare trace says it belongs in this run.
  const showCompare = compareSeen || route === 'direct_compare'
  const ids: StageId[] = route === 'unknown' || route === 'direct_answer' ? ['understand']
    : route === 'search' ? ['understand', 'discover', 'read', 'enrich', 'notes', ...(showCompare ? ['compare' as const] : []), 'report']
    : ['understand', 'read', 'enrich', 'notes', ...(showCompare ? ['compare' as const] : []), 'report']
  const reached = new Set<StageId>()
  for (const event of events) {
    const id = stageForNode(event.node)
    if (id) reached.add(id)
  }
  const failedStage = phase === 'failed' ? failureStage(lastFailureNode, snapshot, events) : null
  let currentStage: StageId | null = null
  const stages = ids.map<UiStage>((id, index) => {
    const laterReached = ids.slice(index + 1).some((later) => reached.has(later))
    let status: StageStatus = 'pending'
    if (id === 'understand' && route !== 'unknown') status = 'completed'
    else if (reached.has(id)) status = laterReached ? 'completed' : 'running'
    else if (id === 'understand') status = 'running'
    if (id === 'compare' && comparisonFlag === false) status = 'failed'
    if (id === 'report' && phase === 'completed') status = 'completed'
    if (phase === 'failed' && id === failedStage) status = 'failed'
    if (phase === 'missing_report' && id === 'report') status = 'failed'
    // Terminal snapshots are authoritative. A final error or missing report
    // must never leave an earlier row looking live while the user is offered
    // recovery actions.
    if (phase !== 'running' && status === 'running') {
      if (phase === 'failed') {
        const failedIndex = failedStage ? ids.indexOf(failedStage) : -1
        status = failedIndex >= 0 && index < failedIndex ? 'completed' : 'pending'
      } else if (phase === 'missing_report' && id === 'report') {
        status = 'failed'
      } else {
        status = 'completed'
      }
    }
    if (status === 'running') currentStage = id
    return {
      id,
      label: id === 'read' && (route === 'direct_read' || route === 'direct_compare') ? 'Read supplied source(s)' : STAGE_LABELS[id],
      status,
      detail: id === 'discover' && events.some((item) => item.node === 'refine_query')
        ? (() => { const attempts = events.reduce((n, item) => Math.max(n, factsOf(item)?.retryCount ?? 0), 0); return attempts ? `Refined query · ${attempts} ${attempts === 1 ? 'time' : 'times'}` : undefined })()
        : id === 'notes' && notesFallbackCount ? `${notesFallbackCount} unavailable` : undefined,
    }
  })

  if (phase !== 'running') currentStage = null
  const current = currentStage ?? stages.find((stage) => stage.status === 'failed')?.id ?? null
  const failureAction = current === 'understand' ? 'understand the request'
    : current === 'discover' ? 'find suitable papers'
    : current === 'read' ? 'read source PDFs'
    : current === 'enrich' ? 'enrich sources'
    : current === 'notes' ? 'write structured notes'
    : current === 'compare' ? 'compare selected papers'
    : current === 'report' ? 'prepare the research report'
    : 'complete research'
  const headline = finalizing ? 'Finalizing research outputs…'
    : phase === 'no_research' ? 'No paper research was run for this request.'
    : phase === 'missing_report' ? 'The final report is unavailable.'
    : phase === 'failed' ? `Could not ${failureAction}.`
    : currentStage ? (currentStage === 'read' && route !== 'search' ? 'Reading supplied source(s)…' : STAGE_COPY[currentStage])
    : 'Starting research…'
  const error = phase === 'failed' ? publicFailureCopy(snapshot.error, failedStage ?? 'understand')
    : phase === 'missing_report' ? 'The workflow finished without a usable report. Start again.'
    : phase === 'no_research' ? 'Add a paper, PDF, or ask to find research papers, then start again.'
    : null

  const activity: UiActivity[] = []
  const activityGeneration = new Map<string, number>()
  const completionIndex = new Map<string, number>()
  const retryCountByNode = new Map<string, number>()
  for (const event of events) {
    const facts = factsOf(event)
    const retryCount = facts?.retryCount
    const previousRetry = retryCountByNode.get(event.node)
    const newAttempt = event.type === 'step.started' || event.status === 'retrying'
      || (retryCount !== undefined && previousRetry !== undefined && retryCount > previousRetry)
    if (newAttempt) activityGeneration.set(event.node, (activityGeneration.get(event.node) ?? 0) + 1)
    if (retryCount !== undefined) retryCountByNode.set(event.node, retryCount)

    const value = activityText(event)
    if (!value) continue
    const duration = typeof event.durationMs === 'number' && Number.isFinite(event.durationMs) && event.durationMs >= 0
      ? ` · ${event.durationMs >= 1000 ? `${(event.durationMs / 1000).toFixed(1)}s` : `${Math.round(event.durationMs)}ms`}`
      : ''
    const isStepCompletion = event.type === 'step.completed' || (event.type === 'step.updated' && event.status === 'completed')
    const completionKey = `${event.node}:${activityGeneration.get(event.node) ?? 0}`
    const earlierCompletion = isStepCompletion ? completionIndex.get(completionKey) : undefined
    if (earlierCompletion !== undefined) {
      if (duration) activity[earlierCompletion].text += duration
      continue
    }
    if (isStepCompletion) completionIndex.set(completionKey, activity.length)
    activity.push({ key: `${event.runId}:${event.seq}`, timestamp: event.timestamp, text: `${value}${duration}` })
  }
  if (route === 'direct_read' || route === 'direct_compare') activity.push({ key: 'discover-skipped', timestamp: routerTime, text: 'Discovery skipped · supplied source' })
  if (lastReadCount === 1 && !compareSeen && readTime) activity.push({ key: 'comparison-skipped', timestamp: readTime, text: 'Comparison skipped · single-paper task' })
  if (phase === 'failed') {
    const failedEvent = [...events].reverse().find((event) => event.type === 'run.failed')
    activity.push({
      key: `failure:${snapshot.runId}:${failedEvent?.seq ?? 'snapshot'}`,
      timestamp: failedEvent?.timestamp ?? snapshot.updatedAt ?? snapshot.createdAt ?? '',
      text: publicFailureCopy(snapshot.error, failedStage ?? 'understand'),
    })
  }
  activity.sort((a, b) => a.timestamp.localeCompare(b.timestamp) || a.key.localeCompare(b.key))

  return { phase, finalizing, route, stages, activity, counts, currentStage, headline, error, reportReady, comparisonReady, warnings, events }
}
