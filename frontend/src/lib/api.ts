import type {
  AppConfig,
  MessageAttachment,
  MessageRole,
  MessageStatus,
  NormalizedAssistantEvent,
  PaperSource,
  ResearchArtifact,
  ReportDocument,
  ReportSummary,
  RunInput,
  RunResult,
  RunSnapshot,
  RunStatus,
  SendMessageInput,
  ServerRunEvent,
  Thread,
  ThreadDetail,
  ThreadInput,
  ThreadMessage,
  ThreadRun,
  TraceEvent,
  TraceFacts,
  TraceEventType,
  TraceKind,
  TraceStatus,
} from '../types'

const API_BASE = (import.meta.env.VITE_API_BASE ?? '/api').replace(/\/$/, '')

export class ApiError extends Error {
  readonly status: number
  readonly body: unknown

  constructor(message: string, status: number, body?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' ? (value as Record<string, unknown>) : {}
}

function firstString(record: Record<string, unknown>, ...keys: string[]): string | undefined {
  for (const key of keys) {
    if (typeof record[key] === 'string') return record[key] as string
  }
  return undefined
}

function firstNumber(record: Record<string, unknown>, ...keys: string[]): number | undefined {
  for (const key of keys) {
    if (typeof record[key] === 'number' && Number.isFinite(record[key])) return record[key] as number
  }
  return undefined
}

function firstArray<T>(record: Record<string, unknown>, ...keys: string[]): T[] {
  for (const key of keys) {
    if (Array.isArray(record[key])) return record[key] as T[]
  }
  return []
}

function firstObject(record: Record<string, unknown>, ...keys: string[]): Record<string, unknown> {
  for (const key of keys) {
    const value = record[key]
    if (value !== null && typeof value === 'object' && !Array.isArray(value)) return value as Record<string, unknown>
  }
  return {}
}

function unwrapRecord(input: unknown, ...keys: string[]): Record<string, unknown> {
  const record = asRecord(input)
  for (const key of keys) {
    const nested = record[key]
    if (nested !== null && typeof nested === 'object' && !Array.isArray(nested)) return nested as Record<string, unknown>
  }
  return record
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
}

function normalizeMessageRole(value: unknown): MessageRole {
  return value === 'user' || value === 'assistant' || value === 'system' ? value : 'assistant'
}

function normalizeMessageStatus(value: unknown): MessageStatus | undefined {
  if (value === 'pending' || value === 'streaming' || value === 'completed' || value === 'error') return value
  if (value === 'complete' || value === 'success' || value === 'done') return 'completed'
  if (value === 'failed') return 'error'
  return undefined
}

function normalizeAttachment(input: unknown): MessageAttachment | null {
  if (typeof input === 'string' && input.trim()) return { name: input.trim() }
  const record = asRecord(input)
  const name = firstString(record, 'name', 'filename', 'fileName', 'file_name')
  if (!name) return null
  return {
    id: firstString(record, 'id', 'attachmentId', 'attachment_id'),
    name,
    mimeType: firstString(record, 'mimeType', 'mime_type', 'contentType', 'content_type'),
    mime_type: firstString(record, 'mime_type'),
    size: firstNumber(record, 'size', 'sizeBytes', 'size_bytes'),
    url: firstString(record, 'url', 'href') ?? null,
  }
}

function normalizeStatus(value: unknown): RunStatus {
  if (value === 'running') return 'running'
  if (value === 'queued' || value === 'pending' || value === 'started') return 'running'
  if (value === 'success' || value === 'completed' || value === 'complete') return 'success'
  if (value === 'degraded' || value === 'partial') return 'degraded'
  if (value === 'error' || value === 'failed' || value === 'failure') return 'error'
  return 'idle'
}

function normalizeTraceStatus(value: unknown, type?: string): TraceStatus {
  if (type === 'step.completed' && value !== 'error' && value !== 'failed') return 'completed'
  if (type === 'run.failed' || type === 'node.failed') return 'error'
  if (type === 'node.retrying') return 'retrying'
  if (type === 'node.completed') return 'completed'
  if (type === 'node.skipped') return 'skipped'
  if (type === 'run.completed') return 'completed'
  if (value === 'running' || value === 'pending' || value === 'completed' || value === 'retrying' || value === 'failed' || value === 'error' || value === 'skipped') {
    return value
  }
  if (value === 'success' || value === 'degraded' || value === 'complete') return 'completed'
  if (value === 'failed' || value === 'failure') return 'error'
  if (type === 'step.started' || type === 'node.started' || type === 'run.started') return 'running'
  return 'pending'
}

function normalizeKind(value: unknown): TraceKind {
  if (typeof value === 'string' && ['node', 'tool', 'source', 'evaluation', 'artifact', 'error', 'info', 'routing', 'sources', 'search', 'paper', 'enrichment', 'notes', 'benchmark', 'report', 'workflow'].includes(value)) return value as TraceKind
  return 'node'
}

function normalizeTraceType(value: unknown): TraceEventType {
  return typeof value === 'string' && value !== 'snapshot' ? value as TraceEventType : 'step.updated'
}

function normalizeProgress(value: unknown): number {
  const number = typeof value === 'number' && Number.isFinite(value) ? value : 0
  return Math.max(0, Math.min(100, number <= 1 ? number * 100 : number))
}

function normalizeFacts(value: unknown): TraceFacts | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const raw = asRecord(value)
  const facts: TraceFacts = {}
  if (raw.intent === 'search' || raw.intent === 'direct_read' || raw.intent === 'direct_compare' || raw.intent === 'direct_answer') facts.intent = raw.intent
  for (const key of ['resultCount', 'selectedCount', 'retryCount', 'paperCount', 'failedPdfCount', 'repositoryCount', 'bibtexCount'] as const) {
    const count = raw[key]
    if (typeof count === 'number' && Number.isInteger(count) && count >= 0) facts[key] = count
  }
  if (Array.isArray(raw.noteFallbackPaperIds)) facts.noteFallbackPaperIds = asStringArray(raw.noteFallbackPaperIds)
  if (Array.isArray(raw.comparedPaperIds)) facts.comparedPaperIds = asStringArray(raw.comparedPaperIds)
  if (typeof raw.comparisonAvailable === 'boolean') facts.comparisonAvailable = raw.comparisonAvailable
  if (typeof raw.reportAvailable === 'boolean') facts.reportAvailable = raw.reportAvailable
  if (typeof raw.reportId === 'string') facts.reportId = raw.reportId
  if (typeof raw.parserWarning === 'string') facts.parserWarning = raw.parserWarning
  if (Array.isArray(raw.parserWarnings)) facts.parserWarnings = asStringArray(raw.parserWarnings)
  if (Array.isArray(raw.headingFallbackPaperIds)) facts.headingFallbackPaperIds = asStringArray(raw.headingFallbackPaperIds)
  return Object.keys(facts).length ? facts : null
}

export function normalizeTraceEvent(input: unknown, fallbackRunId = ''): TraceEvent {
  const record = asRecord(input)
  const type = normalizeTraceType(firstString(record, 'type', 'event'))
  const status = normalizeTraceStatus(record.status, type)
  const node = firstString(record, 'node', 'node_name', 'currentNode', 'current_node') ?? 'workflow'
  const label = firstString(record, 'label', 'name') ?? node
  const summary = firstString(record, 'summary', 'message', 'detail') ?? ''
  const rawDetails = record.details ?? record.description ?? null
  const details = typeof rawDetails === 'string'
    ? rawDetails
    : rawDetails !== null && typeof rawDetails === 'object'
      ? rawDetails as Record<string, unknown>
      : null
  const detailsLinks: unknown[] = rawDetails !== null && typeof rawDetails === 'object' && Array.isArray((rawDetails as Record<string, unknown>).links)
    ? (rawDetails as Record<string, unknown>).links as unknown[]
    : []
  const topLevelLinks: unknown[] = Array.isArray(record.links) ? record.links as unknown[] : []
  const rawLinks: unknown[] = topLevelLinks.length ? topLevelLinks : detailsLinks
  const links = rawLinks.length
    ? rawLinks
        .map((link) => {
          const item = asRecord(link)
          const href = firstString(item, 'href', 'url')
          const linkLabel = firstString(item, 'label', 'name')
          return href && linkLabel ? { href, label: linkLabel } : null
        })
        .filter((link): link is { href: string; label: string } => Boolean(link))
    : undefined

  const detailsRecord = details !== null && typeof details === 'object' ? details : {}
  return {
    runId: firstString(record, 'runId', 'run_id') ?? fallbackRunId,
    threadId: firstString(record, 'threadId', 'thread_id'),
    seq: firstNumber(record, 'seq', 'sequence') ?? 0,
    type,
    node,
    label,
    kind: normalizeKind(record.kind),
    status,
    summary,
    details,
    timestamp: firstString(record, 'timestamp', 'createdAt', 'created_at') ?? new Date().toISOString(),
    progress: normalizeProgress(record.progress),
    durationMs: firstNumber(record, 'durationMs', 'duration_ms') ?? firstNumber(detailsRecord, 'durationMs', 'duration_ms'),
    retry: (() => {
      const retry = firstObject(record, 'retry')
      const current = firstNumber(retry, 'current', 'attempt', 'retryCount', 'retry_count')
      if (current === undefined) return null
      return { current, max: firstNumber(retry, 'max', 'limit') }
    })(),
    artifactIds: asStringArray(record.artifactIds ?? record.artifact_ids),
    links,
    facts: normalizeFacts(record.facts ?? detailsRecord.facts),
  }
}

export function normalizeMessage(input: unknown, fallbackThreadId = '', fallbackId = ''): ThreadMessage {
  const record = unwrapRecord(input, 'message')
  const threadId = firstString(record, 'threadId', 'thread_id') ?? fallbackThreadId
  const runId = firstString(record, 'runId', 'run_id')
  const content = firstString(record, 'content', 'text', 'answer', 'query', 'message') ?? ''
  const attachments = firstArray<unknown>(record, 'attachments', 'files').map(normalizeAttachment).filter((item): item is MessageAttachment => Boolean(item))
  const artifactIds = asStringArray(record.artifactIds ?? record.artifact_ids)
  return {
    id: firstString(record, 'id', 'messageId', 'message_id') ?? fallbackId,
    threadId,
    thread_id: firstString(record, 'thread_id'),
    runId: runId ?? null,
    run_id: firstString(record, 'run_id') ?? null,
    role: normalizeMessageRole(record.role),
    content,
    createdAt: firstString(record, 'createdAt', 'created_at', 'timestamp'),
    created_at: firstString(record, 'created_at'),
    status: normalizeMessageStatus(record.status),
    attachments: attachments.length ? attachments : undefined,
    artifactIds: artifactIds.length ? artifactIds : undefined,
    artifact_ids: asStringArray(record.artifact_ids),
    reportId: firstString(record, 'reportId', 'report_id') ?? null,
    report_id: firstString(record, 'report_id') ?? null,
    query: firstString(record, 'query'),
    answer: firstString(record, 'answer'),
  }
}

export function normalizeThreadSummary(input: unknown): Thread {
  const record = unwrapRecord(input, 'thread')
  const threadId = firstString(record, 'threadId', 'thread_id', 'id') ?? ''
  return {
    threadId,
    id: threadId,
    title: firstString(record, 'title', 'name') ?? 'Nghiên cứu mới',
    createdAt: firstString(record, 'createdAt', 'created_at'),
    created_at: firstString(record, 'created_at'),
    updatedAt: firstString(record, 'updatedAt', 'updated_at', 'timestamp'),
    updated_at: firstString(record, 'updated_at'),
    messageCount: firstNumber(record, 'messageCount', 'message_count'),
    message_count: firstNumber(record, 'message_count'),
    activeRunId: firstString(record, 'activeRunId', 'active_run_id') ?? null,
    active_run_id: firstString(record, 'active_run_id') ?? null,
    runIds: asStringArray(record.runIds ?? record.run_ids),
    run_ids: asStringArray(record.run_ids),
  }
}

export function normalizeThreadRun(input: unknown): ThreadRun {
  const record = unwrapRecord(input, 'run')
  const runId = firstString(record, 'runId', 'run_id', 'id') ?? ''
  return {
    runId,
    run_id: firstString(record, 'run_id'),
    status: normalizeStatus(record.status),
    provider: firstString(record, 'provider') ?? 'unknown',
    model: firstString(record, 'model') ?? 'unknown',
    currentNode: firstString(record, 'currentNode', 'current_node') ?? null,
    current_node: firstString(record, 'current_node') ?? null,
    answer: firstString(record, 'answer', 'content') ?? null,
    reportId: firstString(record, 'reportId', 'report_id') ?? null,
    report_id: firstString(record, 'report_id') ?? null,
    error: firstString(record, 'error', 'errorMessage', 'error_message') ?? null,
    createdAt: firstString(record, 'createdAt', 'created_at'),
    created_at: firstString(record, 'created_at'),
    updatedAt: firstString(record, 'updatedAt', 'updated_at'),
    updated_at: firstString(record, 'updated_at'),
  }
}

export function normalizeArtifact(input: unknown): ResearchArtifact {
  const record = unwrapRecord(input, 'artifact')
  return {
    id: firstString(record, 'id', 'artifactId', 'artifact_id') ?? '',
    kind: firstString(record, 'kind', 'type') ?? 'artifact',
    title: firstString(record, 'title', 'name') ?? 'Research artifact',
    summary: firstString(record, 'summary', 'description') ?? '',
    reportId: firstString(record, 'reportId', 'report_id') ?? null,
    report_id: firstString(record, 'report_id') ?? null,
  }
}

export function normalizeThread(input: unknown): ThreadDetail {
  const record = unwrapRecord(input, 'thread')
  const summary = normalizeThreadSummary(record)
  const rawMessages = firstArray<unknown>(record, 'messages', 'items')
  const messages = rawMessages.map((message, index) => normalizeMessage(message, summary.threadId, `${summary.threadId}:message:${index}`))
  const runs = firstArray<unknown>(record, 'runs').map(normalizeThreadRun).filter((run) => Boolean(run.runId))
  const inferredRunIds = runs.map((run) => run.runId).filter(Boolean)
  const runIds = summary.runIds.length ? summary.runIds : inferredRunIds
  return {
    ...summary,
    messages,
    runIds: [...new Set(runIds.length ? runIds : inferredRunIds)],
    run_ids: asStringArray(record.run_ids),
    runs,
    artifacts: firstArray<unknown>(record, 'artifacts').map(normalizeArtifact).filter((artifact) => Boolean(artifact.id)),
  }
}

export function normalizeAssistantEvent(input: unknown, fallbackRunId = '', fallbackThreadId = ''): NormalizedAssistantEvent | null {
  const record = asRecord(input)
  const rawType = firstString(record, 'type', 'event')
  const type = rawType === 'assistant_delta' || rawType === 'assistant.delta'
    ? 'assistant_delta'
    : rawType === 'assistant_completed' || rawType === 'assistant.completed'
      ? 'assistant_completed'
      : null
  if (!type) return null
  const text = firstString(record, 'text', 'delta', 'content', 'answer') ?? ''
  if (!text && type === 'assistant_delta') return null
  return {
    type,
    runId: firstString(record, 'runId', 'run_id') ?? fallbackRunId,
    threadId: firstString(record, 'threadId', 'thread_id') ?? fallbackThreadId,
    seq: firstNumber(record, 'seq', 'sequence'),
    messageId: firstString(record, 'messageId', 'message_id'),
    text,
    timestamp: firstString(record, 'timestamp', 'createdAt', 'created_at'),
  }
}

export function normalizeSnapshot(input: unknown, fallbackThreadId = ''): RunSnapshot {
  const record = unwrapRecord(input, 'run', 'snapshot')
  const runId = firstString(record, 'runId', 'run_id', 'id') ?? ''
  const threadId = firstString(record, 'threadId', 'thread_id') ?? fallbackThreadId
  const rawInput = asRecord(record.input ?? record.request)
  const rawEvents = firstArray<unknown>(record, 'traceEvents', 'trace_events', 'trace', 'events')
  const normalizedEvents = rawEvents.map((event) => {
    const normalized = normalizeTraceEvent(event, runId)
    return normalized.threadId ? normalized : { ...normalized, threadId: threadId || undefined }
  })
  const rawResult = record.result ?? record.output
  const resultRecord = rawResult !== null && typeof rawResult === 'object' ? asRecord(rawResult) : null
  const result = resultRecord
    ? {
        ...resultRecord,
        answer: firstString(resultRecord, 'answer', 'content', 'text') ?? null,
      } as RunResult
    : null
  const status = normalizeStatus(record.status)
  const paperInputs = firstArray<unknown>(rawInput, 'paperInputs', 'paper_inputs', 'inputs')
    .filter((value): value is string => typeof value === 'string')

  return {
    runId,
    run_id: typeof record.run_id === 'string' ? record.run_id : undefined,
    threadId: threadId || undefined,
    thread_id: firstString(record, 'thread_id') ?? undefined,
    input: {
      query: firstString(rawInput, 'query', 'userQuery', 'user_query') ?? '',
      paperInputs,
      paper_inputs: paperInputs,
      fileNames: firstArray<unknown>(rawInput, 'fileNames', 'file_names', 'pdfNames', 'pdf_names', 'files').filter((value): value is string => typeof value === 'string'),
      pdfNames: firstArray<unknown>(rawInput, 'pdfNames', 'pdf_names', 'fileNames', 'file_names', 'files').filter((value): value is string => typeof value === 'string'),
    },
    provider: firstString(record, 'provider', 'llmProvider', 'llm_provider') ?? 'unknown',
    model: firstString(record, 'model', 'modelName', 'model_name') ?? 'unknown',
    status,
    answer: firstString(record, 'answer') ?? (result ? firstString(asRecord(result), 'answer') ?? null : null),
    currentNode: firstString(record, 'currentNode', 'current_node', 'node') ?? null,
    current_node: firstString(record, 'current_node') ?? undefined,
    error: firstString(record, 'error', 'errorMessage', 'error_message') ?? null,
    errorMessage: firstString(record, 'errorMessage', 'error_message') ?? undefined,
    error_message: firstString(record, 'error_message') ?? undefined,
    progress: record.progress !== undefined ? normalizeProgress(record.progress) : status !== 'running' ? 100 : Math.max(...normalizedEvents.map((event) => event.progress), 0),
    traceEvents: normalizedEvents,
    trace_events: normalizedEvents,
    seq: firstNumber(record, 'seq', 'sequence') ?? Math.max(...normalizedEvents.map((event) => event.seq), 0),
    result,
    startedAt: firstString(record, 'startedAt', 'started_at'),
    started_at: firstString(record, 'started_at'),
    createdAt: firstString(record, 'createdAt', 'created_at'),
    created_at: firstString(record, 'created_at'),
    updatedAt: firstString(record, 'updatedAt', 'updated_at'),
    updated_at: firstString(record, 'updated_at'),
  }
}

function normalizeConfig(input: unknown): AppConfig {
  const record = asRecord(input)
  const provider = firstString(record, 'provider', 'defaultProvider', 'default_provider') ?? 'unknown'
  const model = firstString(record, 'model', 'defaultModel', 'default_model') ?? 'unknown'
  const providerValues = Array.isArray(record.providers)
    ? record.providers
    : record.providers !== null && typeof record.providers === 'object'
      ? Object.entries(record.providers as Record<string, unknown>).map(([name, value]) => ({ name, ...asRecord(value) }))
      : []
  const providers = providerValues.map((value) => {
    const item = asRecord(value)
    return {
      provider: firstString(item, 'provider', 'name') ?? 'unknown',
      model: firstString(item, 'model', 'defaultModel', 'default_model') ?? 'unknown',
      configured: Boolean(item.configured),
      apiBase: firstString(item, 'apiBase', 'api_base'),
    }
  })
  return {
    provider,
    model,
    configured: Boolean(record.configured ?? record.isConfigured ?? record.is_configured),
    providers: providers.length ? providers : undefined,
    apiBase: firstString(record, 'apiBase', 'api_base'),
  }
}

function normalizeReportSummary(input: unknown): ReportSummary {
  const record = asRecord(input)
  return {
    id: firstString(record, 'id', 'reportId', 'report_id', 'name') ?? '',
    title: firstString(record, 'title', 'name') ?? 'Báo cáo chưa đặt tên',
    createdAt: firstString(record, 'createdAt', 'created_at', 'timestamp'),
    created_at: firstString(record, 'created_at'),
    updatedAt: firstString(record, 'updatedAt', 'updated_at'),
    updated_at: firstString(record, 'updated_at'),
    size: firstNumber(record, 'size', 'sizeBytes', 'size_bytes'),
    paperCount: firstNumber(record, 'paperCount', 'paper_count'),
    paper_count: firstNumber(record, 'paper_count'),
    status: record.status ? normalizeStatus(record.status) : undefined,
  }
}

export function normalizeReport(input: unknown): ReportDocument {
  const record = asRecord(input)
  const summary = normalizeReportSummary(record)
  return {
    ...summary,
    markdown: firstString(record, 'markdown', 'content', 'report', 'finalReport', 'final_report') ?? '',
    content: firstString(record, 'content'),
    report: firstString(record, 'report'),
    papers: firstArray<PaperSource>(record, 'papers', 'selectedPapers', 'selected_papers'),
    benchmark: firstString(record, 'benchmark', 'benchmarkMatrix', 'benchmark_matrix') ?? null,
    bibtex: firstString(record, 'bibtex') ?? null,
  }
}

async function readError(response: Response): Promise<unknown> {
  const contentType = response.headers.get('content-type') ?? ''
  try {
    if (contentType.includes('application/json')) return await response.json()
    return await response.text()
  } catch {
    return undefined
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const hasBody = init?.body !== undefined && init?.body !== null
  const isFormData = typeof FormData !== 'undefined' && init?.body instanceof FormData
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      Accept: 'application/json',
      ...(hasBody && !isFormData ? { 'Content-Type': 'application/json' } : {}),
      ...(init?.headers ?? {}),
    },
  })
  if (!response.ok) {
    const body = await readError(response)
    const bodyRecord = asRecord(body)
    const nestedDetail = asRecord(bodyRecord.detail)
    const message = firstString(bodyRecord, 'detail', 'message', 'error') ?? firstString(nestedDetail, 'message', 'detail', 'error') ?? `Yêu cầu thất bại (${response.status})`
    throw new ApiError(message, response.status, body)
  }
  return (await response.json()) as T
}

export async function getConfig(): Promise<AppConfig> {
  return normalizeConfig(await requestJson<unknown>('/config'))
}

export async function getThreads(): Promise<Thread[]> {
  const response = await requestJson<unknown>('/threads')
  const record = asRecord(response)
  const items = Array.isArray(response) ? response : firstArray<unknown>(record, 'threads', 'items')
  return items.map(normalizeThreadSummary).filter((thread) => Boolean(thread.threadId))
}

export async function createThread(input: ThreadInput = {}): Promise<ThreadDetail> {
  const body = input.title?.trim() ? JSON.stringify({ title: input.title.trim() }) : undefined
  return normalizeThread(await requestJson<unknown>('/threads', { method: 'POST', ...(body ? { body } : {}) }))
}

export async function getThread(threadId: string): Promise<ThreadDetail> {
  return normalizeThread(await requestJson<unknown>(`/threads/${encodeURIComponent(threadId)}`))
}

export async function sendThreadMessage(threadId: string, input: SendMessageInput | string): Promise<RunSnapshot> {
  const message = typeof input === 'string' ? { content: input } : input
  const content = message.content.trim()
  const form = new FormData()
  // ``query`` is the canonical field understood by the legacy graph bridge;
  // ``content`` lets the thread endpoint preserve the chat vocabulary.
  form.append('query', content)
  form.append('content', content)
  ;(message.paperInputs ?? message.paper_inputs ?? []).forEach((paperInput) => form.append('paper_inputs', paperInput))
  ;(message.files ?? []).forEach((file) => form.append('files', file, file.name))
  return normalizeSnapshot(await requestJson<unknown>(`/threads/${encodeURIComponent(threadId)}/messages`, { method: 'POST', body: form }), threadId)
}

export async function createRun(input: RunInput): Promise<RunSnapshot> {
  const form = new FormData()
  form.append('query', input.query)
  input.paperInputs.forEach((paperInput) => form.append('paper_inputs', paperInput))
  input.files.forEach((file) => form.append('files', file, file.name))
  return normalizeSnapshot(await requestJson<unknown>('/runs', { method: 'POST', body: form }))
}

export async function getRun(runId: string): Promise<RunSnapshot> {
  return normalizeSnapshot(await requestJson<unknown>(`/runs/${encodeURIComponent(runId)}`))
}

interface SseFrame {
  type: string
  data: string
}

function parseSseFrames(body: string): SseFrame[] {
  const frames: SseFrame[] = []
  let type = 'message'
  let data: string[] = []

  const flush = () => {
    if (data.length) frames.push({ type, data: data.join('\n') })
    type = 'message'
    data = []
  }

  for (const line of body.split(/\r?\n/)) {
    if (!line) {
      flush()
      continue
    }
    if (line.startsWith(':')) continue
    const separator = line.indexOf(':')
    const field = separator === -1 ? line : line.slice(0, separator)
    const value = separator === -1 ? '' : line.slice(separator + 1).replace(/^ /, '')
    if (field === 'event') type = value || 'message'
    if (field === 'data') data.push(value)
  }
  flush()
  return frames
}

function normalizeEventItems(input: unknown, runId: string, afterSeq: number): TraceEvent[] {
  const items = Array.isArray(input) ? input : (() => {
    const record = asRecord(input)
    return firstArray<unknown>(record, 'events', 'traceEvents', 'trace_events')
  })()
  const normalized: TraceEvent[] = []
  items.forEach((item) => {
    const record = asRecord(item)
    if (firstString(record, 'type', 'event') === 'snapshot' || 'traceEvents' in record || 'trace_events' in record) {
      normalized.push(...normalizeSnapshot(item).traceEvents)
      return
    }
    normalized.push(normalizeTraceEvent(item, runId))
  })
  return reconcileEvents([], normalized).filter((event) => event.seq > afterSeq)
}

export async function getRunEvents(runId: string, afterSeq = 0): Promise<TraceEvent[]> {
  const response = await fetch(`${API_BASE}/runs/${encodeURIComponent(runId)}/events?after_seq=${afterSeq}&after=${afterSeq}`, {
    headers: { Accept: 'text/event-stream, application/json' },
  })
  if (!response.ok) {
    const body = await readError(response)
    const bodyRecord = asRecord(body)
    const detail = firstString(bodyRecord, 'detail', 'message', 'error') ?? `Không thể tải trace (${response.status})`
    throw new ApiError(detail, response.status, body)
  }

  const body = await response.text()
  if (!body.trim()) return []
  try {
    // Supporting JSON here keeps the client friendly to lightweight test
    // transports and older bridges, while the production endpoint is SSE.
    if ((response.headers.get('content-type') ?? '').includes('application/json') || /^[\[{]/.test(body.trim())) {
      return normalizeEventItems(JSON.parse(body) as unknown, runId, afterSeq)
    }
  } catch {
    return []
  }

  const frames = parseSseFrames(body)
  const items = frames.flatMap((frame) => {
    try {
      const payload = JSON.parse(frame.data) as unknown
      return [{ ...asRecord(payload), type: frame.type }]
    } catch {
      return []
    }
  })
  return normalizeEventItems(items, runId, afterSeq)
}

export async function getReports(): Promise<ReportSummary[]> {
  const response = await requestJson<unknown>('/reports')
  const record = asRecord(response)
  const items = Array.isArray(response) ? response : firstArray<unknown>(record, 'reports', 'items')
  return items.map(normalizeReportSummary).filter((report) => report.id)
}

export async function getReport(reportId: string): Promise<ReportDocument> {
  return normalizeReport(await requestJson<unknown>(`/reports/${encodeURIComponent(reportId)}`))
}

export async function downloadReport(reportId: string): Promise<Blob> {
  const response = await fetch(`${API_BASE}/reports/${encodeURIComponent(reportId)}/download`, {
    headers: { Accept: 'text/markdown, application/octet-stream' },
  })
  if (!response.ok) {
    const body = await readError(response)
    throw new ApiError(`Không thể tải báo cáo (${response.status})`, response.status, body)
  }
  return response.blob()
}

export function getApiUrl(path: string): string {
  return `${API_BASE}${path}`
}

export type RunEventHandler = (event: TraceEvent | ServerRunEvent | RunSnapshot) => void

const SSE_EVENT_TYPES: string[] = [
  'snapshot',
  'run.started',
  'step.started',
  'step.updated',
  'step.completed',
  'node.started',
  'node.progress',
  'node.completed',
  'node.retrying',
  'node.failed',
  'node.skipped',
  'tool.started',
  'tool.completed',
  'thinking_step',
  'artifact_created',
  'assistant_delta',
  'assistant_completed',
  'assistant.delta',
  'assistant.completed',
  'run.completed',
  'run.failed',
]

/**
 * Subscribe to the server's event stream. The caller owns the returned close
 * function and should reconnect only after fetching a fresh snapshot.
 */
export function subscribeToRun(runId: string, afterSeq: number, onEvent: RunEventHandler, onError: (error: Event) => void): () => void {
  if (typeof EventSource === 'undefined') {
    onError(new Event('error'))
    return () => undefined
  }

  const source = new EventSource(`${getApiUrl(`/runs/${encodeURIComponent(runId)}/events`)}?after_seq=${afterSeq}&after=${afterSeq}`)
  const handle = (event: MessageEvent<string>) => {
    try {
      const parsed = JSON.parse(event.data) as ServerRunEvent
      onEvent({ ...parsed, type: parsed.type ?? event.type })
    } catch {
      // Ignore keep-alive comments or malformed server frames; the next
      // snapshot/reconnect will repair the client state.
    }
  }
  source.onmessage = handle
  SSE_EVENT_TYPES.forEach((type) => source.addEventListener(type, handle as EventListener))
  source.onerror = onError

  return () => {
    source.close()
  }
}

export function reconcileEvents(existing: TraceEvent[], incoming: Array<TraceEvent | ServerRunEvent>): TraceEvent[] {
  const bySeq = new Map<number, TraceEvent>()
  existing.forEach((event) => bySeq.set(event.seq, event))
  const existingRunIds = new Set(existing.map((event) => event.runId).filter(Boolean))
  incoming.forEach((raw) => {
    const rawRunId = 'run_id' in raw ? raw.run_id : undefined
    const normalized = normalizeTraceEvent(raw, raw.runId ?? rawRunId ?? existing[0]?.runId ?? '')
    if (existingRunIds.size && normalized.runId && !existingRunIds.has(normalized.runId)) return
    if (normalized.seq > 0) bySeq.set(normalized.seq, normalized)
  })
  return [...bySeq.values()].sort((a, b) => a.seq - b.seq)
}
