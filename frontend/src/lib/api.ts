import type { RunSnapshot, TraceEvent } from '../types'

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

/** A frame from GET /api/runs/{id}/events: `snapshot` carries a RunSnapshot, every other type a TraceEvent. */
export type RunFrame = { type: 'snapshot'; data: RunSnapshot } | { type: string; data: TraceEvent }

/** Everything the UI needs from the backend. The mock mode implements the same interface. */
export interface Transport {
  createThread(): Promise<string>
  sendMessage(threadId: string, text: string, file?: File | null, arxivId?: string): Promise<RunSnapshot>
  getRun(runId: string): Promise<RunSnapshot>
  subscribe(runId: string, afterSeq: number, onFrame: (frame: RunFrame) => void, onError: () => void): () => void
  downloadReport(reportId: string): Promise<Blob>
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

async function readBody(response: Response): Promise<unknown> {
  try {
    if ((response.headers.get('content-type') ?? '').includes('application/json')) return await response.json()
    return await response.text()
  } catch {
    return undefined
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { Accept: 'application/json', ...(init?.headers ?? {}) },
  })
  if (!response.ok) {
    const body = await readBody(response)
    const detail = isRecord(body) ? body.detail : undefined
    const message =
      (typeof detail === 'string' && detail) ||
      (isRecord(detail) && typeof detail.message === 'string' && detail.message) ||
      `Yêu cầu thất bại (${response.status})`
    throw new ApiError(message, response.status, body)
  }
  return (await response.json()) as T
}

export function normalizeSnapshot(raw: unknown): RunSnapshot {
  const r = isRecord(raw) ? raw : {}
  return {
    runId: String(r.runId ?? ''),
    threadId: typeof r.threadId === 'string' ? r.threadId : null,
    status: (typeof r.status === 'string' ? r.status : 'running') as RunSnapshot['status'],
    answer: typeof r.answer === 'string' ? r.answer : null,
    error: typeof r.error === 'string' ? r.error : null,
    seq: typeof r.seq === 'number' ? r.seq : 0,
    traceEvents: Array.isArray(r.traceEvents) ? (r.traceEvents as TraceEvent[]) : [],
    result: isRecord(r.result) ? (r.result as RunSnapshot['result']) : undefined,
    startedAt: typeof r.startedAt === 'string' ? r.startedAt : null,
    updatedAt: typeof r.updatedAt === 'string' ? r.updatedAt : null,
  }
}

const SSE_TYPES = [
  'snapshot',
  'step.started',
  'step.updated',
  'step.completed',
  'assistant.delta',
  'assistant.completed',
  'answer.summary',
  'run.completed',
  'run.failed',
]

export const httpTransport: Transport = {
  async createThread() {
    const body = await request<{ threadId: string }>('/threads', { method: 'POST' })
    return body.threadId
  },

  async sendMessage(threadId, text, file, arxivId) {
    const form = new FormData()
    form.append('query', text)
    if (arxivId) form.append('arxivId', arxivId)
    if (file) form.append('files', file, file.name)
    return normalizeSnapshot(await request<unknown>(`/threads/${encodeURIComponent(threadId)}/messages`, { method: 'POST', body: form }))
  },

  async getRun(runId) {
    return normalizeSnapshot(await request<unknown>(`/runs/${encodeURIComponent(runId)}`))
  },

  subscribe(runId, afterSeq, onFrame, onError) {
    if (typeof EventSource === 'undefined') {
      onError()
      return () => undefined
    }
    // EventSource reconnects on its own and resends Last-Event-ID; the first frame of
    // every connection is a snapshot, so the client state is rebuilt after a gap.
    const source = new EventSource(`${API_BASE}/runs/${encodeURIComponent(runId)}/events?after_seq=${afterSeq}`)
    const handle = (event: MessageEvent<string>) => {
      try {
        onFrame({ type: event.type, data: JSON.parse(event.data) } as RunFrame)
      } catch {
        // Ignore malformed frames; the next snapshot repairs the state.
      }
    }
    SSE_TYPES.forEach((type) => source.addEventListener(type, handle as EventListener))
    source.onerror = () => onError()
    return () => source.close()
  },

  async downloadReport(reportId) {
    const response = await fetch(`${API_BASE}/reports/${encodeURIComponent(reportId)}/download`)
    if (!response.ok) throw new ApiError(`Không thể tải báo cáo (${response.status})`, response.status)
    return response.blob()
  },
}
