import type { RunSnapshot } from '../../types'

export type ChatRole = 'user' | 'assistant' | 'system'

export interface ChatArtifact {
  id: string
  label: string
  href: string
  kind?: string
  external?: boolean
}

export interface ChatMessageModel {
  id: string
  role: ChatRole
  content: string
  runId?: string
  timestamp?: string
  status?: 'pending' | 'running' | 'completed' | 'error'
  artifacts: ChatArtifact[]
}

export interface ThreadModel {
  id: string
  title: string
  updatedAt?: string
  createdAt?: string
  messageCount?: number
  status?: string
}

export interface ChatInput {
  query: string
  paperInputs: string[]
  files: File[]
}

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' ? value as Record<string, unknown> : {}
}

function firstString(record: Record<string, unknown>, ...keys: string[]): string | undefined {
  for (const key of keys) {
    if (typeof record[key] === 'string' && record[key].trim()) return record[key].trim()
  }
  return undefined
}

function firstArray(record: Record<string, unknown>, ...keys: string[]): unknown[] {
  for (const key of keys) if (Array.isArray(record[key])) return record[key] as unknown[]
  return []
}

function roleFrom(value: unknown): ChatRole {
  if (value === 'user' || value === 'human') return 'user'
  if (value === 'system') return 'system'
  return 'assistant'
}

function artifactFrom(value: unknown, index: number): ChatArtifact | null {
  const record = asRecord(value)
  const href = firstString(record, 'href', 'url', 'link')
  const label = firstString(record, 'label', 'title', 'name')
  if (!href || !label) return null
  return {
    id: firstString(record, 'id', 'artifactId', 'artifact_id') ?? `${href}-${index}`,
    label,
    href,
    kind: firstString(record, 'kind', 'type'),
    external: Boolean(record.external ?? record.isExternal ?? record.is_external),
  }
}

function readMessageContent(record: Record<string, unknown>): string {
  const direct = firstString(record, 'content', 'markdown', 'text', 'message', 'body')
  if (direct) return direct
  const parts = firstArray(record, 'parts', 'contentParts', 'content_parts')
  return parts.map((part) => {
    const partRecord = asRecord(part)
    return firstString(partRecord, 'text', 'content', 'markdown') ?? (typeof part === 'string' ? part : '')
  }).filter(Boolean).join('\n\n')
}

function readMessageArtifacts(record: Record<string, unknown>): ChatArtifact[] {
  const raw = firstArray(record, 'artifacts', 'artifactLinks', 'artifact_links', 'links', 'sources')
  const artifacts = raw.map(artifactFrom).filter((value): value is ChatArtifact => Boolean(value))
  const deduped = new Map<string, ChatArtifact>()
  artifacts.forEach((artifact) => deduped.set(`${artifact.href}-${artifact.label}`, artifact))
  return [...deduped.values()]
}

export function normalizeMessages(thread: unknown): ChatMessageModel[] {
  const threadRecord = asRecord(thread)
  const rawMessages = firstArray(threadRecord, 'messages', 'turns', 'chatMessages', 'chat_messages')
  return rawMessages.map((raw, index) => {
    const record = asRecord(raw)
    const role = roleFrom(record.role ?? record.author ?? record.sender)
    const status = firstString(record, 'status') as ChatMessageModel['status'] | undefined
    return {
      id: firstString(record, 'id', 'messageId', 'message_id') ?? `${role}-${index}`,
      role,
      content: readMessageContent(record),
      runId: firstString(record, 'runId', 'run_id', 'threadRunId', 'thread_run_id'),
      timestamp: firstString(record, 'timestamp', 'createdAt', 'created_at'),
      status: status && ['pending', 'running', 'completed', 'error'].includes(status) ? status : undefined,
      artifacts: readMessageArtifacts(record),
    }
  }).filter((message) => message.content || message.artifacts.length || message.status === 'running')
}

export function normalizeThreads(raw: unknown): ThreadModel[] {
  const record = asRecord(raw)
  const values = Array.isArray(raw) ? raw : firstArray(record, 'threads', 'items', 'data')
  return values.flatMap((value, index): ThreadModel[] => {
    const item = asRecord(value)
    const id = firstString(item, 'id', 'threadId', 'thread_id', 'runId', 'run_id')
    if (!id) return []
    const messages = firstArray(item, 'messages', 'turns', 'chatMessages', 'chat_messages')
    return [{
      id,
      title: firstString(item, 'title', 'name', 'subject', 'query') ?? `Research thread ${index + 1}`,
      updatedAt: firstString(item, 'updatedAt', 'updated_at', 'timestamp'),
      createdAt: firstString(item, 'createdAt', 'created_at'),
      messageCount: typeof item.messageCount === 'number' ? item.messageCount : messages.length || undefined,
      status: firstString(item, 'status'),
    }]
  })
}

export function readThreadId(thread: unknown): string | undefined {
  const record = asRecord(thread)
  return firstString(record, 'id', 'threadId', 'thread_id', 'runId', 'run_id')
}

export function readThreadTitle(thread: unknown): string | undefined {
  const record = asRecord(thread)
  return firstString(record, 'title', 'name', 'subject', 'query')
}

export function messagesForView(thread: unknown, currentRun: RunSnapshot | null | undefined): ChatMessageModel[] {
  const messages = normalizeMessages(thread)
  if (!currentRun) return messages
  const activeAssistant = messages.some((message) => message.role === 'assistant' && (message.runId === currentRun.runId || message.status === 'running'))
  if (activeAssistant) return messages

  const query = currentRun.input?.query?.trim()
  const result = currentRun.result
  const report = currentRun.answer ?? result?.answer ?? result?.report ?? result?.finalReport ?? result?.final_report ?? ''
  const assistantFallback: ChatMessageModel = {
    id: `${currentRun.runId}-assistant`,
    role: 'assistant',
    content: report ?? '',
    runId: currentRun.runId,
    status: currentRun.status === 'running' ? 'running' : currentRun.status === 'error' ? 'error' : 'completed',
    artifacts: [],
  }
  if (messages.length && (report || currentRun.status === 'running' || currentRun.status === 'error')) return [...messages, assistantFallback]
  if (messages.length) return messages
  const fallback: ChatMessageModel[] = []
  if (query) fallback.push({ id: `${currentRun.runId}-user`, role: 'user', content: query, runId: currentRun.runId, artifacts: [] })
  if (report || currentRun.status === 'running' || currentRun.status === 'error') fallback.push(assistantFallback)
  return fallback
}

export function artifactsForRun(currentRun: RunSnapshot | null | undefined): ChatArtifact[] {
  if (!currentRun) return []
  const artifacts: ChatArtifact[] = []
  const result = currentRun.result
  const resultRecord = result as (typeof result & { reportId?: string; report_id?: string }) | null | undefined
  const reportId = resultRecord?.reportId ?? resultRecord?.report_id
  if (reportId) artifacts.push({ id: `report-${reportId}`, label: 'Mở report', href: `/library/${encodeURIComponent(reportId)}`, kind: 'report' })
  if (result?.papers?.length) artifacts.push({ id: 'workspace-papers', label: `${result.papers.length} Papers`, href: '#workspace:papers', kind: 'paper' })
  if (result?.papers?.some((paper) => Boolean(paper.notes))) artifacts.push({ id: 'workspace-pmrl', label: 'PMRL', href: '#workspace:pmrl', kind: 'notes' })
  if (result?.benchmark) artifacts.push({ id: 'workspace-compare', label: 'Benchmark matrix', href: '#workspace:compare', kind: 'compare' })
  if (result?.report) artifacts.push({ id: 'workspace-report', label: 'Research report', href: '#workspace:report', kind: 'report' })
  for (const paper of result?.papers ?? []) {
    const paperId = paper.arxivId ?? paper.arxiv_id ?? paper.paperId ?? paper.paper_id
    if (paperId) artifacts.push({ id: `paper-${paperId}`, label: paperId, href: `https://arxiv.org/abs/${encodeURIComponent(paperId.replace(/^arxiv:/i, ''))}`, kind: 'paper' })
    const pdf = paper.pdfUrl ?? paper.pdf_url
    if (pdf) artifacts.push({ id: `pdf-${paperId ?? pdf}`, label: 'PDF', href: pdf, kind: 'pdf' })
    for (const repo of paper.githubRepos ?? paper.github_repos ?? []) artifacts.push({ id: `repo-${repo.url}`, label: repo.name, href: repo.url, kind: 'github' })
  }
  const deduped = new Map<string, ChatArtifact>()
  artifacts.forEach((artifact) => deduped.set(`${artifact.href}-${artifact.label}`, artifact))
  return [...deduped.values()]
}

export function formatThreadDate(value?: string): string {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  const now = Date.now()
  const delta = now - date.getTime()
  if (delta < 60_000) return 'vừa xong'
  if (delta < 3_600_000) return `${Math.max(1, Math.round(delta / 60_000))} phút`
  if (delta < 86_400_000) return `${Math.max(1, Math.round(delta / 3_600_000))} giờ`
  return new Intl.DateTimeFormat('vi-VN', { day: '2-digit', month: '2-digit' }).format(date)
}
