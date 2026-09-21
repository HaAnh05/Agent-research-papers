export type RunStatus = 'idle' | 'running' | 'success' | 'degraded' | 'error'

export type TraceEventType =
  | 'step.started'
  | 'step.updated'
  | 'step.completed'
  | 'run.completed'
  | 'run.failed'
  | 'run.started'
  | 'node.started'
  | 'node.progress'
  | 'node.completed'
  | 'node.retrying'
  | 'node.failed'
  | 'node.skipped'
  | 'tool.started'
  | 'tool.completed'
  | 'thinking_step'
  | 'artifact_created'
  | 'assistant_delta'
  | 'assistant_completed'
  | (string & {})

export type TraceStatus = 'pending' | 'running' | 'completed' | 'retrying' | 'failed' | 'error' | 'skipped'

export type TraceKind =
  | 'node'
  | 'tool'
  | 'source'
  | 'evaluation'
  | 'artifact'
  | 'error'
  | 'info'
  | 'routing'
  | 'sources'
  | 'search'
  | 'paper'
  | 'enrichment'
  | 'notes'
  | 'benchmark'
  | 'report'
  | 'workflow'

export interface TraceEvent {
  runId: string
  threadId?: string
  seq: number
  type: TraceEventType
  node: string
  label: string
  kind: TraceKind
  status: TraceStatus
  summary: string
  details?: string | Record<string, unknown> | null
  timestamp: string
  progress: number
  durationMs?: number | null
  retry?: {
    current: number
    max?: number
  } | null
  artifactIds?: string[]
  links?: TraceLink[]
  facts?: TraceFacts | null
}

/** Allowlisted measurements emitted by the API from completed graph updates. */
export interface TraceFacts {
  intent?: 'search' | 'direct_read' | 'direct_compare'
  resultCount?: number
  selectedCount?: number
  retryCount?: number
  paperCount?: number
  failedPdfCount?: number
  repositoryCount?: number
  bibtexCount?: number
  noteFallbackPaperIds?: string[]
  comparisonAvailable?: boolean
  comparedPaperIds?: string[]
  reportAvailable?: boolean
  reportId?: string | null
  summaryCardReasons?: Record<string, string>
  /** Measured parser metadata surfaced by the API for source-specific UI. */
  parserWarning?: string | null
  parserWarnings?: string[]
  headingFallbackPaperIds?: string[]
}

export interface TraceLink {
  label: string
  href: string
}

export interface PaperSource {
  paperId?: string
  paper_id?: string
  title: string
  /** ArXiv abstract retained for source context; structured cards live in summaryCards. */
  summary?: string | null
  summaryCards?: SummaryCards | null
  summary_cards?: SummaryCards | null
  authors?: string[]
  published?: string
  publishedDate?: string
  published_date?: string
  submitted?: string
  submittedDate?: string
  submitted_date?: string
  subjects?: string[]
  metadataStatus?: 'complete' | 'missing' | 'unknown' | string
  metadata_status?: 'complete' | 'missing' | 'unknown' | string
  sourceType?: 'arxiv' | 'local_pdf' | string
  source_type?: 'arxiv' | 'local_pdf' | string
  arxivId?: string | null
  arxiv_id?: string | null
  pdfUrl?: string | null
  pdf_url?: string | null
  githubRepos?: GitHubRepo[]
  github_repos?: GitHubRepo[]
  bibtex?: string | null
  notes?: PMRLNotes | null
  sourceQuality?: Record<string, unknown> | null
  source_quality?: Record<string, unknown> | null
  notesQuality?: 'complete' | 'fallback' | 'missing' | 'unknown' | string
  notes_quality?: 'complete' | 'fallback' | 'missing' | 'unknown' | string
  pmrlStatus?: 'complete' | 'missing' | 'invalid' | 'unknown' | string
  pmrl_status?: 'complete' | 'missing' | 'invalid' | 'unknown' | string
}

/** Five short, source grounded cards shown in the default Results summary. */
export type SummaryCardStatus = 'complete' | 'unsupported' | 'missing' | 'invalid' | 'unknown'

export type SummaryCardStatusField =
  | 'tldr'
  | 'problem'
  | 'method'
  | 'keyResults'
  | 'key_results'
  | 'whyItMatters'
  | 'why_it_matters'

export interface SummaryCards {
  tldr?: string
  problem?: string
  method?: string
  keyResults?: string
  key_results?: string
  whyItMatters?: string
  why_it_matters?: string
  status?: 'complete' | 'partial' | 'missing' | 'invalid' | 'unknown' | string
  /** Per-card validation status, accepted in either API naming convention. */
  cardStatuses?: Partial<Record<SummaryCardStatusField, SummaryCardStatus>>
  card_statuses?: Partial<Record<SummaryCardStatusField, SummaryCardStatus>>
  /** Stable local-validator reason codes, accepted in either API naming convention. */
  cardReasons?: Partial<Record<SummaryCardStatusField, string>>
  card_reasons?: Partial<Record<SummaryCardStatusField, string>>
}

export interface PMRLNotes {
  problem: string
  method: string
  result: string
  limitation: string
  briefSummary?: string
  brief_summary?: string
  status?: 'complete' | 'missing' | 'invalid' | 'unknown' | string
}

export interface ComparisonMetric {
  metric: string
  dataset: string
  unit: string
  value: string
  sourceQuote?: string
  source_quote?: string
}

export interface ComparisonPaper {
  paperId?: string
  paper_id?: string
  title: string
  findings: string[]
  metrics: ComparisonMetric[]
}

export interface ComparisonRow {
  metric: string
  dataset: string
  unit: string
  values: Record<string, string>
  sourceQuotes?: Record<string, string>
  source_quotes?: Record<string, string>
  comparable: boolean
  reason: string
}

export interface ComparisonArtifact {
  available?: boolean
  papers: ComparisonPaper[]
  rows: ComparisonRow[]
  synthesis: string[]
  /** Optional legacy/source representation retained for the disclosure view. */
  markdown?: string | null
  warning?: string | null
}

export interface MessageAttachment {
  id?: string
  name: string
  mimeType?: string
  mime_type?: string
  size?: number
  url?: string | null
}

export type MessageRole = 'user' | 'assistant' | 'system'
export type MessageStatus = 'pending' | 'streaming' | 'completed' | 'error'

export interface ThreadMessage {
  id: string
  threadId: string
  thread_id?: string
  runId?: string | null
  run_id?: string | null
  role: MessageRole
  content: string
  createdAt?: string
  created_at?: string
  status?: MessageStatus
  attachments?: MessageAttachment[]
  artifactIds?: string[]
  artifact_ids?: string[]
  reportId?: string | null
  report_id?: string | null
  query?: string | null
  answer?: string | null
}

export interface Thread {
  /** Canonical API key. ``id`` is normalized as a convenience for UI code. */
  threadId: string
  thread_id?: string
  id: string
  title: string
  createdAt?: string
  created_at?: string
  updatedAt?: string
  updated_at?: string
  messageCount?: number
  message_count?: number
  activeRunId?: string | null
  active_run_id?: string | null
  runIds: string[]
  run_ids?: string[]
}

export interface ThreadDetail extends Thread {
  messages: ThreadMessage[]
  runs: ThreadRun[]
  artifacts: ResearchArtifact[]
}

export interface ThreadRun {
  runId: string
  run_id?: string
  status: RunStatus
  provider: string
  model: string
  currentNode?: string | null
  current_node?: string | null
  answer?: string | null
  reportId?: string | null
  report_id?: string | null
  error?: string | null
  createdAt?: string
  created_at?: string
  updatedAt?: string
  updated_at?: string
}

export interface ResearchArtifact {
  id: string
  kind: string
  title: string
  summary?: string
  reportId?: string | null
  report_id?: string | null
}

export interface ThreadInput {
  title?: string
}

export interface SendMessageInput {
  content: string
  paperInputs?: string[]
  paper_inputs?: string[]
  files?: File[]
}

export type MessageInput = SendMessageInput

export interface GitHubRepo {
  name: string
  url: string
  stars?: number
  framework?: string
  isOfficial?: boolean
  is_official?: boolean
  description?: string
}

export interface RunInput {
  query: string
  paperInputs: string[]
  files: File[]
}

export interface PublicProviderConfig {
  provider: string
  model: string
  configured: boolean
  apiBase?: string
}

export interface AppConfig {
  provider: string
  model: string
  configured: boolean
  providers?: PublicProviderConfig[]
  apiBase?: string
}

export interface RunResult {
  answer?: string | null
  reportId?: string | null
  report_id?: string | null
  report?: string | null
  finalReport?: string | null
  final_report?: string | null
  papers?: PaperSource[]
  selectedPapers?: PaperSource[]
  selected_papers?: PaperSource[]
  benchmark?: string | null
  benchmarkMatrix?: string | null
  benchmark_matrix?: string | null
  comparisonArtifact?: ComparisonArtifact | null
  comparison_artifact?: ComparisonArtifact | null
  bibtex?: string | null
  bibtexEntries?: string[]
  bibtex_entries?: string[]
  metadata?: Record<string, unknown>
}

export interface RunSnapshot {
  runId: string
  run_id?: string
  threadId?: string
  thread_id?: string
  answer?: string | null
  input: {
    query: string
    paperInputs: string[]
    paper_inputs?: string[]
    fileNames?: string[]
    pdfNames?: string[]
    pdf_names?: string[]
    files?: string[]
  }
  provider: string
  model: string
  status: RunStatus
  currentNode?: string | null
  current_node?: string | null
  error?: string | null
  errorMessage?: string | null
  error_message?: string | null
  progress: number
  traceEvents: TraceEvent[]
  trace_events?: TraceEvent[]
  trace?: TraceEvent[]
  seq?: number
  result?: RunResult | null
  startedAt?: string
  started_at?: string
  createdAt?: string
  created_at?: string
  updatedAt?: string
  updated_at?: string
}

export interface ReportSummary {
  id: string
  title: string
  createdAt?: string
  created_at?: string
  updatedAt?: string
  updated_at?: string
  size?: number
  paperCount?: number
  paper_count?: number
  status?: RunStatus
}

export interface ReportDocument extends ReportSummary {
  markdown: string
  content?: string
  report?: string
  papers?: PaperSource[]
  benchmark?: string | null
  bibtex?: string | null
}

export interface ServerRunEvent {
  runId?: string
  run_id?: string
  threadId?: string
  thread_id?: string
  seq?: number
  type?: TraceEventType | 'snapshot' | string
  traceEvents?: TraceEvent[]
  trace_events?: TraceEvent[]
  input?: RunSnapshot['input']
  provider?: string
  model?: string
  result?: RunResult | null
  status?: RunStatus | TraceStatus | string
  currentNode?: string | null
  current_node?: string | null
  error?: string | null
  error_message?: string | null
  progress?: number
  node?: string
  label?: string
  kind?: TraceKind | string
  summary?: string
  details?: string | Record<string, unknown> | null
  timestamp?: string
  links?: TraceLink[]
  delta?: string
  text?: string
  answer?: string | null
  content?: string
  message?: string | Record<string, unknown> | null
  messageId?: string
  message_id?: string
  role?: MessageRole | string
  artifactType?: string
  artifact_type?: string
  artifactId?: string
  artifact_id?: string
  artifact?: Record<string, unknown> | null
  facts?: TraceFacts | null
}

export interface NormalizedAssistantEvent {
  type: 'assistant_delta' | 'assistant_completed'
  runId: string
  threadId?: string
  seq?: number
  messageId?: string
  text: string
  timestamp?: string
}

export function readPaperId(paper: PaperSource): string | undefined {
  return paper.arxivId ?? paper.arxiv_id ?? paper.paperId ?? paper.paper_id ?? undefined
}

export function readPaperPdfUrl(paper: PaperSource): string | undefined {
  return paper.pdfUrl ?? paper.pdf_url ?? undefined
}

export function readPaperRepos(paper: PaperSource): GitHubRepo[] {
  return paper.githubRepos ?? paper.github_repos ?? []
}

export function readPaperNotes(paper: PaperSource): PMRLNotes | null {
  return paper.notes ?? null
}

export function readPaperBriefSummary(paper: PaperSource): string {
  const notes = readPaperNotes(paper)
  if (!notes) return ''
  return notes.briefSummary ?? notes.brief_summary ?? ''
}

export function readPaperSummaryCards(paper: PaperSource): SummaryCards | null {
  const candidates: unknown[] = [paper.summaryCards, paper.summary_cards, paper.summary]
  for (const candidate of candidates) {
    if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) continue
    return candidate as SummaryCards
  }
  return null
}

export function readPaperSubjects(paper: PaperSource): string[] {
  const raw = paper.subjects
  return Array.isArray(raw) ? raw.filter((value): value is string => typeof value === 'string' && Boolean(value.trim())).map((value) => value.trim()) : []
}

export function readPaperSourceQuality(paper: PaperSource): Record<string, unknown> {
  return paper.sourceQuality ?? paper.source_quality ?? {}
}

export function readPaperNotesQuality(paper: PaperSource): string {
  return paper.pmrlStatus ?? paper.pmrl_status ?? paper.notes?.status ?? paper.notesQuality ?? paper.notes_quality ?? 'unknown'
}

export function readThreadId(value: Partial<Thread> | Partial<ThreadDetail> | Partial<RunSnapshot> | Partial<ThreadMessage> | Partial<ServerRunEvent> | null | undefined): string | undefined {
  if (!value) return undefined
  return value.threadId ?? value.thread_id ?? undefined
}

export function readRunId(value: Partial<RunSnapshot> | Partial<ThreadMessage> | Partial<ServerRunEvent> | null | undefined): string | undefined {
  if (!value) return undefined
  return value.runId ?? value.run_id ?? undefined
}

export function readResultAnswer(result: RunResult | null | undefined): string {
  if (!result) return ''
  return result.answer ?? ''
}

export function readReportMarkdown(report: ReportDocument | RunResult | null | undefined): string {
  if (!report) return ''
  if ('markdown' in report && typeof report.markdown === 'string') return report.markdown
  if ('content' in report && typeof report.content === 'string') return report.content
  if ('report' in report && typeof report.report === 'string') return report.report
  if ('finalReport' in report && typeof report.finalReport === 'string') return report.finalReport
  if ('final_report' in report && typeof report.final_report === 'string') return report.final_report
  return ''
}

export function readResultPapers(result: RunResult | null | undefined): PaperSource[] {
  if (!result) return []
  return result.papers ?? result.selectedPapers ?? result.selected_papers ?? []
}

export function readBenchmark(result: RunResult | null | undefined): string {
  if (!result) return ''
  return result.benchmark ?? result.benchmarkMatrix ?? result.benchmark_matrix ?? ''
}

export function readComparisonArtifact(result: RunResult | null | undefined): ComparisonArtifact | null {
  if (!result) return null
  const artifact = result.comparisonArtifact ?? result.comparison_artifact
  if (!artifact || typeof artifact !== 'object') return null
  return artifact
}

export function readReportFilename(result: RunResult | null | undefined, fallback?: string): string | undefined {
  const metadata = result?.metadata
  if (metadata && typeof metadata === 'object') {
    for (const key of ['reportFilename', 'report_filename', 'filename', 'fileName', 'file_name']) {
      const value = metadata[key]
      if (typeof value === 'string' && value.trim()) return value.trim()
    }
  }
  return fallback?.trim() || undefined
}

export function readBibtex(result: RunResult | null | undefined): string {
  if (!result) return ''
  if (typeof result.bibtex === 'string') return result.bibtex
  const entries = result.bibtexEntries ?? result.bibtex_entries
  return entries?.join('\n\n') ?? ''
}
