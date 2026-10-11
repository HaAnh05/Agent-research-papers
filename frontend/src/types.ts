// Single place for the types shared with the backend (api.py camelCase DTOs).
// Every field that is not guaranteed by the contract is optional: renderers
// must hide what is missing and ignore what they do not know.

export type RunStatus = 'idle' | 'running' | 'success' | 'degraded' | 'error'

export interface TraceFacts {
  intent?: string
  resultCount?: number
  selectedCount?: number
  retryCount?: number
  paperCount?: number
  failedPdfCount?: number
  repositoryCount?: number
  bibtexCount?: number
  noteFallbackPaperIds?: string[]
  reportAvailable?: boolean
  // chat-UI stage facts (P2)
  tableCount?: number
  referenceCount?: number
  refsTotal?: number
  refsResolved?: number
  hardwareFoundCount?: number
  insightCount?: number
  evalSettings?: { proposed?: number; kept?: number; dropped?: number }
  seeds?: number
  pool?: number
  nodeCount?: number
  edgeCount?: number
  graphStatus?: string
  repoSource?: string
  source?: string
  sectionCount?: number
  reportId?: string
  // quick-summary facts (P6)
  keyResultCount?: number
  resultRowCount?: number
  /** Every LLM call of the step. Token fields exist only when the provider returned them. */
  llmCalls?: LlmCall[]
  paperFacts?: Array<Record<string, unknown>>
  [key: string]: unknown
}

export interface LlmCall {
  kind: string
  paperId?: string
  ms?: number
  inputTokens?: number
  outputTokens?: number
  reasoningTokens?: number
}

/**
 * step.started | step.updated | step.completed | assistant.delta | assistant.completed | answer.summary | run.completed | run.failed
 *
 * `answer.summary` carries the early answer (`answerSummary`, `reportReady: false`) and is sent again when the
 * graph or BibTeX arrives later. `run.completed` carries the final values (`reportReady: true`).
 */
export interface TraceEvent {
  runId: string
  seq: number
  type: string
  node: string
  label: string
  description?: string | null
  stage?: number | null
  parallelGroup?: string | null
  status?: string
  summary?: string
  details?: string | null
  facts?: TraceFacts | null
  delta?: string | null
  durationMs?: number | null
  timestamp?: string
  answerSummary?: AnswerSummary | null
  reportReady?: boolean | null
  citationGraph?: { similarity?: unknown; [key: string]: unknown } | null
  bibtex?: string | null
}

// ---- answerSummary (P3) ------------------------------------------------------

export interface PaperRef {
  paperId?: string
  arxivId?: string
  title?: string
  published?: string
  year?: number | string
  url?: string
}

export interface InsightItem {
  finding?: string
  why?: string
}

export interface MetricItem {
  name?: string
  meaning?: string
  higherIsBetter?: boolean
}

export interface ResultRow {
  setting?: string
  metric?: string
  method?: string
  baseline?: string
  delta?: string
  higherIsBetter?: boolean
}

export interface PaperCard {
  paperId?: string
  arxivId?: string
  title?: string
  year?: number | string
  published?: string
  intuition?: string
  keyResult?: string
  url?: string
  blocks?: SummaryBlock[]
}

/** Blocks are rendered in array order; a block with an unknown `type` is ignored. */
export type SummaryBlock =
  | { type: 'insights'; items?: InsightItem[] }
  | { type: 'problem'; text?: string }
  | { type: 'intuition'; text?: string }
  | { type: 'keyResults'; items?: Array<string | InsightItem>; text?: string }
  | { type: 'metrics'; items?: MetricItem[] }
  | { type: 'setup'; items?: string[]; text?: string; hardware?: string; missing?: string[] }
  | { type: 'results'; grounded?: boolean; source?: string[]; methodName?: string; rows?: ResultRow[] }
  | { type: 'caveats'; text?: string; items?: string[] }
  | { type: 'foundations'; items?: string[] }
  | { type: 'takeaway'; text?: string }
  | { type: 'paperCards'; items?: PaperCard[] }
  | { type: string; [key: string]: unknown }

export interface SummaryActions {
  reportId?: string | null
  bibtexCount?: number
  graphNodes?: number
}

export interface AnswerSummary {
  kind?: 'single' | 'multi' | string
  paper?: PaperRef
  papers?: PaperRef[]
  lead?: string
  tldr?: string
  blocks?: SummaryBlock[]
  actions?: SummaryActions
  durationSec?: number
  /** Optional suggested follow-up questions. The UI derives its own when absent. */
  followUps?: string[]
}

// ---- similarity graph ---------------------------------------------------------

export interface GraphNode {
  id: string
  arxivId?: string | null
  title: string
  year?: number | null
  citedByCount?: number | null
  firstAuthor?: string | null
  abstract?: string | null
  similarity?: number | null
  role?: 'origin' | 'similar' | string
  url?: string | null
  /** Number of map papers that cite / are cited by this work (prior and derivative lists). */
  support?: number | null
}

export interface GraphEdge {
  source: string
  target: string
  weight: number
}

export interface SimilarityGraph {
  nodes: GraphNode[]
  edges: GraphEdge[]
  priorWorks: GraphNode[]
  derivativeWorks: GraphNode[]
  stats?: { seeds?: number; pool?: number; status?: string; [key: string]: unknown }
}

// ---- run snapshot ---------------------------------------------------------------

export interface PaperResult {
  paperId?: string
  arxivId?: string | null
  title?: string
  repository?: { url?: string; source?: string } | null
}

export interface RunResult {
  report?: string | null
  reportId?: string | null
  /** False while report.md is still being written; `answerSummary` is already available then. */
  reportReady?: boolean
  bibtex?: string | null
  answer?: string | null
  answerSummary?: AnswerSummary | null
  citationGraph?: { similarity?: unknown; [key: string]: unknown } | null
  papers?: PaperResult[]
}

export interface RunSnapshot {
  runId: string
  threadId?: string | null
  status: RunStatus
  answer?: string | null
  error?: string | null
  seq: number
  traceEvents: TraceEvent[]
  result?: RunResult
  startedAt?: string | null
  updatedAt?: string | null
}

// ---- view model ------------------------------------------------------------------

export type StepStatus = 'wait' | 'running' | 'completed' | 'error' | 'skipped'

export interface StepView {
  key: string
  node: string
  label: string
  description?: string
  stage?: number
  parallelGroup?: string
  status: StepStatus
  facts: TraceFacts
  /** Server-measured duration; shown once the step is completed. */
  durationMs?: number
  /** Client time (ms) the step started; the running timer counts from here. */
  startedMs?: number
}

export interface RunView {
  runId: string
  status: RunStatus
  steps: StepView[]
  answerText: string
  lastSeq: number
  startedMs?: number
  endedMs?: number
  error?: string
  /** True once report.md exists (`reportReady` of the API). */
  reportReady: boolean
  result?: RunResult
  summary?: AnswerSummary
  graph?: SimilarityGraph
}

export interface Turn {
  id: string
  userText: string
  attachment?: string
  run?: RunView
  /** Set when the message could not be sent (409, network). */
  sendError?: string
}
