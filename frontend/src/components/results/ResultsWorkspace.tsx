import { useEffect, useMemo, useState, type CSSProperties, type KeyboardEvent } from 'react'
import {
  ArrowUpRight,
  BookOpen,
  Clipboard,
  ClipboardCheck,
  Download,
  ExternalLink,
  FileCode2,
  Github,
  Plus,
} from 'lucide-react'
import { MarkdownContent, titleFromMarkdown, withoutFirstMarkdownHeading } from '../MarkdownContent'
import { downloadReport } from '../../lib/api'
import {
  readBenchmark,
  readComparisonArtifact,
  readPaperId,
  readPaperNotesQuality,
  readPaperPdfUrl,
  readPaperRepos,
  readPaperSubjects,
  readPaperSummaryCards,
  readPaperSourceQuality,
  readReportFilename,
  readReportMarkdown,
  readResultPapers,
  type ComparisonArtifact,
  type ComparisonPaper,
  type ComparisonRow,
  type PaperSource,
  type RunResult,
  type RunSnapshot,
  type SummaryCards,
  type SummaryCardStatus,
  type SummaryCardStatusField,
  type TraceEvent,
} from '../../types'
import './results.css'

export interface ResultsWorkspaceProps {
  snapshot: RunSnapshot
  events: TraceEvent[]
  onNewResearch: () => void
}

type ResultTab = 'summary' | 'comparison' | 'report'

type FactsRecord = Record<string, unknown>

const STAGE_LABELS: Record<string, string> = {
  router: 'Understand request',
  search_papers: 'Discover relevant papers',
  eval_search: 'Discover relevant papers',
  refine_query: 'Discover relevant papers',
  read_paper: 'Read source PDFs',
  web_enrich: 'Enrich sources',
  write_notes: 'Write structured notes',
  compare_benchmark: 'Compare selected papers',
  final_report: 'Generate final report',
  error_handler: 'Workflow error',
}

function asRecord(value: unknown): FactsRecord | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as FactsRecord : null
}

function eventFacts(event: TraceEvent): FactsRecord | null {
  const record = event as TraceEvent & { facts?: unknown }
  const direct = asRecord(record.facts)
  if (direct) return direct
  const details = asRecord(event.details)
  return asRecord(details?.facts)
}

function mergedFacts(events: TraceEvent[]): FactsRecord {
  const result: FactsRecord = {}
  for (const event of orderedEvents(events)) {
    const facts = eventFacts(event)
    if (facts) Object.assign(result, facts)
  }
  return result
}

function orderedEvents(events: TraceEvent[]): TraceEvent[] {
  return [...events].sort((a, b) => a.seq - b.seq || a.timestamp.localeCompare(b.timestamp))
}

function stringList(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value.filter((item): item is string => typeof item === 'string' && Boolean(item.trim())).map((item) => item.trim())
}

function resultFrom(snapshot: RunSnapshot): RunResult | null {
  return snapshot.result ?? null
}

function paperKey(paper: PaperSource, index: number): string {
  return readPaperId(paper) ?? `${paper.title}-${index}`
}

function paperDisplayTitle(paper: PaperSource): string {
  const title = paper.title?.trim()
  const paperId = readPaperId(paper)
  if (title && !/^untitled paper$/i.test(title) && !/^arxiv\s+/i.test(title) && title.toLowerCase() !== paperId?.toLowerCase()) return title
  return readPaperId(paper) ?? title ?? 'Untitled paper'
}

function hasOfficialPaperTitle(paper: PaperSource): boolean {
  const title = paper.title?.trim()
  const paperId = readPaperId(paper)
  return Boolean(title && !/^untitled paper$/i.test(title) && !/^arxiv\s+/i.test(title) && title.toLowerCase() !== paperId?.toLowerCase())
}

type SummaryCardKey = 'tldr' | 'problem' | 'method' | 'keyResults' | 'whyItMatters'

const SUMMARY_CARD_DEFINITIONS: Array<{ key: SummaryCardKey; label: string; className: string }> = [
  { key: 'tldr', label: 'TL;DR', className: 'results-summary-card--tldr' },
  { key: 'problem', label: 'Problem', className: 'results-summary-card--problem' },
  { key: 'method', label: 'Method', className: 'results-summary-card--method' },
  { key: 'keyResults', label: 'Key Results', className: 'results-summary-card--results' },
  { key: 'whyItMatters', label: 'Why it matters', className: 'results-summary-card--why' },
]

type ValidatedSummaryCards = {
  cards: Record<SummaryCardKey, string>
  statuses: Partial<Record<SummaryCardKey, SummaryCardStatus>>
  reasons: Partial<Record<SummaryCardKey, string>>
  issues: string[]
}

function summaryWordCount(value: string): number {
  return value.trim().split(/\s+/).filter(Boolean).length
}

function normalizeSummaryText(value: unknown): string {
  if (typeof value !== 'string') return ''
  return value.replace(/<br\s*\/?\s*>/gi, '\n').trim()
}

function summaryCardValue(summary: SummaryCards | null, key: SummaryCardKey): string {
  if (!summary) return ''
  if (key === 'keyResults') return normalizeSummaryText(summary.keyResults ?? summary.key_results)
  if (key === 'whyItMatters') return normalizeSummaryText(summary.whyItMatters ?? summary.why_it_matters)
  return normalizeSummaryText(summary[key])
}

function summaryCardStatus(summary: SummaryCards | null, key: SummaryCardKey): SummaryCardStatus | undefined {
  const statuses = summary?.cardStatuses ?? summary?.card_statuses
  if (!statuses || typeof statuses !== 'object' || Array.isArray(statuses)) return undefined

  const statusKeys: SummaryCardStatusField[] = key === 'keyResults'
    ? ['keyResults', 'key_results']
    : key === 'whyItMatters'
      ? ['whyItMatters', 'why_it_matters']
      : [key]

  for (const statusKey of statusKeys) {
    if (!Object.prototype.hasOwnProperty.call(statuses, statusKey)) continue
    const rawStatus = (statuses as Record<string, unknown>)[statusKey]
    if (typeof rawStatus !== 'string') return 'unknown'
    const normalized = rawStatus.trim().toLowerCase()
    if (normalized === 'complete' || normalized === 'unsupported' || normalized === 'missing' || normalized === 'invalid' || normalized === 'unknown') {
      return normalized
    }
    return 'unknown'
  }
  return undefined
}

function summaryCardReason(summary: SummaryCards | null, key: SummaryCardKey): string | undefined {
  const reasons = summary?.cardReasons ?? summary?.card_reasons
  if (!reasons || typeof reasons !== 'object' || Array.isArray(reasons)) return undefined
  const reasonKeys: SummaryCardStatusField[] = key === 'keyResults'
    ? ['keyResults', 'key_results']
    : key === 'whyItMatters'
      ? ['whyItMatters', 'why_it_matters']
      : [key]
  for (const reasonKey of reasonKeys) {
    const raw = (reasons as Record<string, unknown>)[reasonKey]
    if (typeof raw === 'string' && raw.trim()) return raw.trim()
  }
  return undefined
}

function normalizedSummaryForComparison(value: string): string {
  return value.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, ' ').replace(/\s+/g, ' ').trim()
}

function summaryCardsForPaper(paper: PaperSource): ValidatedSummaryCards {
  const summary = readPaperSummaryCards(paper)
  const cards = Object.fromEntries(SUMMARY_CARD_DEFINITIONS.map(({ key }) => [key, summaryCardValue(summary, key)])) as Record<SummaryCardKey, string>
  const statuses: Partial<Record<SummaryCardKey, SummaryCardStatus>> = {}
  const reasons: Partial<Record<SummaryCardKey, string>> = {}
  const issues: string[] = []
  const summaryStatus = String(summary?.status ?? '').trim().toLowerCase()
  if (!summary) issues.push('The structured summary is unavailable for this run.')
  if (summaryStatus === 'invalid') issues.push('The structured summary was marked invalid.')
  if (summaryStatus === 'missing' || summaryStatus === 'unknown') issues.push('The structured summary is unavailable for this run.')
  if (summaryStatus === 'partial') issues.push('Some structured summary sections are unavailable.')
  if (summary && !Object.values(cards).some(Boolean)) issues.push('The structured summary did not contain any displayable sections.')
  if (summaryStatus && summaryStatus !== 'complete' && summaryStatus !== 'partial') {
    for (const key of Object.keys(cards) as SummaryCardKey[]) {
      cards[key] = ''
      statuses[key] = summaryStatus === 'missing' || summaryStatus === 'invalid' || summaryStatus === 'unknown'
        ? summaryStatus
        : 'unknown'
    }
    return { cards, statuses, reasons, issues: [...new Set(issues)] }
  }

  for (const { key, label } of SUMMARY_CARD_DEFINITIONS) {
    const status = summaryCardStatus(summary, key)
    const reason = summaryCardReason(summary, key)
    if (status) statuses[key] = status
    if (reason) reasons[key] = reason
    if (status && status !== 'complete') {
      cards[key] = ''
      issues.push(`${label} was marked ${status} and was omitted.`)
    } else if (status === 'complete' && !cards[key]) {
      issues.push(`${label} was marked complete but did not contain displayable text.`)
    }
  }

  const hasCardStatuses = Boolean(summary?.cardStatuses ?? summary?.card_statuses)
  if (!hasCardStatuses) {
    const seen = new Map<string, SummaryCardKey>()
    for (const { key, label } of SUMMARY_CARD_DEFINITIONS) {
      const value = cards[key]
      if (!value) continue
      const normalized = normalizedSummaryForComparison(value)
      const previous = [...seen.entries()].find(([other]) => other === normalized || (summaryWordCount(value) >= 12 && (other.includes(normalized) || normalized.includes(other))))
      if (previous) {
        cards[key] = ''
        issues.push(`${label} repeated another summary section and was omitted.`)
        continue
      }
      seen.set(normalized, key)
    }
  }
  return { cards, statuses, reasons, issues: [...new Set(issues)] }
}

function paperScore(paper: PaperSource): number | null {
  const record = paper as unknown as FactsRecord
  for (const key of ['relevanceScore', 'relevance_score', 'score', 'relevance']) {
    const value = record[key]
    if (typeof value === 'number' && Number.isFinite(value) && value >= 0) return value
  }
  return null
}

function scoreLabel(score: number): string {
  if (score <= 1) return `${Math.round(score * 100)}% relevance`
  return `${score} relevance`
}

function safeExternalHref(value: unknown): string | undefined {
  if (typeof value !== 'string' || !value.trim()) return undefined
  try {
    const url = new URL(value)
    return url.protocol === 'https:' ? url.toString() : undefined
  } catch {
    return undefined
  }
}

function arxivHref(paper: PaperSource): string | undefined {
  const rawId = paper.arxivId ?? paper.arxiv_id
  const id = typeof rawId === 'string' ? rawId.replace(/^arxiv:/i, '').trim() : ''
  // Keep local paper keys and arbitrary identifiers out of generated links.
  const isArxivId = /^(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z][\w.-]*(?:\.[A-Z]{2})?\/\d{7}(?:v\d+)?)$/i.test(id)
  if (!id || !isArxivId) return undefined
  return `https://arxiv.org/abs/${id}`
}

function safeArxivHref(value: unknown): string | undefined {
  const href = safeExternalHref(value)
  if (!href) return undefined
  try {
    const hostname = new URL(href).hostname.toLowerCase()
    return new Set(['arxiv.org', 'www.arxiv.org', 'export.arxiv.org']).has(hostname) ? href : undefined
  } catch {
    return undefined
  }
}

function safeGithubHref(value: unknown): string | undefined {
  const href = safeExternalHref(value)
  if (!href) return undefined
  try {
    const hostname = new URL(href).hostname.toLowerCase()
    return hostname === 'github.com' || hostname === 'www.github.com' ? href : undefined
  } catch {
    return undefined
  }
}

function sourceQualityWarnings(paper: PaperSource, fallbackIds: Set<string> = new Set()): string[] {
  const quality = readPaperSourceQuality(paper)
  const warnings: string[] = []
  const identity = paperIdentitySet(paper)
  if ([...identity].some((id) => fallbackIds.has(id))) warnings.push('PDF extraction used a fallback parser or section layout.')

  const coverageStatus = String(quality.coverageStatus ?? quality.coverage_status ?? '').trim().toLowerCase()
  if (coverageStatus && coverageStatus !== 'complete' && coverageStatus !== 'ok' && coverageStatus !== 'full') {
    const missing = quality.missingSections
    const missingSections = Array.isArray(missing) ? missing.filter((value): value is string => typeof value === 'string' && Boolean(value.trim())).join(', ') : ''
    warnings.push(missingSections ? `Source extraction is incomplete; missing sections: ${missingSections}.` : 'Source extraction is incomplete; section coverage could not be verified.')
  }

  for (const [key, rawValue] of Object.entries(quality)) {
    const value = String(rawValue ?? '').trim()
    if (!value) continue
    const normalized = value.toLowerCase()
    const keyLooksRelevant = /(warning|heading|parser|extract|section|quality|status)/i.test(key)
    const valueLooksProblematic = /(fallback|warn|fail|missing|unknown|partial|unavailable|error|degraded)/i.test(normalized)
    if (!keyLooksRelevant || !valueLooksProblematic) continue
    if (/(heading|section)/i.test(key) && /fallback/i.test(normalized)) warnings.push('Section heading extraction used a fallback.')
    else if (/(parser|extract)/i.test(key) && /(fallback|degraded|error)/i.test(normalized)) warnings.push('PDF extraction reported a parser warning.')
    else if (/(warning|error|fail|missing|unavailable)/i.test(key) || value.length > 18) warnings.push(value)
  }
  return [...new Set(warnings)]
}

function reportIdFrom(result: RunResult | null, facts: FactsRecord): string | undefined {
  const value = result?.reportId ?? result?.report_id ?? facts.reportId
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function isComparisonAvailable(result: RunResult | null, papers: PaperSource[], facts: FactsRecord): boolean {
  const artifact = readComparisonArtifact(result)
  if (artifact && artifact.available !== false && (artifact.papers?.length ?? 0) >= 2) return facts.comparisonAvailable !== false
  const benchmark = readBenchmark(result).trim()
  if (!benchmark || facts.comparisonAvailable !== true) return false
  return comparisonPapers(papers, facts).length >= 2
}

function paperIdentitySet(paper: PaperSource): Set<string> {
  const ids = [readPaperId(paper), paper.paperId, paper.paper_id].filter((value): value is string => Boolean(value?.trim()))
  return new Set(ids.map((value) => value.trim().toLowerCase().replace(/^arxiv:/, '')))
}

function comparisonPapers(papers: PaperSource[], facts: FactsRecord): PaperSource[] {
  const comparedIds = stringList(facts.comparedPaperIds).map((id) => id.toLowerCase().replace(/^arxiv:/, ''))
  if (comparedIds.length < 2) return []
  const selected = papers.filter((paper) => {
    const identities = paperIdentitySet(paper)
    return comparedIds.some((id) => identities.has(id))
  })
  return selected.length >= 2 ? selected : []
}

function isFallbackPaper(paper: PaperSource, fallbackIds: Set<string>): boolean {
  return [...paperIdentitySet(paper)].some((id) => fallbackIds.has(id))
}

function formatTimestamp(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(date)
}

function stageLabel(event: TraceEvent): string {
  return STAGE_LABELS[event.node] ?? (event.label || 'Research activity')
}

function safeSummary(value: string): string {
  return value
    .replace(/(?:sk-|api[_-]?key|token|secret)[=: ]+[^\s,;]+/gi, '[redacted]')
    .replace(/(?:[A-Za-z]:\\|\/home\/|\/Users\/|\/tmp\/)[^\s,;]+/g, '[private path]')
    .trim()
    .slice(0, 220)
}

function activityEvents(events: TraceEvent[]): TraceEvent[] {
  const unique = new Map<string, TraceEvent>()
  const completionIndex = new Map<string, string>()
  const generationByNode = new Map<string, number>()
  const retryByNode = new Map<string, number>()
  for (const event of orderedEvents(events)) {
    if (event.type.startsWith('assistant') || event.type === 'thinking_step') continue
    if (!(event.type.startsWith('run.') || event.type.startsWith('step.') || event.type.startsWith('node.') || event.type === 'artifact_created' || event.type.startsWith('tool.'))) continue
    const facts = eventFacts(event)
    const retryCount = typeof facts?.retryCount === 'number' ? facts.retryCount : undefined
    const previousRetry = retryByNode.get(event.node)
    const lifecycleBoundary = event.type.endsWith('.started') || event.status === 'retrying' || event.status === 'failed' || event.status === 'error' || event.status === 'skipped'
    if (lifecycleBoundary || (retryCount !== undefined && previousRetry !== undefined && retryCount > previousRetry)) {
      generationByNode.set(event.node, (generationByNode.get(event.node) ?? 0) + 1)
    }
    if (retryCount !== undefined) retryByNode.set(event.node, retryCount)
    const isCompletion = event.status === 'completed' || event.status === 'failed' || event.status === 'error' || event.status === 'skipped' || event.type.endsWith('.completed')
    if (isCompletion) {
      // Terminal run events live in their own namespace: `run.completed`
      // must never replace the final node's `step.completed` row (or vice
      // versa) — both rows carry distinct information.  All other channel
      // variants (`step.*`, `node.*`, `tool.*`) still collapse per node so a
      // retried or dual-channel completion renders exactly once.
      const family = event.type.startsWith('run.') ? 'run' : 'step'
      const completionGeneration = `${family}:${event.node}:${generationByNode.get(event.node) ?? 0}`
      const previousKey = completionIndex.get(completionGeneration)
      if (previousKey) unique.delete(previousKey)
      completionIndex.set(completionGeneration, `${event.seq}:${event.type}:${event.node}`)
    }
    const key = `${event.seq}:${event.type}:${event.node}`
    unique.set(key, event)
  }
  return [...unique.values()]
}

function activityCopy(event: TraceEvent): string {
  const summary = safeSummary(event.summary || '')
  if (summary) return summary
  if (event.type.endsWith('started') || event.status === 'running') return `${stageLabel(event)} started`
  if (event.type.endsWith('completed') || event.status === 'completed') return `${stageLabel(event)} completed`
  if (event.status === 'failed' || event.status === 'error') return `${stageLabel(event)} failed`
  if (event.status === 'skipped') return `${stageLabel(event)} skipped`
  return `${stageLabel(event)} updated`
}

function durationLabel(durationMs: number | null | undefined): string | null {
  if (durationMs === null || durationMs === undefined || !Number.isFinite(durationMs) || durationMs < 0) return null
  if (durationMs >= 60_000) return `${(durationMs / 60_000).toFixed(1)}m`
  if (durationMs >= 1000) return `${(durationMs / 1000).toFixed(1)}s`
  return `${Math.round(durationMs)}ms`
}

function parserWarningsForEvent(event: TraceEvent): string[] {
  const facts = eventFacts(event)
  const details = asRecord(event.details)
  const values = [facts?.parserWarning, ...(Array.isArray(facts?.parserWarnings) ? facts.parserWarnings : []), ...stringList(details?.parserWarnings), typeof details?.parserWarning === 'string' ? details.parserWarning : undefined]
  return [...new Set(values.filter((value): value is string => typeof value === 'string' && Boolean(value.trim())).map((value) => value.trim()))]
}

function summaryReasonsForEvent(event: TraceEvent): string[] {
  const facts = eventFacts(event)
  const raw = asRecord(facts?.summaryCardReasons)
  if (!raw) return []
  return Object.entries(raw)
    .filter((entry): entry is [string, string] => typeof entry[1] === 'string' && Boolean(entry[1].trim()))
    .map(([card, reason]) => safeSummary(`${card}: ${reason}`))
    .filter(Boolean)
    .sort()
}

function EmptyPanel({ children }: { children: React.ReactNode }) {
  return <div className="results-empty">{children}</div>
}

function SourceRail({ papers, selectedIndex, onSelect }: { papers: PaperSource[]; selectedIndex: number; onSelect: (index: number) => void }) {
  return (
    <aside className="results-sources" aria-labelledby="results-sources-heading">
      <div className="results-sources__heading">
        <h2 id="results-sources-heading">Sources</h2>
        {papers.length ? <span className="results-sources__count">{papers.length}</span> : null}
      </div>
      {papers.length ? (
        <nav className="results-sources__list" aria-label="Research sources">
          {papers.map((paper, index) => <SourceItem key={paperKey(paper, index)} paper={paper} index={index} selected={selectedIndex === index} onSelect={() => onSelect(index)} />)}
        </nav>
      ) : <p className="results-sources__empty">No finalized sources</p>}
    </aside>
  )
}

function SourceItem({ paper, index, selected, onSelect }: { paper: PaperSource; index: number; selected: boolean; onSelect: () => void }) {
  const score = paperScore(paper)
  const style = score !== null ? { '--result-relevance': `${Math.min(100, score <= 1 ? score * 100 : score)}%` } as CSSProperties : undefined
  return (
    <button
      type="button"
      className={`results-source-item${selected ? ' is-selected' : ''}${score !== null ? ' has-score' : ''}`}
      aria-pressed={selected}
      onClick={onSelect}
      style={style}
    >
      <span className="results-source-item__rank">#{String(index + 1).padStart(2, '0')}</span>
      <span className="results-source-item__title">{paperDisplayTitle(paper)}</span>
      {score !== null ? <span className="results-source-item__score">{scoreLabel(score)}</span> : null}
    </button>
  )
}

function MobileSourceSelect({ papers, selectedIndex, onSelect }: { papers: PaperSource[]; selectedIndex: number; onSelect: (index: number) => void }) {
  if (!papers.length) return null
  return (
    <label className="results-source-select">
      <span>Select paper</span>
      <select value={selectedIndex} onChange={(event) => onSelect(Number(event.target.value))}>
        {papers.map((paper, index) => <option value={index} key={paperKey(paper, index)}>#{String(index + 1).padStart(2, '0')} · {paperDisplayTitle(paper)}</option>)}
      </select>
    </label>
  )
}

function SourceActions({ paper }: { paper: PaperSource }) {
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'failed'>('idle')
  const arxiv = arxivHref(paper)
  const rawPdf = readPaperPdfUrl(paper)
  const pdf = safeArxivHref(rawPdf)
  const pdfUnavailable = Boolean(rawPdf && !pdf)
  const repos = readPaperRepos(paper).map((repo) => ({ ...repo, href: safeGithubHref(repo.url) }))
  const validRepos = repos.filter((repo): repo is typeof repo & { href: string } => Boolean(repo.href))
  const invalidRepoCount = repos.length - validRepos.length
  const quality = readPaperSourceQuality(paper)
  const rawCodeUrls = quality.codeUrls ?? quality.code_urls
  const codeUrls = Array.isArray(rawCodeUrls) ? rawCodeUrls.filter((value): value is string => typeof value === 'string' && Boolean(value.trim())) : []
  const verifiedCodeUrls = codeUrls.map((url) => safeGithubHref(url)).filter((url): url is string => Boolean(url)).filter((url, index, values) => values.indexOf(url) === index && !validRepos.some((repo) => repo.href === url))
  const bibtex = typeof paper.bibtex === 'string' ? paper.bibtex.trim() : ''

  const copyBibtex = async () => {
    if (!bibtex || !navigator.clipboard?.writeText) {
      setCopyState('failed')
      return
    }
    try {
      await navigator.clipboard.writeText(bibtex)
      setCopyState('copied')
    } catch {
      setCopyState('failed')
    }
  }

  return (
    <div className="results-source-actions" aria-label="Source actions">
      {arxiv ? <a href={arxiv} target="_blank" rel="noopener noreferrer" aria-label={`Open arXiv page for ${paperDisplayTitle(paper)} (arxiv.org)`} className="results-action-link">arXiv <ArrowUpRight size={13} aria-hidden="true" /></a> : null}
      {pdf ? <a href={pdf} target="_blank" rel="noopener noreferrer" aria-label={`Open PDF for ${paperDisplayTitle(paper)} (arxiv.org)`} className="results-action-link">PDF <ExternalLink size={13} aria-hidden="true" /></a> : null}
      {validRepos.map((repo) => <a href={repo.href} target="_blank" rel="noopener noreferrer" aria-label={`Open related repository ${repo.name} (github.com)`} className="results-action-link" key={repo.url}><span>Related repository</span><Github size={13} aria-hidden="true" /></a>)}
      {verifiedCodeUrls.map((url) => <a href={url} target="_blank" rel="noopener noreferrer" aria-label={`Open source code (${url})`} className="results-action-link" key={url}><span>Code</span><Github size={13} aria-hidden="true" /></a>)}
      {pdfUnavailable ? <span className="results-link-warning" role="note"><ExternalLink size={13} aria-hidden="true" />PDF link unavailable; destination was not verified.</span> : null}
      {invalidRepoCount ? <span className="results-link-warning" role="note"><Github size={13} aria-hidden="true" />{invalidRepoCount === 1 ? 'Repository link unavailable; destination was not verified.' : 'Some repository links were unavailable; destinations were not verified.'}</span> : null}
      {bibtex ? <button type="button" className="results-action-link" onClick={() => void copyBibtex()}><span>{copyState === 'copied' ? 'BibTeX copied' : 'Copy BibTeX'}</span>{copyState === 'copied' ? <ClipboardCheck size={13} aria-hidden="true" /> : <Clipboard size={13} aria-hidden="true" />}</button> : <span className="results-action-unavailable"><FileCode2 size={13} aria-hidden="true" />BibTeX unavailable</span>}
      {copyState === 'failed' ? <span className="results-action-status" role="status">Could not copy BibTeX.</span> : null}
    </div>
  )
}

function PaperMeta({ paper }: { paper: PaperSource }) {
  const authors = paper.authors?.filter(Boolean) ?? []
  const id = readPaperId(paper)
  const sourceType = paper.sourceType ?? paper.source_type
  const published = paper.publishedDate ?? paper.published_date ?? paper.published
  const submitted = paper.submittedDate ?? paper.submitted_date ?? paper.submitted
  const subjects = readPaperSubjects(paper)
  const metadataStatus = String(paper.metadataStatus ?? paper.metadata_status ?? '').toLowerCase()
  const metadataUnavailable = metadataStatus === 'missing' || metadataStatus === 'unknown' || !hasOfficialPaperTitle(paper)
  return (
    <div className="results-paper-meta">
      {authors.length ? <span className="results-paper-meta__authors">{authors.join(', ')}</span> : <span className="results-paper-meta__missing">Authors unavailable</span>}
      {published ? <span><strong>Published</strong> {published}</span> : submitted ? <span><strong>Submitted</strong> {submitted}</span> : <span className="results-paper-meta__missing">Date unavailable</span>}
      {id ? <code>{id}</code> : null}
      {sourceType === 'local_pdf' ? <span>Supplied PDF</span> : null}
      {subjects.length ? <span className="results-paper-meta__subjects" aria-label="Subjects">{subjects.map((subject) => <span className="results-subject-chip" key={subject}>{subject}</span>)}</span> : null}
      {sourceType !== 'local_pdf' && metadataUnavailable ? <p className="results-metadata-warning" role="status"><BookOpen size={14} aria-hidden="true" />Official arXiv metadata is incomplete; available identifiers are shown.</p> : null}
    </div>
  )
}

function unavailableSummaryCopy(label: string, status?: SummaryCardStatus, reason?: string): string {
  if (label === 'Key Results' && (status === 'unsupported' || reason?.startsWith('number_not_in_source:'))) {
    return 'No source-supported result was available.'
  }
  if (status === 'unsupported' || reason === 'no_source_overlap') return 'No source-supported summary was available.'
  if (status === 'invalid') return 'This summary section did not pass validation.'
  if (status === 'missing') return 'No summary was generated for this section.'
  return 'This summary section is unavailable.'
}

function SummaryCard({ label, className, value, status, reason }: { label: string; className: string; value: string; status?: SummaryCardStatus; reason?: string }) {
  return <article className={`results-summary-card ${className}`}><h3>{label}</h3>{value ? <MarkdownContent content={value} /> : <p className="results-summary-card__missing">{unavailableSummaryCopy(label, status, reason)}</p>}</article>
}

function SummaryPanel({ paper, index, total, isFallback, sourceWarnings }: { paper: PaperSource | undefined; index: number; total: number; isFallback: boolean; sourceWarnings: string[] }) {
  if (!paper) return <EmptyPanel>No finalized paper output was returned for this run.</EmptyPanel>
  const validatedSummary = summaryCardsForPaper(paper)
  const notesQuality = readPaperNotesQuality(paper)
  return (
    <section className="results-summary" aria-labelledby="results-paper-heading">
      <p className="results-paper-position">Paper {index + 1} of {total}</p>
      <h2 id="results-paper-heading" className="results-paper-title">{paperDisplayTitle(paper)}</h2>
      <PaperMeta paper={paper} />
      {isFallback || notesQuality === 'fallback' ? <p className="results-note-status"><BookOpen size={14} aria-hidden="true" /> PMRL fallback returned for this source.</p> : null}
      {sourceWarnings.length ? <div className="results-source-warning" role="note"><BookOpen size={14} aria-hidden="true" /><span>{sourceWarnings.join(' ')}</span></div> : null}
      <SourceActions paper={paper} />
      <section className="results-summary-cards" aria-labelledby="results-summary-heading">
        <div className="results-pmrl-heading"><h2 id="results-summary-heading">Summary</h2><span>Source-grounded overview</span></div>
        <div className="results-summary-card-grid">
          {SUMMARY_CARD_DEFINITIONS.map(({ key, label, className }) => <SummaryCard key={key} label={label} className={className} value={validatedSummary.cards[key]} status={validatedSummary.statuses[key]} reason={validatedSummary.reasons[key]} />)}
        </div>
        {validatedSummary.issues.length ? <p className="results-summary-quality" role="status"><BookOpen size={14} aria-hidden="true" />Some summary sections are unavailable because the generated content did not pass validation checks.</p> : null}
      </section>
    </section>
  )
}

function comparisonPaperId(paper: ComparisonPaper, index: number): string {
  return (paper.paperId ?? paper.paper_id ?? paper.title ?? `paper-${index + 1}`).trim().toLowerCase().replace(/^arxiv:/, '')
}

function comparisonPaperTitle(paper: ComparisonPaper, index: number): string {
  return paper.title?.trim() || paper.paperId || paper.paper_id || `Paper ${index + 1}`
}

function comparisonValue(row: ComparisonRow, paper: ComparisonPaper, index: number): string {
  const paperId = comparisonPaperId(paper, index)
  const entry = Object.entries(row.values ?? {}).find(([key]) => key.trim().toLowerCase().replace(/^arxiv:/, '') === paperId)
  return entry?.[1]?.trim() ?? ''
}

function comparisonQuote(row: ComparisonRow, paper: ComparisonPaper, index: number): string {
  const paperId = comparisonPaperId(paper, index)
  const quotes = row.sourceQuotes ?? row.source_quotes ?? {}
  const entry = Object.entries(quotes).find(([key]) => key.trim().toLowerCase().replace(/^arxiv:/, '') === paperId)
  return entry?.[1]?.trim() ?? ''
}

function isComparableRow(row: ComparisonRow, papers: ComparisonPaper[]): boolean {
  if (!row.comparable || !row.metric.trim() || !row.dataset.trim() || !row.unit.trim() || papers.length < 2) return false
  return papers.every((paper, index) => Boolean(comparisonValue(row, paper, index) && comparisonQuote(row, paper, index)))
}

function ComparisonMetricList({ paper }: { paper: ComparisonPaper }) {
  if (!paper.metrics?.length) return <span className="results-comparison-muted">No source-backed metric recorded.</span>
  return <ul className="results-comparison-metrics">{paper.metrics.map((metric, index) => <li key={`${metric.metric}-${metric.dataset}-${index}`}><strong>{metric.metric || 'Metric'}</strong>{metric.dataset ? <span> · {metric.dataset}</span> : null}{metric.unit ? <span> · {metric.unit}</span> : null}{metric.value ? <span>: {metric.value}</span> : null}{(metric.sourceQuote ?? metric.source_quote)?.trim() ? <small>Source: {(metric.sourceQuote ?? metric.source_quote)?.trim()}</small> : null}</li>)}</ul>
}

function StructuredComparison({ artifact }: { artifact: ComparisonArtifact }) {
  const papers = artifact.papers ?? []
  const rows = artifact.rows ?? []
  const comparableRows = rows.filter((row) => isComparableRow(row, papers))
  const incomparableRows = rows.filter((row) => !isComparableRow(row, papers))
  return <>
    <div className="results-comparison-table-wrap">
      <table className="results-comparison-table results-comparison-table--structured">
        <caption className="sr-only">Short structured comparison by paper</caption>
        <thead><tr><th scope="col">Paper</th><th scope="col">Findings</th><th scope="col">Source-backed metrics</th></tr></thead>
        <tbody>{papers.map((paper, index) => <tr key={`${comparisonPaperId(paper, index)}-${index}`}><th scope="row" data-label="Paper">{comparisonPaperTitle(paper, index)}</th><td data-label="Findings">{paper.findings?.length ? <ul className="results-comparison-findings">{paper.findings.map((finding, findingIndex) => <li key={`${findingIndex}-${finding}`}>{finding}</li>)}</ul> : <span className="results-comparison-muted">No concise finding recorded.</span>}</td><td data-label="Source-backed metrics"><ComparisonMetricList paper={paper} /></td></tr>)}</tbody>
      </table>
    </div>
    {comparableRows.length ? <section className="results-comparison-matrix" aria-labelledby="results-comparison-metrics-heading"><div className="results-pmrl-heading"><h3 id="results-comparison-metrics-heading">Comparable metrics</h3><span>Same metric, dataset, unit, and source evidence</span></div><div className="results-comparison-table-wrap"><table className="results-comparison-table results-comparison-table--metrics"><caption className="sr-only">Comparable source-backed metrics</caption><thead><tr><th scope="col">Metric</th><th scope="col">Dataset</th><th scope="col">Unit</th>{papers.map((paper, index) => <th scope="col" key={`${comparisonPaperId(paper, index)}-metric-heading`}>{comparisonPaperTitle(paper, index)}</th>)}</tr></thead><tbody>{comparableRows.map((row, rowIndex) => <tr key={`${row.metric}-${row.dataset}-${rowIndex}`}><th scope="row" data-label="Metric">{row.metric}</th><td data-label="Dataset">{row.dataset}</td><td data-label="Unit">{row.unit}</td>{papers.map((paper, index) => <td data-label={comparisonPaperTitle(paper, index)} key={`${comparisonPaperId(paper, index)}-${rowIndex}`}><strong>{comparisonValue(row, paper, index)}</strong><small>Source: {comparisonQuote(row, paper, index)}</small></td>)}</tr>)}</tbody></table></div></section> : null}
    {incomparableRows.length ? <section className="results-comparison-incomparable" aria-label="Metrics that cannot be compared directly"><h3>Other reported metrics</h3><ul>{incomparableRows.map((row, rowIndex) => <li key={`${row.metric}-${row.dataset}-${rowIndex}`}><strong>{row.metric || 'Metric'}</strong>{row.dataset ? <span> · {row.dataset}</span> : null}{row.unit ? <span> · {row.unit}</span> : null}<span> — Not directly comparable</span>{row.reason ? <small>{row.reason}</small> : null}</li>)}</ul></section> : null}
    {artifact.synthesis?.length ? <section className="results-comparison-synthesis" aria-labelledby="results-comparison-synthesis-heading"><h3 id="results-comparison-synthesis-heading">Synthesis</h3><ul>{artifact.synthesis.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}</ul></section> : null}
  </>
}

function ComparisonPanel({ result, papers, benchmark, facts }: { result: RunResult | null; papers: PaperSource[]; benchmark: string; facts: FactsRecord }) {
  const artifact = readComparisonArtifact(result)
  const structuredArtifact = artifact && artifact.available !== false && (artifact.papers?.length ?? 0) >= 2 ? artifact : null
  const benchmarkSource = benchmark || artifact?.markdown?.trim() || ''
  const compared = comparisonPapers(papers, facts)
  const fallbackIds = new Set(stringList(facts.noteFallbackPaperIds).map((id) => id.toLowerCase().replace(/^arxiv:/, '')))
  const hasFallback = compared.some((paper) => isFallbackPaper(paper, fallbackIds))
  const paperCount = artifact?.papers?.length ?? compared.length
  return (
    <section className="results-comparison" aria-labelledby="results-comparison-heading">
      <div className="results-panel-heading"><div><p className="results-eyebrow">Evidence-based findings</p><h2 id="results-comparison-heading">Comparison</h2></div><span className="results-panel-meta">{paperCount} papers</span></div>
      {structuredArtifact ? <StructuredComparison artifact={structuredArtifact} /> : <div className="results-comparison-legacy"><p>This earlier run has a saved benchmark. Open it below to read the comparison.</p></div>}
      {hasFallback ? <p className="results-comparison-note">Fallback PMRL output is omitted from this comparison because it was not finalized.</p> : null}
      {benchmarkSource ? <details className="results-benchmark-source"><summary>View benchmark Markdown</summary><div className="results-benchmark-source__body"><MarkdownContent content={benchmarkSource} compact /></div></details> : null}
    </section>
  )
}

function markdownFilename(reportId: string): string {
  const name = reportId.split(/[\\/]/).pop()?.trim() || 'research-report'
  const safeName = name.replace(/[^A-Za-z0-9._-]+/g, '_') || 'research-report'
  return safeName.toLowerCase().endsWith('.md') ? safeName : `${safeName}.md`
}

function ReportPanel({ markdown, reportId, reportFilename, hasFallbackNotes, paperTitle, paperCount }: { markdown: string; reportId?: string; reportFilename?: string; hasFallbackNotes: boolean; paperTitle?: string; paperCount: number }) {
  const [downloadState, setDownloadState] = useState<'idle' | 'downloading' | 'downloaded' | 'failed'>('idle')
  const markdownTitle = titleFromMarkdown(markdown, 'Research report')
  const title = paperCount === 1 && paperTitle ? `Research Report: ${paperTitle}` : /https?:\/\//i.test(markdownTitle) ? 'Research report' : markdownTitle
  const filename = markdownFilename(reportFilename ?? reportId ?? 'research-report')

  const handleDownload = async () => {
    if (!reportId) return
    setDownloadState('downloading')
    try {
      const blob = await downloadReport(reportId)
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = filename
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      window.setTimeout(() => URL.revokeObjectURL(url), 0)
      setDownloadState('downloaded')
    } catch {
      setDownloadState('failed')
    }
  }

  return (
    <article className="results-report" aria-labelledby="results-report-heading">
      <header className="results-report__header"><div><p className="results-eyebrow">Finalized Markdown</p><h2 id="results-report-heading">{title}</h2></div>{reportId ? <div className="results-report__download"><button type="button" className="results-button results-button--secondary" aria-label={`Download ${filename}`} onClick={() => void handleDownload()} disabled={downloadState === 'downloading'}><Download size={15} aria-hidden="true" />{downloadState === 'downloading' ? 'Downloading…' : downloadState === 'downloaded' ? 'Downloaded' : downloadState === 'failed' ? 'Download failed' : 'Download'}</button></div> : null}</header>
      {hasFallbackNotes ? <p className="results-report__warning" role="note"><BookOpen size={14} aria-hidden="true" />This report may include fallback PMRL notes. The Markdown is shown exactly as returned.</p> : null}
      <div className="results-report__body"><MarkdownContent content={withoutFirstMarkdownHeading(markdown)} /></div>
    </article>
  )
}

function ActivityDisclosure({ events }: { events: TraceEvent[] }) {
  const entries = activityEvents(events)
  if (!entries.length) return null
  return (
    <details className="results-activity">
      <summary><span>Activity</span><span className="results-activity__count">{entries.length} events</span></summary>
      <ol className="results-activity__list">
        {entries.map((event, index) => { const duration = durationLabel(event.durationMs); const parserWarnings = parserWarningsForEvent(event); const summaryReasons = summaryReasonsForEvent(event); return <li key={`${event.seq}-${event.type}-${event.node}-${index}`}><time dateTime={event.timestamp}>{formatTimestamp(event.timestamp)}</time><span><strong>{stageLabel(event)}</strong><span className="results-activity__copy">{activityCopy(event)}</span>{duration ? <small className="results-activity__duration">Duration · {duration}</small> : null}{parserWarnings.map((warning) => <small className="results-activity__warning" role="note" key={warning}>Extraction warning · {warning}</small>)}{summaryReasons.map((reason) => <small className="results-activity__warning" role="note" key={reason}>Summary validation · {reason}</small>)}</span></li> })}
      </ol>
    </details>
  )
}

function handleTabKey(event: KeyboardEvent<HTMLButtonElement>, tabs: ResultTab[], tab: ResultTab, onChange: (tab: ResultTab) => void) {
  const current = tabs.indexOf(tab)
  if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft' && event.key !== 'Home' && event.key !== 'End') return
  event.preventDefault()
  const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length
  const nextTab = tabs[next]
  onChange(nextTab)
  document.getElementById(`results-tab-${nextTab}`)?.focus()
}

export function ResultsWorkspace({ snapshot, events, onNewResearch }: ResultsWorkspaceProps) {
  const result = resultFrom(snapshot)
  const papers = useMemo(() => readResultPapers(result), [result])
  const facts = useMemo(() => mergedFacts(events), [events])
  const benchmark = readBenchmark(result).trim()
  const reportMarkdown = readReportMarkdown(result).trim()
  const reportId = reportIdFrom(result, facts)
  const reportFilename = readReportFilename(result, reportId)
  const comparisonAvailable = isComparisonAvailable(result, papers, facts)
  const availableTabs = useMemo<ResultTab[]>(() => ['summary', ...(comparisonAvailable ? ['comparison' as const] : []), ...(reportMarkdown ? ['report' as const] : [])], [comparisonAvailable, reportMarkdown])
  const [activeTab, setActiveTab] = useState<ResultTab>('summary')
  const [selectedIndex, setSelectedIndex] = useState(0)
  const safeSelectedIndex = papers.length ? Math.min(selectedIndex, papers.length - 1) : 0

  useEffect(() => {
    if (!availableTabs.includes(activeTab)) setActiveTab('summary')
  }, [activeTab, availableTabs])

  useEffect(() => {
    if (selectedIndex >= papers.length && papers.length) setSelectedIndex(papers.length - 1)
  }, [papers.length, selectedIndex])

  const selectedPaper = papers[safeSelectedIndex]
  const fallbackPaperIds = new Set(stringList(facts.noteFallbackPaperIds).map((id) => id.toLowerCase().replace(/^arxiv:/, '')))
  const headingFallbackPaperIds = new Set(stringList(facts.headingFallbackPaperIds).map((id) => id.toLowerCase().replace(/^arxiv:/, '')))
  const isFallback = selectedPaper ? isFallbackPaper(selectedPaper, fallbackPaperIds) || ['fallback', 'invalid'].includes(readPaperNotesQuality(selectedPaper)) : false
  const reportHasFallbackNotes = papers.some((paper) => isFallbackPaper(paper, fallbackPaperIds) || ['fallback', 'invalid'].includes(readPaperNotesQuality(paper)))

  return (
    <div className="results-workspace">
      <SourceRail papers={papers} selectedIndex={safeSelectedIndex} onSelect={setSelectedIndex} />
      <main className="results-main">
        <div className="results-main__inner">
              <header className="results-main__header"><div><p className="results-eyebrow">Research Scout</p><h1 id="results-heading" data-results-heading tabIndex={-1}>Research findings</h1></div><button type="button" className="results-button results-button--quiet" onClick={onNewResearch}><Plus size={15} aria-hidden="true" />New research</button></header>
          <MobileSourceSelect papers={papers} selectedIndex={safeSelectedIndex} onSelect={setSelectedIndex} />
          <div className="results-tabs" role="tablist" aria-label="Research findings">
            {availableTabs.map((tab) => <button type="button" role="tab" id={`results-tab-${tab}`} aria-controls={`results-panel-${tab}`} aria-selected={activeTab === tab} tabIndex={activeTab === tab ? 0 : -1} className={activeTab === tab ? 'is-active' : ''} key={tab} onClick={() => setActiveTab(tab)} onKeyDown={(event) => handleTabKey(event, availableTabs, tab, setActiveTab)}>{tab === 'summary' ? 'Summary' : tab === 'comparison' ? 'Comparison' : 'Report'}</button>)}
          </div>
          <div className="results-panel" role="tabpanel" id={`results-panel-${activeTab}`} aria-labelledby={`results-tab-${activeTab}`}>
            {activeTab === 'summary' ? <SummaryPanel paper={selectedPaper} index={safeSelectedIndex} total={papers.length} isFallback={isFallback} sourceWarnings={selectedPaper ? sourceQualityWarnings(selectedPaper, headingFallbackPaperIds) : []} /> : null}
            {activeTab === 'comparison' && comparisonAvailable ? <ComparisonPanel result={result} papers={papers} benchmark={benchmark} facts={facts} /> : null}
            {activeTab === 'report' && reportMarkdown ? <ReportPanel markdown={reportMarkdown} reportId={reportId} reportFilename={reportFilename} hasFallbackNotes={reportHasFallbackNotes} paperTitle={papers.length === 1 ? paperDisplayTitle(papers[0]) : undefined} paperCount={papers.length} /> : null}
          </div>
          <ActivityDisclosure events={events} />
        </div>
      </main>
    </div>
  )
}
