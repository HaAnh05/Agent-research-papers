import { useState } from 'react'
import { Download, ExternalLink, FileCode2, Github, Layers3, Quote, Sparkles } from 'lucide-react'
import { MarkdownContent, withoutFirstMarkdownHeading } from './MarkdownContent'
import { SectionHeading } from './AppShell'
import { readPaperId, readPaperNotes, readPaperPdfUrl, readPaperRepos, type PaperSource } from '../types'

type ReportTab = 'report' | 'papers' | 'benchmark' | 'bibtex'

interface ReportViewProps {
  title?: string
  markdown: string
  papers?: PaperSource[]
  benchmark?: string
  bibtex?: string
  filename?: string
  onDownload?: () => void
  downloading?: boolean
  compact?: boolean
}

const tabs: Array<{ id: ReportTab; label: string; icon: typeof Sparkles }> = [
  { id: 'report', label: 'Báo cáo', icon: Sparkles },
  { id: 'papers', label: 'Papers & PMRL', icon: Layers3 },
  { id: 'benchmark', label: 'Benchmark', icon: Quote },
  { id: 'bibtex', label: 'BibTeX', icon: FileCode2 },
]

function safeHttpsHref(value: unknown): string | undefined {
  if (typeof value !== 'string' || !value.trim()) return undefined
  try {
    const url = new URL(value)
    return url.protocol === 'https:' ? url.toString() : undefined
  } catch {
    return undefined
  }
}

function safeGithubHref(value: unknown): string | undefined {
  const href = safeHttpsHref(value)
  if (!href) return undefined
  const hostname = new URL(href).hostname.toLowerCase()
  return hostname === 'github.com' || hostname === 'www.github.com' ? href : undefined
}

function arxivHref(paper: PaperSource): string | undefined {
  const rawId = paper.arxivId ?? paper.arxiv_id ?? readPaperId(paper)
  const id = typeof rawId === 'string' ? rawId.replace(/^arxiv:/i, '').trim() : ''
  if (!/^(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z][\w.-]*(?:\.[A-Z]{2})?\/\d{7}(?:v\d+)?)$/i.test(id)) return undefined
  return `https://arxiv.org/abs/${id}`
}

function paperDisplayTitle(paper: PaperSource): string {
  const title = paper.title?.trim()
  if (title && !/^untitled paper$/i.test(title)) return title
  return readPaperId(paper) ?? title ?? 'Untitled paper'
}

function PaperCard({ paper, index }: { paper: PaperSource; index: number }) {
  const id = readPaperId(paper)
  const arxivUrl = arxivHref(paper)
  const rawPdfUrl = readPaperPdfUrl(paper)
  const pdfUrl = safeHttpsHref(rawPdfUrl)
  const pdfUnavailable = Boolean(rawPdfUrl && !pdfUrl)
  const repos = readPaperRepos(paper).map((repo) => ({ ...repo, href: safeGithubHref(repo.url) }))
  const validRepos = repos.filter((repo): repo is typeof repo & { href: string } => Boolean(repo.href))
  const invalidRepoCount = repos.length - validRepos.length
  const notes = readPaperNotes(paper)
  return (
    <article className="paper-detail-card">
      <div className="paper-detail-card__number">{String(index + 1).padStart(2, '0')}</div>
      <div className="paper-detail-card__body">
        <div className="paper-detail-card__heading">
          <h3>{paperDisplayTitle(paper)}</h3>
          {arxivUrl ? <a className="inline-link" href={arxivUrl} target="_blank" rel="noreferrer">arXiv <ExternalLink size={13} /></a> : null}
          {pdfUrl ? <a className="inline-link" href={pdfUrl} target="_blank" rel="noreferrer">Mở PDF <ExternalLink size={13} /></a> : null}
          {pdfUnavailable ? <span className="report-link-warning" role="note">PDF link unavailable; destination was not verified.</span> : null}
        </div>
        <div className="paper-meta-line">{id ? <code>{id}</code> : <span>Local PDF</span>}{paper.published ? <><span>·</span><span>{paper.published}</span></> : null}</div>
        {paper.authors?.length ? <p className="paper-authors">{paper.authors.join(', ')}</p> : null}
        <div className="paper-enrichment-grid">
          {notes ? (
            <div className="pmrl-grid">
              {(['problem', 'method', 'result', 'limitation'] as const).map((key) => <div className={`pmrl-card pmrl-card--${key}`} key={key}><span>{key === 'problem' ? 'Problem' : key === 'method' ? 'Method' : key === 'result' ? 'Result' : 'Limitation'}</span><p>{notes[key]}</p></div>)}
            </div>
          ) : null}
          {validRepos.length ? <div className="repo-list"><div className="subsection-label"><Github size={14} /> GitHub liên quan</div>{validRepos.map((repo) => <a key={repo.url} className="repo-row" href={repo.href} target="_blank" rel="noreferrer"><span><strong>{repo.name}</strong>{repo.description ? <small>{repo.description}</small> : null}</span><span className="repo-row__meta">{repo.stars ? `★ ${repo.stars}` : ''}</span></a>)}</div> : null}
          {invalidRepoCount ? <span className="report-link-warning" role="note">{invalidRepoCount === 1 ? 'Repository link unavailable; destination was not verified.' : 'Some repository links were unavailable; destinations were not verified.'}</span> : null}
        </div>
        {paper.bibtex ? <details className="bibtex-inline"><summary><FileCode2 size={14} /> Trích dẫn BibTeX</summary><pre>{paper.bibtex}</pre></details> : null}
      </div>
    </article>
  )
}

export function ReportView({ title, markdown, papers = [], benchmark = '', bibtex = '', filename, onDownload, downloading = false, compact = false }: ReportViewProps) {
  const [activeTab, setActiveTab] = useState<ReportTab>('report')
  return (
    <section className={`report-view ${compact ? 'report-view--compact' : ''}`}>
      <div className="report-toolbar">
        <div><p className="eyebrow">Output</p><h2>{title ? 'Nội dung báo cáo' : 'Research report'}</h2></div>
        {onDownload ? <div className="report-toolbar__download"><button type="button" className="secondary-button" onClick={onDownload} disabled={downloading}><Download size={15} />{downloading ? 'Đang tải…' : 'Tải Markdown'}</button>{filename ? <small>Filename · <code>{filename}</code></small> : null}</div> : null}
      </div>
      <div className="report-tabs" role="tablist" aria-label="Các phần báo cáo">
        {tabs.map(({ id, label, icon: Icon }) => <button type="button" key={id} role="tab" aria-selected={activeTab === id} className={`report-tab ${activeTab === id ? 'is-active' : ''}`} onClick={() => setActiveTab(id)}><Icon size={15} aria-hidden="true" />{label}{id === 'papers' && papers.length ? <span className="count-chip">{papers.length}</span> : null}</button>)}
      </div>
      <div className="report-panel" role="tabpanel">
        {activeTab === 'report' ? <MarkdownContent content={withoutFirstMarkdownHeading(markdown)} /> : null}
        {activeTab === 'papers' ? <div className="papers-panel"><SectionHeading title="Papers đã phân tích" count={papers.length} />{papers.length ? papers.map((paper, index) => <PaperCard key={`${paper.title}-${index}`} paper={paper} index={index} />) : <div className="empty-panel"><Layers3 size={20} /> Chưa có dữ liệu paper.</div>}</div> : null}
        {activeTab === 'benchmark' ? <div className="benchmark-panel"><SectionHeading title="Ma trận so sánh" />{benchmark ? <MarkdownContent content={benchmark} /> : <div className="empty-panel"><Quote size={20} /> Benchmark sẽ xuất hiện khi workflow phân tích từ hai paper trở lên.</div>}</div> : null}
        {activeTab === 'bibtex' ? <div className="bibtex-panel"><SectionHeading title="Trích dẫn BibTeX" /><pre className="bibtex-block">{bibtex || '% Chưa có BibTeX cho phiên này.'}</pre>{bibtex ? <button type="button" className="copy-button" onClick={() => void navigator.clipboard?.writeText(bibtex)}>Sao chép BibTeX</button> : null}</div> : null}
      </div>
    </section>
  )
}
