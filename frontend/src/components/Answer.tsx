import { useEffect, useState, type ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { durationText, fmtNum, listJoin, shortTitle } from '../lib/format'
import { followupSources, runSeconds } from '../lib/run'
import type { AnswerSummary, InsightItem, MetricItem, PaperCard, ResultRow, RunView, SummaryBlock } from '../types'
import { Copy, Download, MapIcon, Ok } from './icons'

export interface AnswerActions {
  onMap: () => void
  onDownload: (reportId: string) => void
  onCopyBibtex: (bibtex: string) => void
}

interface Ctx {
  summary: AnswerSummary
  paperName: string
  graphNodes?: number
  /** Repository URL per paperId / arxivId, from result.papers[].repository. */
  repos: Record<string, string>
}

function RepoLink({ url }: { url: string }) {
  return (
    <>
      Mã nguồn:{' '}
      <a href={url} target="_blank" rel="noreferrer">
        {url.replace(/^https?:\/\/(www\.)?/, '').replace(/\/$/, '')}
      </a>
    </>
  )
}

// ---- small tolerant helpers ------------------------------------------------------

const asText = (v: unknown): string => (typeof v === 'string' ? v : '')
const asArray = <T,>(v: unknown): T[] => (Array.isArray(v) ? (v as T[]) : [])
const arxivUrl = (id?: string) => (id ? `https://arxiv.org/abs/${id}` : undefined)

const HEADINGS: Record<string, string> = {
  insights: 'Insight',
  problem: 'Vấn đề',
  intuition: 'Ý tưởng cốt lõi',
  keyResults: 'Kết quả chính',
  metrics: 'Đại lượng đo',
  setup: 'Thiết lập',
  results: 'Kết quả chính',
  caveats: 'Cần lưu ý',
  foundations: 'Nền tảng',
  takeaway: 'Nên dùng khi nào',
}

// ---- blocks -------------------------------------------------------------------------

function Insights({ items }: { items: InsightItem[] }) {
  if (!items.length) return null
  return (
    <ul>
      {items.map((i, k) => (
        <li key={k}>
          {i.finding && <strong>{i.finding}</strong>} {i.why}
        </li>
      ))}
    </ul>
  )
}

function Metrics({ items }: { items: MetricItem[] }) {
  if (!items.length) return null
  return (
    <dl className="defs">
      {items.map((m, k) => (
        <div key={k} className="def">
          <dt>{m.name}</dt>
          <dd>
            {m.meaning}
            {typeof m.higherIsBetter === 'boolean' && (m.higherIsBetter ? ' Cao hơn là tốt hơn.' : ' Thấp hơn là tốt hơn.')}
          </dd>
        </div>
      ))}
    </dl>
  )
}

const isGain = (r: ResultRow) => {
  const d = (r.delta ?? '').trim()
  if (!d) return false
  const negative = /^[-−–]/.test(d)
  return r.higherIsBetter === false ? negative : !negative
}

function Results({ rows, grounded, source, methodLabel }: { rows: ResultRow[]; grounded?: boolean; source?: string[]; methodLabel: string }) {
  if (!rows.length) return null
  const has = (key: keyof ResultRow) => rows.some((r) => r[key])
  const cols: Array<[keyof ResultRow, string]> = [
    ['setting', 'Thiết lập'],
    ['metric', 'Metric'],
    ['method', methodLabel],
    ['baseline', 'Đối chứng'],
    ['delta', 'Chênh lệch'],
  ]
  const shown = cols.filter(([key]) => has(key))
  const src = asArray<string>(source)
  return (
    <>
      <table className="res">
        <thead>
          <tr>
            {shown.map(([key, label]) => (
              <th key={key}>{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, k) => (
            <tr key={k}>
              {shown.map(([key]) => (
                <td key={key} className={key === 'method' ? 'm' : key === 'delta' && isGain(r) ? 'up' : undefined}>
                  {r[key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {(grounded || src.length > 0) && (
        <div className="prov">
          {grounded && (
            <span className="ok">
              <Ok /> Mọi số khớp nguồn
            </span>
          )}
          {src.length > 0 && <span>Nguồn: {src.join(' và ')} trong paper</span>}
        </div>
      )}
    </>
  )
}

function Cards({ items, ctx }: { items: PaperCard[]; ctx: Ctx }) {
  const [openIds, setOpen] = useState<Set<number>>(new Set())
  if (!items.length) return null
  const toggle = (k: number) =>
    setOpen((prev) => {
      const next = new Set(prev)
      if (!next.delete(k)) next.add(k)
      return next
    })
  return (
    <div className="cards">
      {items.map((c, k) => {
        const open = openIds.has(k)
        const year = c.year ?? c.published
        const arxivId = c.arxivId || ctx.summary.papers?.find((p) => p.paperId && p.paperId === c.paperId)?.arxivId
        const url = c.url ?? arxivUrl(arxivId)
        const repo = ctx.repos[c.paperId ?? arxivId ?? '']
        const inner = asArray<SummaryBlock>(c.blocks)
        return (
          <div key={k} className={`card${open ? ' open' : ''}`}>
            <button type="button" aria-expanded={open} onClick={() => toggle(k)}>
              <span className="ct">{c.title}</span>
              {year != null && <span className="cy">{year}</span>}
              {c.intuition && <span className="cs">{c.intuition}</span>}
            </button>
            {open && (
              <div className="more">
                {inner.length > 0 ? (
                  <Blocks blocks={inner} ctx={ctx} />
                ) : (
                  <>
                    {c.keyResult && (
                      <>
                        <h3>Kết quả chính</h3>
                        <p>{c.keyResult}</p>
                      </>
                    )}
                    {(url || repo) && (
                      <p className="paper-ref">
                        {url && (
                          <a href={url} target="_blank" rel="noreferrer">
                            {arxivId ? `arXiv ${arxivId}` : 'Mở bài'}
                          </a>
                        )}
                        {url && repo ? ' · ' : ''}
                        {repo && <RepoLink url={repo} />}
                      </p>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function Blocks({ blocks, ctx }: { blocks: SummaryBlock[]; ctx: Ctx }) {
  const out: ReactNode[] = []
  let lastHeading = ''
  blocks.forEach((raw, index) => {
    const b = raw as Record<string, any>
    let heading = HEADINGS[b.type] ?? ''
    let body: ReactNode = null
    switch (b.type) {
      case 'insights':
        body = <Insights items={asArray<InsightItem>(b.items)} />
        if (ctx.summary.kind === 'multi') heading = 'Insight chung'
        break
      case 'problem':
      case 'intuition':
      case 'takeaway':
        body = asText(b.text) ? <p>{b.text}</p> : null
        break
      case 'keyResults': {
        const items = asArray<string | InsightItem>(b.items)
        body = (
          <>
            {asText(b.text) && <p>{b.text}</p>}
            {items.length > 0 && (
              <ul>
                {items.map((i, k) =>
                  typeof i === 'string' ? (
                    <li key={k}>{i}</li>
                  ) : (
                    <li key={k}>
                      {i.finding && <strong>{i.finding}</strong>} {i.why}
                    </li>
                  ),
                )}
              </ul>
            )}
          </>
        )
        break
      }
      case 'metrics':
        body = <Metrics items={asArray<MetricItem>(b.items)} />
        break
      case 'setup': {
        const missing = asArray<string>(b.missing)
        const items = asArray<string>(b.items).filter(Boolean)
        if (asText(b.text)) items.unshift(b.text)
        const chips = missing.map((m) => (
          <span key={m}>
            {' '}
            <span className="absent">{m}: paper không nêu</span>
          </span>
        ))
        body =
          items.length > 1 ? (
            <>
              <ul>
                {items.map((i, k) => (
                  <li key={k}>{i}</li>
                ))}
              </ul>
              {chips.length > 0 && <p>{chips}</p>}
            </>
          ) : items.length === 1 || chips.length ? (
            <p>
              {items[0]}
              {chips}
            </p>
          ) : null
        break
      }
      case 'results':
        body = <Results rows={asArray<ResultRow>(b.rows)} grounded={b.grounded} source={b.source} methodLabel={asText(b.methodName) || ctx.paperName || 'Phương pháp'} />
        break
      case 'caveats': {
        const items = asArray<string>(b.items)
        body = (
          <>
            {asText(b.text) && <p>{b.text}</p>}
            {items.length > 0 && (
              <ul>
                {items.map((i, k) => (
                  <li key={k}>{i}</li>
                ))}
              </ul>
            )}
          </>
        )
        break
      }
      case 'foundations': {
        const items = asArray<string>(b.items).filter(Boolean)
        body = items.length ? (
          <p>
            Bài dựa trên {listJoin(items)}.
            {ctx.graphNodes ? ` Bản đồ bên phải cho thấy ${ctx.graphNodes} bài gần nhất.` : ''}
          </p>
        ) : null
        break
      }
      case 'paperCards': {
        const items = asArray<PaperCard>(b.items)
        heading = items.length ? `${items.length} bài` : ''
        body = <Cards items={items} ctx={ctx} />
        break
      }
      default:
        return // unknown block type: ignore
    }
    if (!body) return
    const showHeading = heading && heading !== lastHeading
    lastHeading = heading
    out.push(
      <section key={index} data-block={b.type}>
        {showHeading && <h3>{heading}</h3>}
        {body}
      </section>,
    )
  })
  return <>{out}</>
}

// ---- answer ---------------------------------------------------------------------------

function useEnter() {
  const [on, setOn] = useState(false)
  useEffect(() => {
    const id = requestAnimationFrame(() => requestAnimationFrame(() => setOn(true)))
    return () => cancelAnimationFrame(id)
  }, [])
  return on
}

export function Markdown({ text }: { text: string }) {
  return (
    <div className="md">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  )
}

/** A follow-up answered from the existing report: streamed text plus the "Trả lời từ báo cáo đã có" line. */
export function QuickAnswer({ run }: { run: RunView }) {
  const shown = useEnter()
  const done = run.status !== 'running'
  const secs = runSeconds(run)
  const sources = followupSources(run)
  return (
    <>
      <div className="quick">
        {done ? <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="m3.5 8.5 3 3 6-7" /></svg> : <span className="pulse" />}
        {done ? `Trả lời từ báo cáo đã có${secs != null ? `, ${secs} giây` : ''}` : 'Đang trả lời từ báo cáo đã có…'}
      </div>
      {run.answerText && (
        <div className={`answer${shown ? ' in' : ''}`}>
          <Markdown text={run.answerText} />
          {done && sources.length > 0 && <div className="prov"><span>Nguồn: {sources.join(', ')}</span></div>}
        </div>
      )}
    </>
  )
}

export function AnswerView({ run, graphNodes, actions }: { run: RunView; graphNodes: number; actions: AnswerActions }) {
  const shown = useEnter()
  const summary = run.summary
  const result = run.result
  const reportId = summary?.actions?.reportId ?? result?.reportId ?? null
  const bibtex = asText(result?.bibtex)
  const bibCount = summary?.actions?.bibtexCount ?? (bibtex.match(/^@/gm)?.length ?? 0)
  const paper = summary?.paper
  const paperName = shortTitle(paper?.title ?? '') || ''
  const repos: Record<string, string> = {}
  for (const p of result?.papers ?? []) {
    const url = p.repository?.url
    if (!url) continue
    if (p.paperId) repos[p.paperId] = url
    if (p.arxivId) repos[p.arxivId] = url
  }
  const ctx: Ctx = { summary: summary ?? {}, paperName, graphNodes: summary?.actions?.graphNodes ?? graphNodes, repos }
  const singleRepo = summary?.kind !== 'multi' ? repos[paper?.paperId ?? ''] ?? repos[paper?.arxivId ?? ''] ?? Object.values(repos)[0] : undefined
  // While the run goes on, only the quick summary exists: no total time yet.
  const running = run.status === 'running'
  const secs = running ? undefined : runSeconds(run)
  const blocks = asArray<SummaryBlock>(summary?.blocks)
  const reportReady = run.reportReady && !!reportId
  // The report button is grey and inert until report.md exists; it is hidden when the run ended without one.
  const reportPending = !reportReady && running
  const showActions = reportReady || reportPending || !!bibtex || graphNodes > 0

  return (
    <div className={`answer${shown ? ' in' : ''}`} data-testid="answer">
      {summary ? (
        <>
          {paper?.title && (
            <div className="paper-ref">
              <a href={paper.url ?? arxivUrl(paper.arxivId)} target="_blank" rel="noreferrer">
                {paper.title}
              </a>
              {paper.arxivId ? `, arXiv ${paper.arxivId}` : ''}
              {paper.published || paper.year ? `, ${paper.published ?? paper.year}` : ''}
              {singleRepo && (
                <>
                  {' · '}
                  <RepoLink url={singleRepo} />
                </>
              )}
            </div>
          )}
          {summary.lead && <p className="lead">{summary.lead}</p>}
          <Blocks blocks={blocks} ctx={ctx} />
        </>
      ) : run.answerText ? (
        <Markdown text={run.answerText} />
      ) : null}
      {showActions && (
        <div className="actions">
          {reportReady && reportId && (
            <button type="button" className="btn primary" onClick={() => actions.onDownload(reportId)}>
              <Download /> Tải report.md
            </button>
          )}
          {reportPending && (
            <button type="button" className="btn pending" aria-disabled="true" title="Báo cáo đầy đủ đang được viết…" onClick={(e) => e.preventDefault()}>
              <Download /> Tải report.md
            </button>
          )}
          {bibtex && (
            <button type="button" className="btn" onClick={() => actions.onCopyBibtex(bibtex)}>
              <Copy /> Sao chép BibTeX{bibCount > 1 ? ` (${fmtNum(bibCount)})` : ''}
            </button>
          )}
          {graphNodes > 0 && (
            <button type="button" className="btn" onClick={actions.onMap}>
              <MapIcon /> Bản đồ liên quan
            </button>
          )}
          {secs != null && <span className="meta-line tnum">{durationText(secs)}</span>}
        </div>
      )}
    </div>
  )
}
