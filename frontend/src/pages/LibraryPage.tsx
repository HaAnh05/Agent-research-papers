import { useCallback, useEffect, useState } from 'react'
import { BookOpen, CalendarDays, Download, FileText, LoaderCircle, RefreshCw, Search, Sparkles } from 'lucide-react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { BackButton, PageIntro, SectionHeading } from '../components/AppShell'
import { MarkdownContent, titleFromMarkdown } from '../components/MarkdownContent'
import { ReportView } from '../components/ReportView'
import { getReport, getReports, downloadReport } from '../lib/api'
import type { ReportDocument, ReportSummary } from '../types'
import { readBenchmark, readBibtex, readResultPapers, readReportMarkdown } from '../types'

function reportDate(report: ReportSummary): string {
  const value = report.updatedAt ?? report.updated_at ?? report.createdAt ?? report.created_at
  if (!value) return 'Không rõ thời gian'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat('vi-VN', { dateStyle: 'medium', timeStyle: 'short' }).format(date)
}

function formatSize(size?: number): string {
  if (!size) return ''
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${Math.round(size / 1024)} KB`
  return `${(size / (1024 * 1024)).toFixed(1)} MB`
}

function markdownFilename(id: string): string {
  return id.toLowerCase().endsWith('.md') ? id : `${id}.md`
}

function ReportCard({ report, onDownload }: { report: ReportSummary; onDownload: (report: ReportSummary) => void }) {
  return <article className="library-card"><div className="library-card__mark"><FileText size={18} /></div><div className="library-card__body"><div className="library-card__topline"><span className="eyebrow">Research report</span><span className="library-card__date"><CalendarDays size={13} />{reportDate(report)}</span></div><h3><Link to={`/library/${encodeURIComponent(report.id)}`}>{report.title}</Link></h3><div className="library-card__meta">{typeof report.paperCount === 'number' || typeof report.paper_count === 'number' ? <span>{report.paperCount ?? report.paper_count} paper</span> : null}{report.size ? <span>· {formatSize(report.size)}</span> : null}<code>{report.id}</code></div></div><div className="library-card__actions"><Link className="icon-button" to={`/library/${encodeURIComponent(report.id)}`} aria-label={`Mở ${report.title}`}><BookOpen size={16} /></Link><button type="button" className="icon-button" onClick={() => onDownload(report)} aria-label={`Tải ${report.title}`}><Download size={16} /></button></div></article>
}

export function LibraryPage() {
  const [reports, setReports] = useState<ReportSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [downloadId, setDownloadId] = useState<string | null>(null)

  const loadReports = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setReports(await getReports())
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Không thể tải thư viện báo cáo.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void loadReports() }, [loadReports])

  const filtered = reports.filter((report) => `${report.title} ${report.id}`.toLowerCase().includes(search.toLowerCase().trim()))
  const handleDownload = async (report: ReportSummary) => {
    setDownloadId(report.id)
    try {
      const blob = await downloadReport(report.id)
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = markdownFilename(report.id)
      anchor.click()
      window.setTimeout(() => URL.revokeObjectURL(url), 0)
    } catch (downloadError) {
      setError(downloadError instanceof Error ? downloadError.message : 'Không thể tải báo cáo.')
    } finally {
      setDownloadId(null)
    }
  }

  return <div className="page page--library"><PageIntro eyebrow="Archive" title="Thư viện báo cáo" description="Các Markdown report đã được tạo và lưu bền vững dưới reports/." action={<button type="button" className="secondary-button" onClick={() => void loadReports()}><RefreshCw size={15} /> Làm mới</button>} /><div className="library-toolbar"><div className="library-count"><span className="library-count__number">{reports.length}</span><span>reports đã lưu</span></div><label className="library-search"><Search size={16} /><span className="sr-only">Tìm báo cáo</span><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Tìm theo title hoặc ID…" /></label></div>{error ? <div className="inline-error" role="alert"><Sparkles size={16} />{error}</div> : null}{loading ? <div className="loading-state"><LoaderCircle className="spin" size={20} /> Đang đọc thư viện…</div> : filtered.length ? <div className="library-list">{filtered.map((report) => <ReportCard key={report.id} report={report} onDownload={(item) => void handleDownload(item)} />)}</div> : <div className="empty-state empty-state--library"><BookOpen size={24} /><h2>{reports.length ? 'Không tìm thấy report' : 'Thư viện còn trống'}</h2><p>{reports.length ? 'Thử một title hoặc ArXiv ID khác.' : 'Hoàn tất research scout đầu tiên, report sẽ xuất hiện ở đây.'}</p><Link className="primary-button" to="/">Bắt đầu nghiên cứu</Link></div>}{downloadId ? <span className="sr-only" role="status">Đang tải báo cáo…</span> : null}</div>
}

export function LibraryReportPage() {
  const { reportId } = useParams<{ reportId: string }>()
  const navigate = useNavigate()
  const [report, setReport] = useState<ReportDocument | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [downloading, setDownloading] = useState(false)

  useEffect(() => {
    if (!reportId) return
    setLoading(true)
    setError(null)
    void getReport(reportId).then(setReport).catch((loadError) => setError(loadError instanceof Error ? loadError.message : 'Không thể mở báo cáo.')).finally(() => setLoading(false))
  }, [reportId])

  const handleDownload = async () => {
    if (!reportId) return
    setDownloading(true)
    try {
      const blob = await downloadReport(reportId)
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = markdownFilename(reportId)
      anchor.click()
      window.setTimeout(() => URL.revokeObjectURL(url), 0)
    } catch (downloadError) {
      setError(downloadError instanceof Error ? downloadError.message : 'Không thể tải báo cáo.')
    } finally {
      setDownloading(false)
    }
  }

  if (loading) return <div className="page page--library-detail"><div className="loading-state"><LoaderCircle className="spin" size={20} /> Đang mở báo cáo…</div></div>
  if (error || !report) return <div className="page page--library-detail"><BackButton /><div className="error-state"><FileText size={22} /><h1>Không mở được báo cáo</h1><p>{error ?? 'Báo cáo không tồn tại.'}</p><button type="button" className="primary-button" onClick={() => navigate('/library')}>Về thư viện</button></div></div>

  const markdown = readReportMarkdown(report)
  return <div className="page page--library-detail"><BackButton>Về thư viện</BackButton><div className="detail-heading"><div><p className="eyebrow">Saved report · {report.id}</p><h1>{report.title || titleFromMarkdown(markdown)}</h1><p>{reportDate(report)}</p></div><button type="button" className="secondary-button" onClick={() => void handleDownload()} disabled={downloading}><Download size={15} />{downloading ? 'Đang tải…' : 'Tải Markdown'}</button></div><ReportView title={report.title} markdown={markdown} papers={report.papers ?? []} benchmark={report.benchmark ?? ''} bibtex={report.bibtex ?? ''} filename={markdownFilename(report.id)} /> </div>
}
