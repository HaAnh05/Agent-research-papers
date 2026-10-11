import type { LlmCall, StepView } from '../types'

const nf = new Intl.NumberFormat('vi-VN')
export const fmtNum = (n: number) => nf.format(n)

/** 4000 ms -> "0:04" */
export function clock(ms?: number): string {
  if (ms == null || !Number.isFinite(ms)) return '—'
  const total = Math.max(0, Math.round(ms / 1000))
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}

/** 213 -> "3 phút 33 giây", 8 -> "8 giây" */
export function durationText(sec?: number): string {
  if (sec == null || !Number.isFinite(sec)) return ''
  const total = Math.max(0, Math.round(sec))
  const m = Math.floor(total / 60)
  const s = total % 60
  if (!m) return `${s} giây`
  return `${m} phút ${String(s).padStart(2, '0')} giây`
}

/** "ReFlect: An Effective ..." -> "ReFlect"; otherwise the first three words. */
export function shortTitle(title: string): string {
  const m = title.match(/^([^:]{2,26}):/)
  return m ? m[1] : title.split(/\s+/).slice(0, 3).join(' ')
}

export function listJoin(items: string[]): string {
  if (items.length <= 1) return items.join('')
  return `${items.slice(0, -1).join(', ')} và ${items[items.length - 1]}`
}

const SOURCE_LABEL: Record<string, string> = { ar5iv: 'ar5iv', html: 'HTML', pdf: 'PDF', arxiv_html: 'HTML', arxiv: 'arXiv' }

/** One line of measured facts per step, in Vietnamese. Returns '' when the step has none. */
export function factsLine(step: StepView): string {
  const f = step.facts
  const parts: string[] = []
  const n = (v: unknown): v is number => typeof v === 'number'
  switch (step.node) {
    case 'router':
      if (f.intent === 'direct_read') parts.push('Đọc một bài báo cụ thể')
      else if (f.intent === 'search') parts.push('Tìm theo chủ đề')
      else if (f.intent === 'followup') parts.push('Câu hỏi tiếp theo')
      break
    case 'search_papers':
      if (n(f.resultCount)) parts.push(`${f.resultCount} kết quả`)
      break
    case 'eval_search':
      if (n(f.selectedCount)) parts.push(`giữ ${f.selectedCount} bài`)
      break
    case 'refine_query':
      if (n(f.retryCount)) parts.push(`lần thử lại thứ ${f.retryCount}`)
      break
    case 'read_paper':
      if (n(f.paperCount) && f.paperCount > 1) parts.push(`${f.paperCount} bài`)
      if (n(f.tableCount)) parts.push(`${f.tableCount} bảng`)
      if (n(f.referenceCount)) {
        parts.push(
          n(f.refsResolved) && n(f.refsTotal)
            ? `${f.refsResolved}/${f.refsTotal} tài liệu tham khảo khớp với arXiv`
            : `${f.referenceCount} tài liệu tham khảo`,
        )
      }
      if (typeof f.source === 'string') parts.push(`nguồn ${SOURCE_LABEL[f.source] ?? f.source}`)
      if (n(f.failedPdfCount) && f.failedPdfCount > 0) parts.push(`${f.failedPdfCount} PDF lỗi`)
      break
    case 'write_notes': {
      if (n(f.insightCount)) parts.push(`${f.insightCount} insight`)
      const e = f.evalSettings
      if (e && n(e.kept) && n(e.proposed)) parts.push(`${e.kept}/${e.proposed} hàng kết quả khớp nguồn`)
      if (e && n(e.dropped)) parts.push(`${e.dropped} hàng bị loại`)
      if (Array.isArray(f.noteFallbackPaperIds) && f.noteFallbackPaperIds.length) {
        parts.push(`${f.noteFallbackPaperIds.length} bài dùng ghi chú dự phòng`)
      }
      break
    }
    case 'citation_graph':
      if (n(f.seeds)) parts.push(`${f.seeds} bài gốc`)
      if (n(f.pool)) parts.push(`${f.pool} ứng viên`)
      if (n(f.nodeCount)) parts.push(`giữ ${f.nodeCount} bài`)
      if (f.graphStatus && f.graphStatus !== 'ok') parts.push(`trạng thái ${f.graphStatus}`)
      break
    case 'web_enrich':
      if (n(f.repositoryCount)) {
        parts.push(f.repositoryCount ? `${f.repositoryCount} repository${f.repoSource ? ` (${f.repoSource})` : ''}` : 'chưa có repository')
      }
      if (n(f.bibtexCount)) parts.push(`${f.bibtexCount} BibTeX`)
      break
    case 'summarize':
      if (n(f.keyResultCount)) parts.push(`${f.keyResultCount} kết quả chính`)
      if (n(f.resultRowCount)) parts.push(`${f.resultRowCount} hàng kết quả khớp nguồn`)
      break
    case 'final_report':
      if (n(f.sectionCount)) parts.push(`report.md ${f.sectionCount} mục`)
      break
    default:
      break
  }
  return parts.join(' · ')
}

/** 3100 -> "3.1k", 12400 -> "12k", 640 -> "640" */
export function compactCount(n: number): string {
  if (n < 1000) return String(n)
  if (n < 10000) return `${(Math.round(n / 100) / 10).toString()}k`
  return `${Math.round(n / 1000)}k`
}

/**
 * One line about the LLM calls of a step, with commas: "LLM 1:12, 4.2k in, 3.1k out, 9.4k reasoning".
 * Several calls run in parallel, so the time is the longest call. A token total is shown only when
 * every call reported it; a missing value is left out, never guessed. Returns '' when there is no call.
 */
export function llmLine(step: StepView): string {
  const raw = step.facts.llmCalls
  const calls = (Array.isArray(raw) ? raw : []).filter((c): c is LlmCall => !!c && typeof c === 'object')
  if (!calls.length) return ''
  const parts: string[] = []
  const times = calls.map((c) => c.ms).filter((v): v is number => typeof v === 'number')
  const longest = times.length ? clock(Math.max(...times)) : ''
  if (calls.length > 1) parts.push(`${calls.length} lệnh LLM`, ...(longest ? [`dài nhất ${longest}`] : []))
  else parts.push(longest ? `LLM ${longest}` : 'LLM')
  const total = (key: 'inputTokens' | 'outputTokens' | 'reasoningTokens'): number | undefined => {
    const values = calls.map((c) => c[key])
    return values.every((v) => typeof v === 'number') ? (values as number[]).reduce((a, b) => a + b, 0) : undefined
  }
  const input = total('inputTokens')
  const output = total('outputTokens')
  const reasoning = total('reasoningTokens')
  if (input != null) parts.push(`${compactCount(input)} in`)
  if (output != null) parts.push(`${compactCount(output)} out`)
  if (reasoning != null) parts.push(`${compactCount(reasoning)} reasoning`)
  return parts.join(', ')
}
