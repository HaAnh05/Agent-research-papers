import { useRef, useState } from 'react'
import { ArrowUp, FileText, Link2, LoaderCircle, Paperclip, X } from 'lucide-react'
import type { RunInput } from '../types'

interface ResearchComposerProps {
  onSubmit: (input: RunInput) => Promise<void>
  disabled?: boolean
  initialQuery?: string
}

function parsePaperInputs(value: string): string[] {
  return value.split(/[\n,]+/).map((item) => item.trim()).filter(Boolean)
}

export function validateResearchInput(query: string, paperInputs: string[], files: File[]): string | null {
  if (!query.trim() && paperInputs.length === 0 && files.length === 0) return 'Thêm một câu hỏi, ArXiv ID/URL hoặc file PDF để bắt đầu.'
  const invalidFile = files.find((file) => !isPdfFile(file))
  if (invalidFile) return `“${invalidFile.name}” không phải file PDF được hỗ trợ.`
  return null
}

export function isPdfFile(file: File): boolean {
  const filenameIsPdf = file.name.toLowerCase().endsWith('.pdf')
  const mimeIsPdf = !file.type || file.type === 'application/pdf' || file.type === 'application/octet-stream'
  return filenameIsPdf && mimeIsPdf
}

export function ResearchComposer({ onSubmit, disabled = false, initialQuery = '' }: ResearchComposerProps) {
  const [query, setQuery] = useState(initialQuery)
  const [paperText, setPaperText] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [validationError, setValidationError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFiles = (selected: FileList | null) => {
    if (!selected) return
    const next = [...selected]
    const error = validateResearchInput(query, parsePaperInputs(paperText), next)
    if (error && next.some((file) => !isPdfFile(file))) {
      setValidationError(error)
      return
    }
    setValidationError(null)
    setFiles((current) => [...current, ...next.filter((file) => !current.some((item) => item.name === file.name && item.size === file.size))])
  }

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const paperInputs = parsePaperInputs(paperText)
    const error = validateResearchInput(query, paperInputs, files)
    if (error) {
      setValidationError(error)
      return
    }
    setValidationError(null)
    setSubmitting(true)
    try {
      await onSubmit({ query: query.trim(), paperInputs, files })
    } catch (submitError) {
      setValidationError(submitError instanceof Error ? submitError.message : 'Không thể bắt đầu workflow.')
    } finally {
      setSubmitting(false)
    }
  }

  const isDisabled = disabled || submitting
  return (
    <form className="research-composer" onSubmit={handleSubmit} noValidate onKeyDown={(event) => {
      if (event.key === 'Enter' && (event.metaKey || event.ctrlKey) && !isDisabled) {
        event.preventDefault()
        event.currentTarget.requestSubmit()
      }
    }}>
      <div className="composer-card">
        <label className="composer-label" htmlFor="research-query">Bạn muốn tìm hiểu điều gì?</label>
        <textarea
          id="research-query"
          className="composer-query"
          value={query}
          onChange={(event) => { setQuery(event.target.value); if (validationError) setValidationError(null) }}
          placeholder="Ví dụ: So sánh các phương pháp 3D Gaussian Splatting cho SLAM trong môi trường indoor…"
          rows={3}
          disabled={isDisabled}
          autoComplete="off"
        />
        <div className="composer-divider" />
        <div className="composer-field composer-field--source">
          <Link2 size={16} aria-hidden="true" />
          <label htmlFor="paper-inputs">ArXiv ID hoặc URL</label>
          <textarea
            id="paper-inputs"
            value={paperText}
            onChange={(event) => { setPaperText(event.target.value); if (validationError) setValidationError(null) }}
            placeholder="1706.03762, https://arxiv.org/abs/… (mỗi dòng một nguồn)"
            rows={1}
            disabled={isDisabled}
          />
        </div>
        <div className="composer-footer">
          <div className="composer-attachments">
            <input ref={fileInputRef} className="sr-only" type="file" accept="application/pdf,.pdf" multiple onChange={(event) => { handleFiles(event.target.files); event.currentTarget.value = '' }} disabled={isDisabled} />
            <button type="button" className="attach-button" onClick={() => fileInputRef.current?.click()} disabled={isDisabled}><Paperclip size={15} aria-hidden="true" /> Thêm PDF</button>
            {files.map((file) => <span className="file-chip" key={`${file.name}-${file.size}`}><FileText size={13} aria-hidden="true" /><span title={file.name}>{file.name}</span><button type="button" onClick={() => setFiles((current) => current.filter((item) => item !== file))} aria-label={`Xóa ${file.name}`} disabled={isDisabled}><X size={13} /></button></span>)}
          </div>
          <button type="submit" className="submit-button" disabled={isDisabled} aria-describedby={validationError ? 'composer-error' : undefined}>
            {submitting ? <LoaderCircle size={16} className="spin" aria-hidden="true" /> : <ArrowUp size={17} strokeWidth={2.4} aria-hidden="true" />}
            <span>{submitting ? 'Đang khởi chạy…' : 'Bắt đầu scout'}</span>
          </button>
        </div>
      </div>
      {validationError ? <p id="composer-error" className="form-error" role="alert">{validationError}</p> : <p className="composer-hint"><span>Tip</span> Bạn có thể bắt đầu bằng câu hỏi, một ArXiv ID, hoặc PDF. Scout sẽ giữ lại trace để bạn kiểm tra.</p>}
    </form>
  )
}
