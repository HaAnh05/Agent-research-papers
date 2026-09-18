/*
 * PromptBar follows the compose-first interaction from Beautiful UI's Prompt
 * Bar catalog (https://www.beautifului.dev/r/registry.json): a single paper
 * surface, source attachments in the footer, and one explicit submit action.
 * Demo streaming/timers are intentionally absent. See frontend/NOTICE.
 */
import { useEffect, useRef, useState } from 'react'
import { ArrowUp, FileText, Link2, LoaderCircle, Paperclip, X } from 'lucide-react'
import { isPdfFile, validateResearchInput } from '../ResearchComposer'
import type { ChatInput } from './contracts'

interface PromptBarProps {
  onSend: (input: ChatInput) => Promise<void>
  disabled?: boolean
  busy?: boolean
  busyLabel?: string
  draft?: string | null
  onDraftConsumed?: () => void
}

function parsePaperInputs(value: string): string[] {
  return value.split(/[\n,]+/).map((item) => item.trim()).filter(Boolean)
}

function detectPaperInputs(value: string): string[] {
  const matches = value.match(/(?:arxiv:\s*)?\d{4}\.\d{4,5}(?:v\d+)?|https?:\/\/arxiv\.org\/[^\s,]+/gi) ?? []
  return matches.map((match) => match.replace(/[).,;]+$/, '').trim())
}

export function PromptBar({ onSend, disabled = false, busy = false, busyLabel = 'Đang xử lý…', draft = null, onDraftConsumed }: PromptBarProps) {
  const [query, setQuery] = useState('')
  const [paperText, setPaperText] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [sourcesOpen, setSourcesOpen] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const consumedDraft = useRef<string | null>(null)
  const isBusy = disabled || busy || submitting

  useEffect(() => {
    if (!draft) {
      consumedDraft.current = null
      return
    }
    if (consumedDraft.current === draft) return
    consumedDraft.current = draft
    setQuery(draft)
    onDraftConsumed?.()
  }, [draft, onDraftConsumed])

  useEffect(() => {
    const fillDraft = (event: Event) => {
      const value = (event as CustomEvent<string>).detail
      if (!value) return
      setQuery(value)
      textareaRef.current?.focus()
    }
    window.addEventListener('research-scout:draft', fillDraft)
    return () => window.removeEventListener('research-scout:draft', fillDraft)
  }, [])

  const addFiles = (selection: FileList | null) => {
    if (!selection) return
    const next = [...selection]
    const invalid = next.find((file) => !isPdfFile(file))
    if (invalid) {
      setError(`“${invalid.name}” không phải file PDF được hỗ trợ.`)
      return
    }
    setError(null)
    setFiles((current) => [...current, ...next.filter((file) => !current.some((item) => item.name === file.name && item.size === file.size))])
  }

  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const paperInputs = [...new Set([...parsePaperInputs(paperText), ...detectPaperInputs(query)])]
    const validation = validateResearchInput(query, paperInputs, files)
    if (validation) {
      setError(validation)
      return
    }
    setError(null)
    setSubmitting(true)
    try {
      await onSend({ query: query.trim(), paperInputs, files })
      setQuery('')
      setPaperText('')
      setFiles([])
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : 'Không thể gửi yêu cầu.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form className="prompt-bar" onSubmit={submit} noValidate onKeyDown={(event) => {
      if (event.key === 'Enter' && (event.metaKey || event.ctrlKey) && !isBusy) {
        event.preventDefault()
        event.currentTarget.requestSubmit()
      }
    }}>
      <div className="prompt-bar__surface">
        <label className="prompt-bar__label" htmlFor="chat-prompt">Bạn muốn tìm hiểu điều gì?</label>
        <textarea
          ref={textareaRef}
          id="chat-prompt"
          className="prompt-bar__textarea"
          value={query}
          onChange={(event) => { setQuery(event.target.value); if (error) setError(null) }}
          placeholder="Đặt một câu hỏi về paper, phương pháp hoặc benchmark…"
          rows={2}
          disabled={isBusy}
          autoComplete="off"
          aria-describedby={error ? 'chat-prompt-error' : 'chat-prompt-hint'}
        />
        <details className="prompt-bar__sources" open={sourcesOpen} onToggle={(event) => setSourcesOpen(event.currentTarget.open)}>
          <summary><Link2 size={14} aria-hidden="true" /> <span>Thêm nguồn đã biết</span><small>ArXiv ID hoặc URL</small></summary>
          <div className="prompt-bar__source-row">
            <label htmlFor="chat-paper-inputs">ArXiv ID hoặc URL</label>
            <textarea
              id="chat-paper-inputs"
              value={paperText}
              onChange={(event) => { setPaperText(event.target.value); if (error) setError(null) }}
              placeholder="1706.03762 hoặc https://arxiv.org/abs/…"
              rows={1}
              disabled={isBusy}
            />
          </div>
        </details>
        <div className="prompt-bar__footer">
          <div className="prompt-bar__attachments">
            <input ref={fileInputRef} className="sr-only" type="file" accept="application/pdf,.pdf" multiple onChange={(event) => { addFiles(event.target.files); event.currentTarget.value = '' }} disabled={isBusy} />
            <button type="button" className="prompt-bar__attach" onClick={() => fileInputRef.current?.click()} disabled={isBusy}><Paperclip size={14} aria-hidden="true" /> Thêm PDF</button>
            {files.map((file) => <span className="prompt-file-chip" key={`${file.name}-${file.size}`}><FileText size={12} aria-hidden="true" /><span title={file.name}>{file.name}</span><button type="button" onClick={() => setFiles((current) => current.filter((item) => item !== file))} aria-label={`Xóa ${file.name}`} disabled={isBusy}><X size={12} /></button></span>)}
          </div>
          <button type="submit" className="prompt-bar__submit" disabled={isBusy} aria-label="Bắt đầu scout · Gửi" aria-describedby={error ? 'chat-prompt-error' : undefined}>
            {isBusy ? <LoaderCircle size={16} className="spin" aria-hidden="true" /> : <ArrowUp size={17} strokeWidth={2.4} aria-hidden="true" />}
            <span>{isBusy ? busyLabel : 'Gửi'}</span>
          </button>
        </div>
      </div>
      {error ? <p id="chat-prompt-error" className="prompt-bar__error" role="alert">{error}</p> : <p id="chat-prompt-hint" className="prompt-bar__hint"><span>Tip</span> <kbd>⌘</kbd><kbd>Enter</kbd> để gửi nhanh. Thêm paper hoặc PDF nếu đã có nguồn.</p>}
    </form>
  )
}
