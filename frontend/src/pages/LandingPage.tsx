import { useEffect, useRef, useState } from 'react'
import { ArrowUp, Paperclip, X } from 'lucide-react'
import { useLocation, useNavigate } from 'react-router-dom'
import { ApiError } from '../lib/api'
import { useResearch } from '../context/ResearchContext'
import type { RunInput } from '../types'

const MAX_UPLOAD_BYTES = 25 * 1024 * 1024
const ARXIV_ID = /(?:arxiv:\s*)?\d{4}\.\d{4,5}(?:v\d+)?/gi
const ARXIV_URL = /^https?:\/\/(?:www\.|export\.)?arxiv\.org\/(?:abs|pdf)\/(\d{4}\.\d{4,5}(?:v\d+)?)(?:\.pdf)?\/?$/i

export function parseRequest(text: string): { query: string; paperInputs: string[]; error: string | null } {
  const value = text.trim()
  if (!value) return { query: '', paperInputs: [], error: null }
  const urls = value.match(/https?:\/\/[^\s,;]+/gi) ?? []
  for (const raw of urls) {
    const url = raw.replace(/[).,;]+$/, '')
    if (!ARXIV_URL.test(url)) return { query: value, paperInputs: [], error: 'Only public arXiv paper URLs are supported. Attach other PDFs as files.' }
  }
  const hasLocalPdfName = value.split(/[\s,;]+/).some((token) =>
    !/^https?:\/\//i.test(token) && /\.pdf$/i.test(token.replace(/[).]+$/, '')),
  )
  if (hasLocalPdfName || /(?:^|\s)(?:[./~]|[A-Za-z]:\\)[^\s]*\.pdf\b/i.test(value)) {
    return { query: value, paperInputs: [], error: 'Local PDF paths cannot be read by the browser. Attach the PDF instead.' }
  }
  const inputs = [...new Set([
    ...urls.map((url) => url.replace(/[).,;]+$/, '')).map((url) => ARXIV_URL.exec(url)?.[1]).filter((id): id is string => Boolean(id)),
    ...(value.replace(/https?:\/\/[^\s,;]+/gi, '').match(ARXIV_ID) ?? []).map((id) => id.replace(/^arxiv:\s*/i, '')),
  ])]
  const withoutRefs = value.replace(/https?:\/\/[^\s,;]+/gi, '').replace(ARXIV_ID, '').replace(/\b(?:and|vs|versus)\b/gi, '').replace(/[\s,;]+/g, '')
  return { query: withoutRefs ? value : '', paperInputs: inputs, error: null }
}

async function validatePdf(file: File): Promise<string | null> {
  if (!file.name.toLowerCase().endsWith('.pdf') || (file.type && !['application/pdf', 'application/octet-stream'].includes(file.type))) {
    return `“${file.name}” is not a supported PDF file.`
  }
  if (file.size > MAX_UPLOAD_BYTES) return `“${file.name}” exceeds the 25 MB PDF limit.`
  try {
    const header = new Uint8Array(await file.slice(0, 5).arrayBuffer())
    if (String.fromCharCode(...header) !== '%PDF-') return `“${file.name}” is not a valid PDF file.`
  } catch {
    return `“${file.name}” could not be read. Choose the file again.`
  }
  return null
}

function activeRunId(error: ApiError): string | null {
  if (error.status !== 409 || !error.body || typeof error.body !== 'object') return null
  const detail = (error.body as { detail?: unknown }).detail
  if (!detail || typeof detail !== 'object') return null
  const id = (detail as { activeRunId?: unknown }).activeRunId
  return typeof id === 'string' ? id : null
}

export function LandingPage() {
  const research = useResearch()
  const navigate = useNavigate()
  const location = useLocation()
  const draft = (location.state as { draft?: RunInput } | null)?.draft
  const [text, setText] = useState(() => [draft?.query, ...(draft?.paperInputs ?? [])].filter(Boolean).join('\n'))
  const [files, setFiles] = useState<File[]>(draft?.files ?? [])
  const [checking, setChecking] = useState(false)
  const [fileError, setFileError] = useState<string | null>(null)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [showStarting, setShowStarting] = useState(false)
  const [placeholder, setPlaceholder] = useState('Ask about a paper or topic…')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const formRef = useRef<HTMLFormElement>(null)
  const pendingRef = useRef(false)
  const request = parseRequest(text)
  const validation = fileError ?? request.error ?? (!text.trim() && files.length === 0 ? 'Enter a topic, arXiv reference, or attach a PDF.' : null)
  const visibleError = submitError ?? validation
  const disabled = Boolean(validation || checking || submitting || research.activeRun)

  useEffect(() => { textareaRef.current?.focus() }, [])

  useEffect(() => {
    if (!draft) return
    setText([draft.query, ...draft.paperInputs].filter(Boolean).join('\n'))
    setFiles(draft.files ?? [])
    setFileError(null)
    setSubmitError(null)
    window.requestAnimationFrame(() => textareaRef.current?.focus())
  }, [draft])

  async function addFiles(selection: FileList | null) {
    if (!selection?.length) return
    setChecking(true)
    setFileError(null)
    const chosen = [...selection]
    for (const file of chosen) {
      const error = await validatePdf(file)
      if (error) { setFileError(error); setChecking(false); return }
    }
    setFiles((existing) => [...existing, ...chosen.filter((file) => !existing.some((item) => item.name === file.name && item.size === file.size))])
    setChecking(false)
  }

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (pendingRef.current || disabled) return
    pendingRef.current = true
    setSubmitError(null)
    setSubmitting(true)
    const reveal = window.setTimeout(() => setShowStarting(true), 150)
    try {
      const originalInput = { query: request.query, paperInputs: request.paperInputs, files }
      const [snapshot] = await Promise.all([
        research.startRun(originalInput),
        new Promise<void>((resolve) => window.setTimeout(resolve, 180)),
      ])
      navigate(`/run/${encodeURIComponent(snapshot.runId)}`, { state: { originalInput } })
    } catch (error) {
      const runningId = error instanceof ApiError ? activeRunId(error) : null
      if (runningId) navigate(`/run/${encodeURIComponent(runningId)}`)
      else setSubmitError(error instanceof Error ? error.message : 'Could not start research. Try again.')
      setSubmitting(false)
      setShowStarting(false)
      textareaRef.current?.focus()
    } finally {
      window.clearTimeout(reveal)
      pendingRef.current = false
    }
  }

  function suggestion(nextPlaceholder: string) {
    setPlaceholder(nextPlaceholder)
    textareaRef.current?.focus()
  }

  return <main id="main-content" className={`landing ${submitting ? 'landing--leaving' : ''}`}>
    {!showStarting ? <div className="landing__inner">
      <div className="landing__intro">
        <p className="landing__brand">Research Scout</p>
        <h1>What do you want to understand?</h1>
        <p className="landing__subtitle">Give me a paper, topic or PDF.</p>
      </div>
      <form ref={formRef} className="research-prompt" onSubmit={submit} noValidate>
        <label className="sr-only" htmlFor="research-request">Research request</label>
        <textarea ref={textareaRef} id="research-request" value={text} placeholder={placeholder} rows={1}
          aria-describedby={visibleError ? 'research-prompt-error' : undefined}
          onChange={(event) => { setText(event.target.value); setSubmitError(null) }}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault()
              formRef.current?.requestSubmit()
            }
          }} disabled={submitting} />
        <div className="research-prompt__footer">
          <div className="research-prompt__attachments">
            <input ref={fileRef} className="sr-only" type="file" accept="application/pdf,.pdf" multiple
              onChange={(event) => { void addFiles(event.target.files); event.currentTarget.value = '' }} />
            <button type="button" className="research-prompt__attach" onClick={() => fileRef.current?.click()} aria-label="Attach PDF" disabled={checking || submitting}>
              <Paperclip size={17} strokeWidth={1.65} aria-hidden="true" />
            </button>
            {files.map((file) => <span className="research-prompt__file" key={`${file.name}:${file.size}`}>
              <span title={file.name}>{file.name}</span>
              <button type="button" aria-label={`Remove ${file.name}`} onClick={() => { setFiles((current) => current.filter((item) => item !== file)); setFileError(null) }} disabled={submitting}><X size={15} aria-hidden="true" /></button>
            </span>)}
            {checking ? <span className="research-prompt__checking">Checking PDF…</span> : null}
          </div>
          <button type="submit" className="research-prompt__submit" aria-label="Start research" disabled={disabled}><ArrowUp size={18} strokeWidth={1.7} aria-hidden="true" /></button>
        </div>
      </form>
      {visibleError ? <p id="research-prompt-error" className="landing__error" role="alert">{visibleError}</p> : null}
      <div className="landing__suggestions" aria-label="Request examples">
        <button type="button" onClick={() => suggestion('Paste an arXiv ID or link…')}>Analyze paper</button>
        <button type="button" onClick={() => suggestion('Describe a research topic…')}>Scout literature</button>
        <button type="button" onClick={() => suggestion('Paste two arXiv references…')}>Compare papers</button>
      </div>
      {research.activeRun && research.currentRun?.runId ? <button type="button" className="landing__active" onClick={() => navigate(`/run/${encodeURIComponent(research.currentRun!.runId)}`)}>A research run is active · View progress</button> : null}
    </div> : <div className="landing__starting" role="status"><h1>Running research</h1><p>Submitting your request…</p></div>}
  </main>
}
