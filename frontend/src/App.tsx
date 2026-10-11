import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { AnswerView, QuickAnswer, type AnswerActions } from './components/Answer'
import { Composer } from './components/Composer'
import { GraphPanel } from './components/GraphPanel'
import { MapIcon } from './components/icons'
import { ProgressLine } from './components/ProgressLine'
import { shortTitle } from './lib/format'
import { isQuick } from './lib/run'
import { transport } from './lib/transport'
import { useChat, type SendOptions } from './lib/useChat'
import type { GraphNode, RunView, SimilarityGraph, Turn } from './types'

const EXAMPLE_PAPER = 'https://arxiv.org/abs/2605.05737'
const EXAMPLE_TOPIC = 'harness cho suy luận LLM dài hạn'
const chipName = (title: string) => title.match(/^([^:]{2,26}):/)?.[1] ?? clip(title, 30)
const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1).trimEnd()}…` : s)

function followUps(run: RunView | undefined, graph: SimilarityGraph | undefined): Array<{ text: string; display?: string; arxivId?: string; attachment?: string }> {
  const summary = run?.summary
  if (!summary) return []
  const chips: Array<{ text: string; display?: string; arxivId?: string; attachment?: string }> = []
  for (const q of summary.followUps ?? []) if (typeof q === 'string' && q.trim()) chips.push({ text: q })
  if (chips.length) return chips.slice(0, 3)
  const blocks = summary.blocks ?? []
  const insights = blocks.find((b) => b.type === 'insights') as { items?: Array<{ finding?: string }> } | undefined
  const finding = insights?.items?.[0]?.finding
  if (finding) chips.push({ text: `Giải thích: ${clip(finding.replace(/\.$/, ''), 64)}` })
  else if (blocks.some((b) => b.type === 'intuition')) chips.push({ text: 'Giải thích kỹ hơn ý tưởng cốt lõi' })
  const results = blocks.find((b) => b.type === 'results') as { source?: string[] } | undefined
  const table = results?.source?.[0]
  if (table) chips.push({ text: `${table} đo gì và đo thế nào?` })
  const next = graph?.nodes.filter((n) => n.role !== 'origin' && n.arxivId).sort((a, b) => (b.similarity ?? 0) - (a.similarity ?? 0))[0]
  if (next?.arxivId) {
    chips.push({ text: `Phân tích ${chipName(next.title)}`, arxivId: next.arxivId, attachment: `arXiv ${next.arxivId}` })
  }
  return chips.slice(0, 3)
}

function TurnView({ turn, graphNodes, actions }: { turn: Turn; graphNodes: number; actions: AnswerActions }) {
  const run = turn.run
  const quick = !!run && run.status !== 'error' && isQuick(run)
  return (
    <>
      <div className="msg-user">
        <div>
          {turn.userText}
          {turn.attachment && <span className="att">{turn.attachment}</span>}
        </div>
      </div>
      <div className="turn">
        {turn.sendError ? (
          <div className="banner err">{turn.sendError}</div>
        ) : !run ? (
          <div className="prog">
            <button type="button" tabIndex={-1}>
              <span className="pulse" />
              <span className="pl">Đang gửi…</span>
            </button>
          </div>
        ) : quick ? (
          <QuickAnswer run={run} />
        ) : (
          <>
            <ProgressLine run={run} />
            {run.status === 'error' && <div className="banner err">{run.error ?? 'Run thất bại.'}</div>}
            {run.status !== 'running' && run.status !== 'error' && run.result && <AnswerView run={run} graphNodes={graphNodes} actions={actions} />}
          </>
        )}
      </div>
    </>
  )
}

export function App() {
  const { turns, busy, send, reset } = useChat()
  const [input, setInput] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [panelOpen, setPanelOpen] = useState(false)
  const [graphTurnId, setGraphTurnId] = useState<string | null>(null)
  const [toast, setToast] = useState<{ text: string; on: boolean }>({ text: '', on: false })

  const formRef = useRef<HTMLFormElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const threadRef = useRef<HTMLDivElement>(null)
  const firstRect = useRef<DOMRect | null>(null)
  const stick = useRef(true)
  const seenGraphs = useRef(new Set<string>())
  const toastTimer = useRef<number>()

  const mode = turns.length ? 'chat' : 'landing'

  // ---- landing -> chat: FLIP the composer into the bottom bar ----
  useLayoutEffect(() => {
    const first = firstRect.current
    const el = formRef.current
    if (mode !== 'chat' || !first || !el) return
    firstRect.current = null
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const last = el.getBoundingClientRect()
    el.animate(
      [
        { transform: `translate(${first.left - last.left}px, ${first.top - last.top}px)`, height: `${first.height}px` },
        { transform: 'none', height: `${last.height}px` },
      ],
      { duration: 520, easing: 'cubic-bezier(.2,.8,.2,1)' },
    )
  }, [mode])

  // ---- auto-open the graph panel when a run delivers one ----
  useEffect(() => {
    for (const t of turns) {
      const run = t.run
      if (run?.graph && run.status !== 'running' && !seenGraphs.current.has(t.id)) {
        seenGraphs.current.add(t.id)
        setGraphTurnId(t.id)
        requestAnimationFrame(() => requestAnimationFrame(() => setPanelOpen(true)))
      }
    }
  }, [turns])

  const graphTurn = turns.find((t) => t.id === graphTurnId)
  const graph = graphTurn?.run?.graph
  const lastWithSummary = [...turns].reverse().find((t) => t.run?.summary && t.run.status !== 'running')
  const lastTurn = turns[turns.length - 1]
  const hasResult = !!lastTurn?.run && lastTurn.run.status !== 'running' && lastTurn.run.status !== 'error' && (!!lastTurn.run.result || !!lastTurn.run.answerText)
  const chips = useMemo(
    () => (hasResult && !busy ? followUps(lastWithSummary?.run, lastWithSummary?.run?.graph) : []),
    [hasResult, busy, lastWithSummary],
  )

  const firstSummary = turns.find((t) => t.run?.summary?.paper?.title)?.run?.summary?.paper?.title
  const convTitle = firstSummary ? clip(firstSummary, 64) : turns[0] ? clip(turns[0].userText, 64) : ''

  // ---- keep the newest content in view unless the reader scrolled up ----
  useEffect(() => {
    const el = threadRef.current
    if (el && stick.current) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [turns])

  const notify = useCallback((text: string) => {
    setToast({ text, on: true })
    window.clearTimeout(toastTimer.current)
    toastTimer.current = window.setTimeout(() => setToast((t) => ({ ...t, on: false })), 1800)
  }, [])

  const submit = useCallback(
    (text: string, options?: SendOptions) => {
      if (busy) {
        notify('Đang có một lượt phân tích chạy. Chờ lượt này xong.')
        return
      }
      if (!turns.length && formRef.current) firstRect.current = formRef.current.getBoundingClientRect()
      stick.current = true
      void send(text, options)
    },
    [busy, turns.length, send, notify],
  )

  const onComposerSubmit = () => {
    const text = input
    const attached = file
    setInput('')
    setFile(null)
    submit(text, { file: attached })
  }

  const actions: AnswerActions = {
    onMap: () => setPanelOpen(true),
    onDownload: async (reportId) => {
      try {
        const blob = await transport.downloadReport(reportId)
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = 'report.md'
        a.click()
        setTimeout(() => URL.revokeObjectURL(url), 1000)
        notify('Đã tải report.md')
      } catch {
        notify('Không tải được report.md')
      }
    },
    onCopyBibtex: async (bibtex) => {
      try {
        await navigator.clipboard.writeText(bibtex)
        notify('Đã sao chép BibTeX')
      } catch {
        notify('Không sao chép được BibTeX')
      }
    },
  }

  const ask = (text: string) => {
    setInput(text)
    textareaRef.current?.focus()
  }
  const analyse = (node: GraphNode) => {
    if (!node.arxivId) return
    submit(`Phân tích ${chipName(node.title)}`, { arxivId: node.arxivId, attachment: `arXiv ${node.arxivId}` })
  }
  const newChat = () => {
    if (busy) return
    reset()
    seenGraphs.current.clear()
    setGraphTurnId(null)
    setPanelOpen(false)
    setInput('')
    setFile(null)
  }

  return (
    <>
      <div className={`frame${panelOpen && graph ? ' panel-open' : ''}`}>
        <div className={`main ${mode}${hasResult ? ' has-result' : ''}`}>
          <header className="top">
            <button type="button" className="brand" onClick={newChat} aria-label="Research Scout, bắt đầu cuộc trò chuyện mới">
              Research Scout
            </button>
            <span className="conv">{convTitle}</span>
            <span className="spacer" />
            {graph && mode === 'chat' && (
              <button type="button" className="ghost mapbtn" aria-controls="panel" aria-expanded={panelOpen} onClick={() => setPanelOpen((v) => !v)}>
                <MapIcon width={16} height={16} /> Bản đồ liên quan
              </button>
            )}
          </header>

          <div className="stage">
            <section className="hero" aria-labelledby="h">
              <small>Research Scout</small>
              <h1 id="h">Bạn muốn hiểu điều gì?</h1>
              <p>Đưa tôi một bài báo, một chủ đề hoặc một file PDF.</p>
            </section>

            <div
              className="thread"
              ref={threadRef}
              aria-live="polite"
              onScroll={(e) => {
                const el = e.currentTarget
                stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 160
              }}
            >
              <div className="col">
                {turns.map((t) => (
                  <TurnView key={t.id} turn={t} graphNodes={t.run?.graph ? t.run.graph.nodes.filter((n) => n.role !== 'origin').length : 0} actions={{ ...actions, onMap: () => { setGraphTurnId(t.id); setPanelOpen(true) } }} />
                ))}
              </div>
            </div>

            <div className="dock">
              {chips.length > 0 && (
                <div className="follow">
                  {chips.map((c) => (
                    <button key={c.text} type="button" className="chip" onClick={() => submit(c.text, { arxivId: c.arxivId, attachment: c.attachment })}>
                      {c.text}
                    </button>
                  ))}
                </div>
              )}
              <Composer formRef={formRef} textareaRef={textareaRef} value={input} onChange={setInput} file={file} onFile={setFile} onSubmit={onComposerSubmit} busy={busy} />
              <div className="hint modes">
                <button type="button" className="chip" onClick={() => ask(EXAMPLE_PAPER)}>
                  Phân tích một bài
                </button>
                <button type="button" className="chip" onClick={() => ask(EXAMPLE_TOPIC)}>
                  Tìm theo chủ đề
                </button>
              </div>
              <div className="note">Số liệu trong báo cáo được kiểm tra với bảng của paper. Phần diễn giải do AI viết và có thể sai.</div>
            </div>
          </div>
        </div>

        {graph && (
          <GraphPanel key={graphTurnId} graph={graph} onClose={() => setPanelOpen(false)} onAsk={ask} onAnalyse={analyse} />
        )}
      </div>
      <div className={`toast${toast.on ? ' on' : ''}`} role="status">
        {toast.text}
      </div>
    </>
  )
}
