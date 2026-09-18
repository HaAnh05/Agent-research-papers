import { useEffect, useMemo } from 'react'
import { AlertTriangle, BookOpen, Circle, LoaderCircle } from 'lucide-react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { useResearch } from '../context/ResearchContext'
import { ChatTimeline } from '../components/chat/ChatTurn'
import { PromptBar } from '../components/chat/PromptBar'
import { artifactsForRun, messagesForView, readThreadId, readThreadTitle, type ChatInput } from '../components/chat/contracts'

function responseId(value: unknown): string | undefined {
  return readThreadId(value)
}

export function CurrentRunPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const research = useResearch()
  const currentThreadId = readThreadId(research.currentThread)
  const currentRunId = research.currentRun?.runId
  const runLoading = Boolean(research.runLoading)
  const runError = research.runError ?? (research.currentRun?.status === 'error' ? research.currentRun.error : null)
  const running = research.activeRun || research.currentRun?.status === 'running'

  useEffect(() => {
    if (!id) return
    if (currentThreadId === id || currentRunId === id) return
    void research.loadThread(id).catch(() => research.loadRun(id)).catch(() => undefined)
  }, [currentRunId, currentThreadId, id, research.loadRun, research.loadThread])

  const messages = useMemo(() => messagesForView(research.currentThread, research.currentRun), [research.currentThread, research.currentRun])
  const events = research.events ?? []
  const title = readThreadTitle(research.currentThread) ?? research.currentRun?.input.query ?? 'Current run'

  const send = async (input: ChatInput) => {
    const result = await research.sendMessage(input.query, { paperInputs: input.paperInputs, files: input.files })
    const resultId = responseId(result)
    if (resultId) navigate(`/run/${encodeURIComponent(resultId)}`)
  }

  if (runLoading && !research.currentRun && !research.currentThread) return <div className="chat-page chat-page--loading"><LoaderCircle size={18} className="spin" /> Đang mở thread…</div>
  if (runError && !research.currentRun && !research.currentThread) return <div className="chat-page chat-page--loading"><div className="chat-error" role="alert"><AlertTriangle size={16} />{runError}</div><Link className="chat-link-button" to="/">Về research desk</Link></div>

  return <div className="chat-page chat-page--run">
    <header className="chat-page__header"><div><p className="chat-eyebrow"><Circle size={10} fill="currentColor" /> Current run</p><h1>{title}</h1><p className="chat-page__subtitle">{running ? 'Scout đang xử lý request và sẽ cập nhật Thinking trong assistant turn.' : research.currentRun?.status === 'error' ? 'Workflow dừng tại một node cần kiểm tra.' : 'Thread trong bộ nhớ phiên này — có thể đọc lại và hỏi tiếp.'}</p></div><div className="chat-page__header-meta"><span className={`desk-status ${running ? 'is-live' : ''}`}><span className={`status-dot ${running ? 'is-ready' : 'is-muted'}`} aria-hidden="true" />{running ? 'Đang chạy' : research.currentRun?.status === 'error' ? 'Có lỗi' : 'Hoàn tất'}</span><button type="button" className="minimal-button" onClick={() => navigate('/')} aria-label="Về research desk"><BookOpen size={15} /></button></div></header>
    <div className="chat-page__body">
      <ChatTimeline messages={messages} events={events} running={Boolean(running)} currentNode={research.currentRun?.currentNode} activeRunId={research.currentRun?.runId} runArtifacts={artifactsForRun(research.currentRun)} />
      {runError ? <div className="chat-error" role="alert"><AlertTriangle size={16} /><span>{runError}</span></div> : null}
    </div>
    <div className="chat-page__composer"><PromptBar onSend={send} disabled={Boolean(research.activeRun)} busy={Boolean(running)} busyLabel="Đang xử lý…" /></div>
    <footer className="chat-page__footer"><span><BookOpen size={13} /> Trace và nguồn được giữ trong thread.</span>{research.currentRun?.runId ? <code>{research.currentRun.runId}</code> : null}</footer>
  </div>
}

export function CurrentRunRedirect() {
  const research = useResearch()
  const navigate = useNavigate()
  const currentId = readThreadId(research.currentThread) ?? research.currentRun?.runId
  useEffect(() => {
    navigate(currentId ? `/run/${encodeURIComponent(currentId)}` : '/', { replace: true })
  }, [currentId, navigate])
  return <div className="chat-page chat-page--loading"><LoaderCircle size={18} className="spin" /> Đang mở current run…</div>
}
