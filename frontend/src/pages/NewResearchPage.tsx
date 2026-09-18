import { useMemo } from 'react'
import { AlertTriangle, BookOpen, ChevronDown, ShieldCheck } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useResearch } from '../context/ResearchContext'
import { ChatTimeline } from '../components/chat/ChatTurn'
import { PromptBar } from '../components/chat/PromptBar'
import { artifactsForRun, messagesForView, readThreadId, readThreadTitle, type ChatInput } from '../components/chat/contracts'

function responseId(value: unknown): string | undefined {
  return readThreadId(value)
}

export function NewResearchPage() {
  const navigate = useNavigate()
  const research = useResearch()
  const messages = useMemo(() => messagesForView(research.currentThread, research.currentRun), [research.currentThread, research.currentRun])
  const events = research.events ?? []
  const running = research.activeRun || research.currentRun?.status === 'running'
  const title = readThreadTitle(research.currentThread) ?? (research.currentRun?.input.query || 'Bắt đầu một research scout')

  const send = async (input: ChatInput) => {
    const result = await research.sendMessage(input.query, { paperInputs: input.paperInputs, files: input.files })
    const id = responseId(result)
    if (id) navigate(`/run/${encodeURIComponent(id)}`)
  }

  return <div className="chat-page chat-page--home">
    <header className="chat-page__header"><div><p className="chat-eyebrow"><BookOpen size={13} /> Annotated Research Desk</p><h1>{title}</h1><p className="chat-page__subtitle">Trao đổi với Scout về paper, phương pháp và bằng chứng — mỗi câu trả lời giữ lại nguồn để đọc lại.</p></div><div className="chat-page__header-meta"><span className="desk-status"><span className={`status-dot ${running ? 'is-ready' : 'is-muted'}`} aria-hidden="true" />{running ? 'Đang chạy' : 'Sẵn sàng'}</span></div></header>
    <div className="chat-page__body">
      <ChatTimeline messages={messages} events={events} running={Boolean(running)} currentNode={research.currentRun?.currentNode} activeRunId={research.currentRun?.runId} runArtifacts={artifactsForRun(research.currentRun)} />
      {research.runError || research.currentRun?.status === 'error' ? <div className="chat-error" role="alert"><AlertTriangle size={16} /><span>{research.runError ?? research.currentRun?.error ?? 'Workflow gặp lỗi. Mở Thinking để xem node cuối.'}</span></div> : null}
    </div>
    <div className="chat-page__composer"><PromptBar onSend={send} disabled={Boolean(research.activeRun)} busy={Boolean(running)} busyLabel="Đang xử lý…" /></div>
    <div className="chat-page__footer"><span><ShieldCheck size={13} /> Secret chỉ nằm ở backend.</span><span><ChevronDown size={12} /> Operational trace có thể mở lại trong assistant turn.</span></div>
  </div>
}
