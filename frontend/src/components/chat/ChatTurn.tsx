/*
 * ChatTurn keeps the Chat catalog's conversational turn anatomy (author row,
 * content, then tool/artifact chips) while feeding it only Research Scout
 * state. No assistant text is fabricated while a run is active. See
 * https://www.beautifului.dev/r/registry.json and frontend/NOTICE.
 */
import { BookOpen, Circle, LoaderCircle } from 'lucide-react'
import { MarkdownContent } from '../MarkdownContent'
import type { TraceEvent } from '../../types'
import { ArtifactLinks } from './ArtifactLinks'
import type { ChatMessageModel } from './contracts'
import { OperationalThinking } from './OperationalThinking'

interface ChatTurnProps {
  message: ChatMessageModel
  events?: TraceEvent[]
  running?: boolean
  currentNode?: string | null
  runArtifacts?: ChatMessageModel['artifacts']
}

function statusCopy(message: ChatMessageModel, running: boolean): string {
  if (message.status === 'error') return 'Không hoàn tất'
  if (running || message.status === 'running') return 'Đang xử lý'
  if (message.status === 'pending') return 'Đang chờ'
  return 'Research Scout'
}

export function ChatTurn({ message, events = [], running = false, currentNode, runArtifacts = [] }: ChatTurnProps) {
  if (message.role === 'system') return null
  const isUser = message.role === 'user'
  const artifacts = [...message.artifacts, ...runArtifacts]

  return (
    <article className={`chat-turn chat-turn--${isUser ? 'user' : 'assistant'}`}>
      <div className="chat-turn__author">
        <span className={`chat-avatar ${isUser ? 'chat-avatar--user' : 'chat-avatar--assistant'}`} aria-hidden="true">
          {isUser ? <Circle size={10} fill="currentColor" /> : <BookOpen size={15} strokeWidth={1.7} />}
        </span>
        <span className="chat-turn__name">{isUser ? 'Bạn' : 'Research Scout'}</span>
        <span className="chat-turn__status">{statusCopy(message, running && !isUser)}</span>
      </div>
      <div className="chat-turn__body">
        {message.content ? <MarkdownContent content={message.content} compact={isUser} /> : null}
        {!isUser && running && !message.content ? <div className="chat-turn__working" role="status" aria-live="polite"><LoaderCircle size={15} className="spin" /> Đang xử lý request qua workflow…</div> : null}
        {!isUser && events.length ? <div className="chat-turn__thinking"><OperationalThinking events={events} running={running} currentNode={currentNode} /></div> : null}
        {!isUser ? <ArtifactLinks artifacts={artifacts} /> : null}
      </div>
    </article>
  )
}

export function ChatTimeline({ messages, events, running, currentNode, activeRunId, runArtifacts }: { messages: ChatMessageModel[]; events: TraceEvent[]; running: boolean; currentNode?: string | null; activeRunId?: string; runArtifacts?: ChatMessageModel['artifacts'] }) {
  if (!messages.length) return <div className="chat-empty" aria-live="polite"><div className="chat-empty__mark" aria-hidden="true"><BookOpen size={20} /></div><h1>Annotated Research Desk</h1><p>Bắt đầu bằng một câu hỏi, ArXiv ID hoặc PDF. Mỗi câu trả lời sẽ giữ lại nguồn và trace vận hành để bạn kiểm tra.</p></div>

  const lastAssistantIndex = messages.map((message) => message.role).lastIndexOf('assistant')
  return <div className="chat-timeline" aria-label="Cuộc hội thoại nghiên cứu">{messages.map((message, index) => {
    const isActive = message.role === 'assistant' && (message.runId === activeRunId || message.status === 'running' || (running && index === lastAssistantIndex))
    return <ChatTurn key={message.id || `${message.role}-${index}`} message={message} events={isActive ? events : []} running={isActive && running} currentNode={isActive ? currentNode : null} runArtifacts={isActive ? runArtifacts : []} />
  })}</div>
}
