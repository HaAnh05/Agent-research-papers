import { useCallback, useEffect, useReducer, useRef } from 'react'
import type { RunSnapshot, RunView, TraceEvent, Turn } from '../types'
import { ApiError, normalizeSnapshot } from './api'
import { applyEvent, runFromSnapshot } from './run'
import { transport } from './transport'

type Action =
  | { type: 'add'; turn: Turn }
  | { type: 'run'; turnId: string; run: RunView }
  | { type: 'event'; turnId: string; event: TraceEvent }
  | { type: 'fail'; turnId: string; message: string }
  | { type: 'reset' }

function reducer(turns: Turn[], action: Action): Turn[] {
  switch (action.type) {
    case 'add':
      return [...turns, action.turn]
    case 'reset':
      return []
    case 'fail':
      return turns.map((t) => (t.id === action.turnId ? { ...t, sendError: action.message } : t))
    case 'run':
      return turns.map((t) => {
        if (t.id !== action.turnId) return t
        // A snapshot is authoritative, except when it is older than what the stream already delivered.
        const current = t.run
        if (current && !action.run.result && action.run.lastSeq < current.lastSeq) return t
        return { ...t, run: action.run }
      })
    case 'event':
      return turns.map((t) => (t.id === action.turnId && t.run ? { ...t, run: applyEvent(t.run, action.event) } : t))
  }
}

const TERMINAL = new Set(['run.completed', 'run.failed'])
let counter = 0
const uid = () => `t${Date.now().toString(36)}${counter++}`

export interface SendOptions {
  file?: File | null
  /** Text for the user bubble when it differs from what is sent to the API. */
  display?: string
  attachment?: string
  /** Analyse this arXiv paper (form field `arxivId`); `text` is then only a readable query. */
  arxivId?: string
}

export function useChat() {
  const [turns, dispatch] = useReducer(reducer, [])
  const threadRef = useRef<string | null>(null)
  const closers = useRef<Array<() => void>>([])
  const turnsRef = useRef(turns)
  turnsRef.current = turns

  useEffect(() => () => closers.current.forEach((close) => close()), [])

  const busy = turns.some((t) => (!t.run && !t.sendError) || t.run?.status === 'running')

  const attach = useCallback((turnId: string, runId: string, afterSeq: number) => {
    let finished = false
    let close: () => void = () => undefined
    const settle = () => {
      if (finished) return
      finished = true
      close()
      transport
        .getRun(runId)
        .then((snap: RunSnapshot) => dispatch({ type: 'run', turnId, run: runFromSnapshot(snap) }))
        .catch(() => undefined)
    }
    close = transport.subscribe(
      runId,
      afterSeq,
      (frame) => {
        if (frame.type === 'snapshot') {
          const run = runFromSnapshot(normalizeSnapshot(frame.data))
          dispatch({ type: 'run', turnId, run })
          if (run.status !== 'running') settle()
          return
        }
        dispatch({ type: 'event', turnId, event: frame.data as TraceEvent })
        if (TERMINAL.has(frame.type)) settle()
      },
      () => {
        // EventSource retries by itself; check once whether the run ended while the stream was down.
        setTimeout(() => {
          if (finished) return
          transport
            .getRun(runId)
            .then((snap) => {
              if (snap.status !== 'running') settle()
            })
            .catch(() => undefined)
        }, 2000)
      },
    )
    closers.current.push(() => {
      finished = true
      close()
    })
  }, [])

  const send = useCallback(
    async (text: string, options: SendOptions = {}) => {
      const content = text.trim()
      if (!content && !options.file) return false
      const turnId = uid()
      dispatch({
        type: 'add',
        turn: { id: turnId, userText: options.display ?? (content || 'Phân tích file PDF'), attachment: options.attachment ?? options.file?.name },
      })
      try {
        if (!threadRef.current) threadRef.current = await transport.createThread()
        const snap = await transport.sendMessage(threadRef.current, content, options.file, options.arxivId)
        dispatch({ type: 'run', turnId, run: runFromSnapshot(snap) })
        attach(turnId, snap.runId, snap.seq)
      } catch (error) {
        const message =
          error instanceof ApiError && error.status === 409
            ? 'Đang có một lượt phân tích chạy. Chờ lượt này xong rồi gửi lại.'
            : error instanceof Error
              ? error.message
              : 'Không gửi được tin nhắn.'
        dispatch({ type: 'fail', turnId, message })
      }
      return true
    },
    [attach],
  )

  const reset = useCallback(() => {
    closers.current.forEach((close) => close())
    closers.current = []
    threadRef.current = null
    dispatch({ type: 'reset' })
  }, [])

  return { turns, busy, send, reset }
}
