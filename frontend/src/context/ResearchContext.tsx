import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type PropsWithChildren } from 'react'
import {
  ApiError, createRun, createThread, getConfig, getRun, getThread, getThreads,
  normalizeAssistantEvent, normalizeSnapshot, normalizeTraceEvent, reconcileEvents,
  sendThreadMessage, subscribeToRun,
} from '../lib/api'
import type {
  AppConfig, RunInput, RunSnapshot, SendMessageInput, ServerRunEvent,
  Thread, ThreadDetail, TraceEvent,
} from '../types'

type ConnectionState = 'idle' | 'live' | 'polling' | 'lost' | 'closed'

interface ResearchContextValue {
  config: AppConfig | null
  configLoading: boolean
  configError: string | null
  threads: Thread[]
  currentThread: ThreadDetail | null
  currentRun: RunSnapshot | null
  events: TraceEvent[]
  runLoading: boolean
  runError: string | null
  connectionState: ConnectionState
  activeRun: boolean
  refreshConfig: () => Promise<void>
  refreshThreads: () => Promise<Thread[]>
  newThread: () => Promise<ThreadDetail>
  selectThread: (threadId: string) => void
  loadThread: (threadId: string) => Promise<ThreadDetail>
  sendMessage: (content: string, metadata?: Partial<SendMessageInput>) => Promise<RunSnapshot>
  startRun: (input: RunInput) => Promise<RunSnapshot>
  loadRun: (runId: string) => Promise<RunSnapshot>
  clearRun: () => void
}

const ResearchContext = createContext<ResearchContextValue | undefined>(undefined)

function isOperational(event: TraceEvent): boolean {
  return !event.type.startsWith('assistant.') && !event.type.startsWith('assistant_')
}

function eventRunId(value: ServerRunEvent | RunSnapshot | TraceEvent, fallback: string): string {
  return value.runId ?? ('run_id' in value ? value.run_id : undefined) ?? fallback
}

function isSnapshot(value: ServerRunEvent | RunSnapshot | TraceEvent): boolean {
  return 'traceEvents' in value || 'trace_events' in value || value.type === 'snapshot'
}

export function ResearchProvider({ children }: PropsWithChildren) {
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [configLoading, setConfigLoading] = useState(true)
  const [configError, setConfigError] = useState<string | null>(null)
  const [threads, setThreads] = useState<Thread[]>([])
  const [currentThread, setCurrentThread] = useState<ThreadDetail | null>(null)
  const [currentRun, setCurrentRun] = useState<RunSnapshot | null>(null)
  const [runLoading, setRunLoading] = useState(false)
  const [runError, setRunError] = useState<string | null>(null)
  const [connectionState, setConnectionState] = useState<ConnectionState>('idle')
  const [activeRunId, setActiveRunId] = useState<string | null>(null)
  const selectedThreadRef = useRef<string | null>(null)
  const selectedRunRef = useRef<string | null>(null)
  const activeRunRef = useRef<string | null>(null)
  const runsRef = useRef<Map<string, RunSnapshot>>(new Map())
  const lastSeqRef = useRef<Map<string, number>>(new Map())
  const closeStreamRef = useRef<(() => void) | null>(null)
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const closeStream = useCallback(() => {
    closeStreamRef.current?.()
    closeStreamRef.current = null
    if (pollingRef.current) clearInterval(pollingRef.current)
    pollingRef.current = null
  }, [])

  const showSnapshot = useCallback((snapshot: RunSnapshot) => {
    const previous = runsRef.current.get(snapshot.runId)
    const selected = selectedRunRef.current === snapshot.runId ||
      activeRunRef.current === snapshot.runId ||
      (Boolean(snapshot.threadId ?? previous?.threadId) && selectedThreadRef.current === (snapshot.threadId ?? previous?.threadId))
    // An in-flight poll/SSE response from a previous route must not resurrect
    // its run after the user opened a different run or cleared the workspace.
    if (!selected) return
    if (previous && (snapshot.seq ?? 0) < (previous.seq ?? 0)) return
    runsRef.current.set(snapshot.runId, snapshot)
    lastSeqRef.current.set(snapshot.runId, Math.max(lastSeqRef.current.get(snapshot.runId) ?? 0, snapshot.seq ?? 0))
    const visible = selectedRunRef.current === snapshot.runId ||
      (Boolean(snapshot.threadId) && selectedThreadRef.current === snapshot.threadId)
    if (visible) setCurrentRun(snapshot)
    if (snapshot.status === 'running') {
      setRunError(null)
      activeRunRef.current = snapshot.runId
      setActiveRunId(snapshot.runId)
    } else if (activeRunRef.current === snapshot.runId) {
      activeRunRef.current = null
      setActiveRunId(null)
      closeStream()
      setConnectionState('closed')
    }
  }, [closeStream])

  const refreshConfig = useCallback(async () => {
    setConfigLoading(true)
    try { setConfig(await getConfig()); setConfigError(null) }
    catch (error) { setConfigError(error instanceof Error ? error.message : 'Không thể đọc cấu hình provider.') }
    finally { setConfigLoading(false) }
  }, [])

  const refreshThreads = useCallback(async () => {
    const list = await getThreads()
    setThreads(list)
    return list
  }, [])

  const refreshSelectedThread = useCallback(async (threadId: string) => {
    try {
      const detail = await getThread(threadId)
      if (selectedThreadRef.current === threadId) setCurrentThread(detail)
      void refreshThreads().catch(() => undefined)
    } catch { /* run snapshot remains authoritative if thread refresh fails */ }
  }, [refreshThreads])

  const applyEvent = useCallback((expectedRunId: string, incoming: TraceEvent | ServerRunEvent | RunSnapshot) => {
    if (eventRunId(incoming, expectedRunId) !== expectedRunId) return
    if (isSnapshot(incoming)) {
      const snapshot = normalizeSnapshot(incoming)
      if (snapshot.runId !== expectedRunId) return
      showSnapshot(snapshot)
      if (snapshot.status !== 'running' && snapshot.threadId) void refreshSelectedThread(snapshot.threadId)
      return
    }

    const raw = incoming as ServerRunEvent
    const normalized = normalizeTraceEvent(raw, expectedRunId)
    const previousSeq = lastSeqRef.current.get(expectedRunId) ?? 0
    if (normalized.seq > 0 && normalized.seq <= previousSeq) return
    if (normalized.seq > 0) lastSeqRef.current.set(expectedRunId, normalized.seq)
    const previous = runsRef.current.get(expectedRunId)
    if (!previous) return
    const assistant = normalizeAssistantEvent(raw, expectedRunId, previous.threadId)
    const answer = assistant?.type === 'assistant_delta'
      ? `${previous.answer ?? previous.result?.answer ?? ''}${assistant.text}`
      : previous.answer ?? previous.result?.answer ?? null
    // A terminal event is evidence that the graph stopped, but the result
    // payload arrives in the authoritative snapshot fetched below.
    const status = raw.type === 'run.failed' ? 'error' : previous.status
    const traceEvents = reconcileEvents(previous.traceEvents, [raw])
    const next: RunSnapshot = {
      ...previous,
      status,
      answer,
      result: { ...(previous.result ?? {}), answer },
      traceEvents,
      seq: Math.max(previous.seq ?? 0, normalized.seq),
      currentNode: raw.node ?? previous.currentNode,
      error: raw.error ?? (status === 'error' ? raw.summary ?? previous.error : previous.error),
      updatedAt: normalized.timestamp,
    }
    showSnapshot(next)
    if (raw.type === 'run.completed' || raw.type === 'run.failed') {
      void getRun(expectedRunId).then((final) => {
        showSnapshot(final)
        if (final.threadId) void refreshSelectedThread(final.threadId)
      }).catch(() => {
        const selected = selectedRunRef.current === expectedRunId ||
          activeRunRef.current === expectedRunId ||
          (Boolean(previous.threadId) && selectedThreadRef.current === previous.threadId)
        if (!selected) return
        setConnectionState('lost')
        setRunError('Connection lost; run status unknown.')
      })
    }
  }, [refreshSelectedThread, showSnapshot])

  const connectToRun = useCallback((runId: string) => {
    closeStream()
    const afterSeq = lastSeqRef.current.get(runId) ?? 0
    const startPolling = () => {
      if (pollingRef.current) return
      setConnectionState('polling')
      pollingRef.current = setInterval(() => {
        void getRun(runId).then((snapshot) => {
          const known = runsRef.current.get(runId)
          const threadId = snapshot.threadId ?? known?.threadId
          const selected = selectedRunRef.current === runId ||
            activeRunRef.current === runId ||
            (Boolean(threadId) && selectedThreadRef.current === threadId)
          if (!selected) return
          applyEvent(runId, snapshot)
          setRunError(null)
          // ``showSnapshot`` closes and clears the interval for terminal
          // snapshots. Do not overwrite that state with ``polling`` after
          // the callback returns.
          setConnectionState(snapshot.status === 'running' ? 'polling' : 'closed')
        }).catch((error) => {
          if (selectedRunRef.current !== runId && activeRunRef.current !== runId) return
          setRunError(error instanceof Error ? error.message : 'Connection lost; run status unknown.')
          setConnectionState('lost')
        })
      }, 2000)
    }
    closeStreamRef.current = subscribeToRun(runId, afterSeq, (event) => applyEvent(runId, event), startPolling)
    if (typeof EventSource === 'undefined') startPolling()
    else setConnectionState('live')
  }, [applyEvent, closeStream])

  const loadThread = useCallback(async (threadId: string) => {
    selectedThreadRef.current = threadId
    selectedRunRef.current = null
    setRunLoading(true)
    setRunError(null)
    try {
      const detail = await getThread(threadId)
      if (selectedThreadRef.current !== threadId) return detail
      setCurrentThread(detail)
      const latestRunId = detail.runIds.at(-1)
      if (latestRunId) {
        const snapshot = await getRun(latestRunId)
        if (selectedThreadRef.current === threadId) {
          showSnapshot(snapshot)
          if (snapshot.status === 'running' && !closeStreamRef.current) connectToRun(snapshot.runId)
        }
      } else setCurrentRun(null)
      void refreshThreads().catch(() => undefined)
      return detail
    } catch (error) {
      setRunError(error instanceof Error ? error.message : 'Không thể tải thread.')
      throw error
    } finally { setRunLoading(false) }
  }, [connectToRun, refreshThreads, showSnapshot])

  const newThread = useCallback(async () => {
    const thread = await createThread()
    selectedThreadRef.current = thread.threadId
    selectedRunRef.current = null
    setCurrentThread(thread)
    setCurrentRun(null)
    setRunError(null)
    void refreshThreads().catch(() => undefined)
    return thread
  }, [refreshThreads])

  const sendMessage = useCallback(async (content: string, metadata: Partial<SendMessageInput> = {}) => {
    if (activeRunRef.current) throw new ApiError('Một workflow đang chạy. Vui lòng đợi run hiện tại hoàn tất.', 409)
    const threadId = selectedThreadRef.current ?? (await newThread()).threadId
    setRunLoading(true)
    setRunError(null)
    try {
      const snapshot = await sendThreadMessage(threadId, {
        content, paperInputs: metadata.paperInputs ?? metadata.paper_inputs ?? [], files: metadata.files ?? [],
      })
      selectedThreadRef.current = threadId
      selectedRunRef.current = null
      setCurrentRun(snapshot)
      runsRef.current.set(snapshot.runId, snapshot)
      lastSeqRef.current.set(snapshot.runId, snapshot.seq ?? 0)
      if (snapshot.status === 'running') {
        activeRunRef.current = snapshot.runId
        setActiveRunId(snapshot.runId)
        connectToRun(snapshot.runId)
      }
      void refreshSelectedThread(threadId)
      return snapshot
    } catch (error) {
      setRunError(error instanceof Error ? error.message : 'Không thể gửi yêu cầu.')
      throw error
    } finally { setRunLoading(false) }
  }, [connectToRun, newThread, refreshSelectedThread])

  const loadRun = useCallback(async (runId: string) => {
    // A failed EventSource keeps its close callback so the provider can stop
    // polling when it eventually recovers. Close it before a manual retry so
    // a running snapshot always gets a fresh live/polling subscription.
    closeStream()
    setRunLoading(true)
    setRunError(null)
    try {
      const snapshot = await getRun(runId)
      if (snapshot.threadId) {
        await loadThread(snapshot.threadId)
      } else {
        selectedThreadRef.current = null
        selectedRunRef.current = runId
        setCurrentThread(null)
        setCurrentRun(snapshot)
      }
      showSnapshot(snapshot)
      if (snapshot.status === 'running' && !closeStreamRef.current) connectToRun(snapshot.runId)
      return snapshot
    } catch (error) {
      setRunError(error instanceof Error ? error.message : 'Không thể tải run.')
      throw error
    } finally { setRunLoading(false) }
  }, [closeStream, connectToRun, loadThread, showSnapshot])

  const startRun = useCallback(async (input: RunInput) => {
    if (activeRunRef.current) throw new ApiError('Một workflow đang chạy. Vui lòng đợi run hiện tại hoàn tất.', 409)
    setRunLoading(true)
    setRunError(null)
    try {
      const snapshot = await createRun(input)
      selectedThreadRef.current = null
      selectedRunRef.current = snapshot.runId
      setCurrentThread(null)
      setCurrentRun(snapshot)
      showSnapshot(snapshot)
      if (snapshot.status === 'running') connectToRun(snapshot.runId)
      return snapshot
    } catch (error) {
      setRunError(error instanceof Error ? error.message : 'Could not start research.')
      throw error
    } finally {
      setRunLoading(false)
    }
  }, [connectToRun, showSnapshot])

  const clearRun = useCallback(() => {
    closeStream()
    selectedThreadRef.current = null
    selectedRunRef.current = null
    activeRunRef.current = null
    setActiveRunId(null)
    setCurrentThread(null)
    setCurrentRun(null)
    setRunError(null)
    setConnectionState('idle')
  }, [closeStream])

  useEffect(() => {
    void refreshConfig()
    void refreshThreads().catch(() => undefined)
    return closeStream
  }, [closeStream, refreshConfig, refreshThreads])

  const value = useMemo<ResearchContextValue>(() => ({
    config, configLoading, configError, threads, currentThread, currentRun,
    events: currentRun?.traceEvents.filter(isOperational) ?? [], runLoading, runError,
    connectionState, activeRun: Boolean(activeRunId), refreshConfig,
    refreshThreads, newThread, selectThread: (threadId) => {
      selectedThreadRef.current = threadId
      selectedRunRef.current = null
    }, loadThread, sendMessage, startRun, loadRun, clearRun,
  }), [activeRunId, clearRun, config, configError, configLoading, connectionState,
    currentRun, currentThread, loadRun, loadThread, newThread, refreshConfig,
    refreshThreads, runError, runLoading, sendMessage, startRun, threads])

  return <ResearchContext.Provider value={value}>{children}</ResearchContext.Provider>
}

export function useResearch(): ResearchContextValue {
  const value = useContext(ResearchContext)
  if (!value) throw new Error('useResearch phải được dùng bên trong ResearchProvider.')
  return value
}
