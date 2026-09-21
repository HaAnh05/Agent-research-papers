import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate, useParams } from 'react-router-dom'
import { ChevronDown } from 'lucide-react'
import { ApiError, getThread } from '../lib/api'
import { deriveUiRun, type UiRun } from '../lib/runAdapter'
import { useResearch } from '../context/ResearchContext'
import { ResultsWorkspace } from '../components/results/ResultsWorkspace'
import type { RunInput, RunSnapshot } from '../types'

function clockTime(timestamp: string): string {
  const time = new Date(timestamp)
  return Number.isNaN(time.getTime()) ? '' : time.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function elapsedSince(snapshot: RunSnapshot, now: number): string | null {
  const started = Date.parse(snapshot.startedAt ?? snapshot.createdAt ?? '')
  if (!Number.isFinite(started)) return null
  return `${Math.max(0, (now - started) / 1000).toFixed(1)}s`
}

function Activity({ view }: { view: UiRun }) {
  return <details className="activity">
    <summary><span>Activity · {view.activity.length} events</span><ChevronDown size={15} aria-hidden="true" /></summary>
    <div className="activity__scroll"><ol>
      {view.activity.map((item) => <li key={item.key}>
        <time dateTime={item.timestamp}>{clockTime(item.timestamp)}</time><span>{item.text}</span>
      </li>)}
    </ol></div>
  </details>
}

function Pipeline({ snapshot, view, connectionLost, onEdit, onRestart, onReconnect, canRestart }: {
  snapshot: RunSnapshot
  view: UiRun
  connectionLost: boolean
  onEdit: () => void
  onRestart: () => void
  onReconnect: () => void
  canRestart: boolean
}) {
  const [now, setNow] = useState(Date.now())
  const headingRef = useRef<HTMLHeadingElement>(null)
  useEffect(() => {
    headingRef.current?.focus()
  }, [snapshot.runId])
  useEffect(() => {
    if (view.phase !== 'running') return
    const timer = window.setInterval(() => setNow(Date.now()), 100)
    return () => window.clearInterval(timer)
  }, [view.phase])
  return <main id="main-content" className="pipeline-page">
    <section className={`pipeline-panel ${view.phase !== 'running' ? 'pipeline-panel--terminal' : ''}`} aria-labelledby="pipeline-heading">
      <header className="pipeline-panel__header">
        <h1 id="pipeline-heading" ref={headingRef} tabIndex={-1}>{view.phase === 'running' ? 'Running research' : 'Research stopped'}</h1>
        {view.phase === 'running' && !connectionLost && !view.finalizing ? <span className="pipeline-panel__elapsed">{elapsedSince(snapshot, now)}</span> : null}
      </header>
      <p className="pipeline-panel__headline" aria-live="polite" aria-atomic="true">{view.headline}</p>
      {view.phase === 'running' && view.counts.selected !== undefined ? <p className="pipeline-panel__verified">{view.counts.selected} {view.counts.selected === 1 ? 'paper selected' : 'papers selected'}</p> : null}
      {connectionLost ? <div className="pipeline-panel__connection" role="status"><p>Connection lost; run status unknown.</p><button type="button" className="action-button" onClick={onReconnect}>Reconnect</button></div> : null}
      <ol className="stage-list">
        {view.stages.map((stage) => <li className={`stage-row stage-row--${stage.status}`} key={stage.id}>
          <span className="stage-row__dot" aria-hidden="true" />
          <span className="stage-row__main"><span>{stage.label}</span>{stage.detail ? <small>{stage.detail}</small> : null}</span>
          <span className="stage-row__status">{stage.status === 'completed' ? 'Done' : stage.status === 'running' ? 'Running' : stage.status === 'failed' ? 'Failed' : stage.status === 'skipped' ? 'Skipped' : ''}</span>
        </li>)}
      </ol>
      {Object.values(view.counts).some((value) => value !== undefined) ? <div className="activity-chips" aria-label="Verified research counts">
        {view.counts.candidates !== undefined ? <span>searched · {view.counts.candidates} papers</span> : null}
        {view.counts.selected !== undefined ? <span>selected · {view.counts.selected} papers</span> : null}
        {view.counts.pdfsRead !== undefined ? <span>read · {view.counts.pdfsRead} PDFs</span> : null}
        {view.counts.repositories !== undefined ? <span>GitHub · {view.counts.repositories} related repos</span> : null}
        {view.counts.bibtexEntries !== undefined ? <span>BibTeX · {view.counts.bibtexEntries} entries</span> : null}
      </div> : null}
      {view.warnings.map((warning) => <p className="pipeline-panel__warning" key={warning}>{warning}</p>)}
      {view.error ? <p className="pipeline-panel__error" role="alert">{view.error}</p> : null}
      {view.phase !== 'running' ? <div className="pipeline-panel__actions">
        <button type="button" className="action-button action-button--primary" onClick={onEdit}>Edit request</button>
        {canRestart ? <button type="button" className="action-button" onClick={onRestart}>Start again</button> : null}
      </div> : null}
      <Activity view={view} />
    </section>
  </main>
}

export function RunPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const location = useLocation()
  const research = useResearch()
  const [loadError, setLoadError] = useState<string | null>(null)
  const currentRun = research.currentRun?.runId === id ? research.currentRun : null
  const originalInput = (location.state as { originalInput?: RunInput } | null)?.originalInput
  const view = useMemo(() => currentRun ? deriveUiRun(currentRun, research.events) : null, [currentRun, research.events])

  const load = useCallback(async (runId: string) => {
    try {
      await research.loadRun(runId)
      setLoadError(null)
    } catch (error) {
      // A 404 can be an old thread link. Network/5xx failures must remain a
      // connection error so the user gets a retry path instead of a false
      // claim that an in-memory run disappeared.
      if (!(error instanceof ApiError) || error.status !== 404) {
        setLoadError('Connection lost; run status unknown. Retry loading this run.')
        return
      }
      try {
        const thread = await getThread(runId)
        const latestRunId = thread.runIds.at(-1)
        if (latestRunId) navigate(`/run/${encodeURIComponent(latestRunId)}`, { replace: true })
        else setLoadError('This thread contains no research run.')
      } catch (threadError) {
        setLoadError(threadError instanceof ApiError && threadError.status === 404
          ? 'This run is no longer available. Runs are held in server memory and cannot resume after a restart.'
          : 'Connection lost; run status unknown. Retry loading this run.')
      }
    }
  }, [navigate, research.loadRun])

  useEffect(() => {
    if (!id || currentRun) return
    void load(id)
  }, [currentRun, id, load])

  useEffect(() => {
    if (view?.phase === 'completed') {
      window.requestAnimationFrame(() => document.querySelector<HTMLElement>('[data-results-heading]')?.focus())
    }
  }, [view?.phase])

  const snapshotInput = currentRun ? {
    query: currentRun.input.query,
    paperInputs: currentRun.input.paperInputs,
    files: originalInput?.files ?? [],
  } : null
  const hasLostFiles = Boolean(currentRun?.input.fileNames?.length || currentRun?.input.pdfNames?.length) && !originalInput?.files?.length
  const canRestart = Boolean(snapshotInput && !hasLostFiles && (snapshotInput.query || snapshotInput.paperInputs.length || snapshotInput.files.length))
  const edit = () => navigate('/', { state: { draft: snapshotInput ?? undefined } })
  const restart = async () => {
    if (!snapshotInput || !canRestart) return
    try {
      const fresh = await research.startRun(snapshotInput)
      navigate(`/run/${encodeURIComponent(fresh.runId)}`, { replace: true, state: { originalInput: snapshotInput } })
    } catch { /* Landing retains the request and displays the API error. */
      edit()
    }
  }

  if (!id || loadError) return <main id="main-content" className="pipeline-page"><section className="pipeline-panel"><h1 tabIndex={-1}>{loadError?.startsWith('Connection lost') ? 'Connection lost' : 'Run unavailable'}</h1><p role="alert">{loadError ?? 'No run was selected.'}</p><div className="pipeline-panel__actions">{id ? <button type="button" className="action-button action-button--primary" onClick={() => { setLoadError(null); void load(id) }}>Retry</button> : null}<button type="button" className="action-button" onClick={() => navigate('/')}>New research</button></div></section></main>
  if (!currentRun || !view) return <main id="main-content" className="pipeline-page"><section className="pipeline-panel"><h1>Opening research run</h1><p>Loading the current run state…</p></section></main>
  if (view.phase === 'completed') return <ResultsWorkspace snapshot={currentRun} events={view.events} onNewResearch={() => navigate('/')} />
  return <Pipeline snapshot={currentRun} view={view} connectionLost={research.connectionState === 'lost'} onEdit={edit} onRestart={() => void restart()} onReconnect={() => { if (id) void research.loadRun(id) }} canRestart={canRestart} />
}
