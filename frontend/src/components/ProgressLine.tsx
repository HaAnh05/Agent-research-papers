import { useState } from 'react'
import { clock, durationText, factsLine, llmLine } from '../lib/format'
import { displaySteps, runSeconds } from '../lib/run'
import { useNow } from '../lib/useNow'
import type { RunView, StepView } from '../types'
import { Chev, Tick, Warn } from './icons'

/** Completed: the server duration. Running: elapsed time since the step started, ticking once per second. */
function stepTime(step: StepView, now: number): string {
  if (step.status === 'completed') return clock(step.durationMs)
  if (step.status === 'running' && step.startedMs != null) return clock(Math.max(0, now - step.startedMs))
  return '—'
}

function Step({ step, parallel, now }: { step: StepView; parallel: boolean; now: number }) {
  const cls = step.status === 'completed' || step.status === 'skipped' ? 'done' : step.status === 'running' ? 'now' : step.status === 'error' ? 'err' : 'wait'
  const facts = step.status === 'wait' ? '' : factsLine(step)
  const llm = step.status === 'wait' ? '' : llmLine(step)
  return (
    <div className={`step ${cls}`}>
      <span className="s-ico" />
      <div>
        <b>{step.label}</b>
        {parallel && <span className="par">song song</span>}
        {step.description && <div className="d">{step.description}</div>}
        {facts && <div className="facts">{facts}</div>}
        {llm && <div className="facts llm">{llm}</div>}
      </div>
      <span className="t tnum">{stepTime(step, now)}</span>
    </div>
  )
}

function headerLabel(run: RunView, rows: StepView[], paperCount: number, now: number): string {
  if (run.status === 'error') return 'Phân tích không hoàn tất'
  if (run.status === 'running' || !run.result) {
    const active = rows.filter((s) => s.status === 'running')
    const names = (active.length ? active : rows.filter((s) => s.status !== 'wait').slice(-1)).map((s) => s.label)
    const elapsed = run.status === 'running' && run.startedMs != null ? ` đang chạy ${clock(Math.max(0, now - run.startedMs))}` : ''
    return names.length ? `${names.join(', ')}…${elapsed}` : 'Đang bắt đầu…'
  }
  const dur = durationText(runSeconds(run))
  const what = paperCount > 1 ? `${paperCount} bài` : 'bài'
  return dur ? `Đã phân tích ${what} trong ${dur}` : `Đã phân tích ${what}`
}

/** Expandable progress line, like "Searched the web": one row per step with description, facts and duration. */
export function ProgressLine({ run, open: initialOpen = false }: { run: RunView; open?: boolean }) {
  const [open, setOpen] = useState(initialOpen)
  // One timer per progress line; it stops when the run ends.
  const now = useNow(run.status === 'running')
  const rows = displaySteps(run)
  const parallelGroups = rows.reduce<Record<string, number>>((acc, s) => {
    if (s.parallelGroup) acc[s.parallelGroup] = (acc[s.parallelGroup] ?? 0) + 1
    return acc
  }, {})
  const paperCount = run.summary?.papers?.length ?? 0
  const finished = run.status !== 'running' && !!run.result
  return (
    <div className={`prog${open ? ' open' : ''}`} data-testid="progress">
      <button type="button" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
        {run.status === 'error' ? <Warn /> : finished ? <Tick /> : <span className="pulse" />}
        <span className="pl">{headerLabel(run, rows, paperCount, now)}</span>
        <Chev />
      </button>
      {open && (
        <div className="steps">
          {rows.length === 0 && <div className="step wait"><span className="s-ico" /><div><b>Đang bắt đầu…</b></div><span className="t">—</span></div>}
          {rows.map((s) => (
            <Step key={s.key} step={s} parallel={!!s.parallelGroup && (parallelGroups[s.parallelGroup] ?? 0) > 1} now={now} />
          ))}
        </div>
      )}
    </div>
  )
}
