import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { RunPage } from './RunPage'
import type { RunResult, RunSnapshot, TraceEvent, TraceFacts, TraceStatus } from '../types'

const RUN_ID = 'run-page-fixture'
const now = '2026-09-18T10:00:00.000Z'
const mocked = vi.hoisted(() => ({
  research: {
    currentRun: null as RunSnapshot | null,
    events: [] as TraceEvent[],
    runLoading: false,
    runError: null as string | null,
    connectionState: 'closed' as string,
    activeRun: false,
    loadRun: vi.fn(),
    startRun: vi.fn(),
  },
}))

vi.mock('../context/ResearchContext', () => ({
  useResearch: () => mocked.research,
}))

vi.mock('../lib/api', () => ({
  getThread: vi.fn(),
}))

function event(
  seq: number,
  node: string,
  type: TraceEvent['type'] = 'step.updated',
  facts?: TraceFacts,
  status: TraceStatus = type === 'step.completed' ? 'completed' : 'running',
): TraceEvent {
  return {
    runId: RUN_ID,
    seq,
    type,
    node,
    label: node,
    kind: node === 'final_report' ? 'report' : 'node',
    status,
    summary: `${node} ${type}`,
    timestamp: new Date(Date.parse(now) + seq * 1000).toISOString(),
    progress: seq,
    facts,
  }
}

function snapshot(overrides: Partial<RunSnapshot> = {}, result: RunResult | null = null): RunSnapshot {
  return {
    runId: RUN_ID,
    input: { query: 'Explain attention', paperInputs: ['1706.03762'], fileNames: [] },
    provider: 'fixture',
    model: 'fixture-model',
    status: 'running',
    progress: 0,
    traceEvents: [],
    seq: 0,
    result,
    ...overrides,
  }
}

function renderRun(run: RunSnapshot, events = run.traceEvents) {
  mocked.research.currentRun = run
  mocked.research.events = events
  return render(
    <MemoryRouter initialEntries={[`/run/${RUN_ID}`]}>
      <Routes>
        <Route path="/run/:id" element={<RunPage />} />
        <Route path="/" element={<LocationProbe />} />
      </Routes>
    </MemoryRouter>,
  )
}

function LocationProbe() {
  const location = useLocation()
  return <output data-testid="location">{location.pathname}</output>
}

beforeEach(() => {
  mocked.research.currentRun = null
  mocked.research.events = []
  mocked.research.runLoading = false
  mocked.research.runError = null
  mocked.research.connectionState = 'closed'
  mocked.research.activeRun = false
  mocked.research.loadRun.mockReset().mockResolvedValue(snapshot())
  mocked.research.startRun.mockReset().mockResolvedValue(snapshot())
})

describe('RunPage semantic pipeline transitions', () => {
  it('shows only Understand before routing is known', () => {
    renderRun(snapshot())

    expect(screen.getByRole('heading', { name: 'Running research' })).toBeInTheDocument()
    expect(screen.getByText('Understand request')).toBeInTheDocument()
    expect(screen.queryByText('Discover relevant papers')).not.toBeInTheDocument()
    expect(screen.queryByText('Read source PDFs')).not.toBeInTheDocument()
  })

  it('renders one Discover row while search refinement events accumulate', () => {
    const events = [
      event(1, 'router', 'step.updated', { intent: 'search' }, 'completed'),
      event(2, 'search_papers', 'step.updated', { resultCount: 5 }, 'completed'),
      event(3, 'eval_search', 'step.updated', { selectedCount: 2 }, 'completed'),
      event(4, 'refine_query', 'step.updated', { retryCount: 1 }, 'completed'),
      event(5, 'refine_query', 'step.updated', { retryCount: 2 }, 'completed'),
    ]
    renderRun(snapshot({ traceEvents: events, seq: 5 }), events)

    expect(screen.getAllByText('Discover relevant papers')).toHaveLength(1)
    expect(screen.getByText('Refined query · 2 times')).toBeInTheDocument()
    expect(screen.getByText('searched · 5 papers')).toBeInTheDocument()
    expect(screen.getByText('selected · 2 papers')).toBeInTheDocument()
  })

  it('keeps a direct answer in Pipeline with an edit path instead of opening Results', async () => {
    const user = userEvent.setup()
    const events = [
      event(1, 'router', 'step.updated', { intent: 'direct_answer' }, 'completed'),
      event(2, 'direct_answer', 'step.completed', undefined, 'completed'),
      event(3, 'run.completed', 'run.completed', undefined, 'completed'),
    ]
    renderRun(snapshot({ status: 'success', traceEvents: events, seq: 3 }, { answer: 'Direct answer', papers: [] }), events)

    expect(screen.getByRole('heading', { name: 'Research request needed' })).toBeInTheDocument()
    expect(screen.getByText('No paper research was run for this request.')).toBeInTheDocument()
    expect(screen.queryByRole('tab', { name: 'Summary' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Edit request' }))
    expect(screen.getByTestId('location')).toHaveTextContent('/')
  })

  it.each([
    ['read_paper', 'Could not read source PDFs.', 'Fixture PDF parser failed.'],
    ['write_notes', 'Could not write structured notes.', 'The language-model provider did not respond.'],
    ['final_report', 'Could not prepare the research report.', 'The report artifact was unavailable.'],
  ] as const)('renders a truthful %s failure with recovery controls', async (node, headline, error) => {
    const user = userEvent.setup()
    const events = [
      event(1, 'router', 'step.updated', { intent: 'direct_read' }, 'completed'),
      event(2, node, 'run.failed', undefined, 'error'),
    ]
    renderRun(snapshot({ status: 'error', currentNode: node, error, traceEvents: events, seq: 2 }), events)

    expect(screen.getByRole('heading', { name: 'Research stopped' })).toBeInTheDocument()
    expect(screen.getByText(headline)).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent(error)
    expect(screen.getByRole('button', { name: 'Edit request' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Start again' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Start again' }))
    await waitFor(() => expect(mocked.research.startRun).toHaveBeenCalledTimes(1))
  })

  it('shows connection loss separately from a failed run', () => {
    mocked.research.connectionState = 'lost'
    const events = [event(1, 'router', 'step.updated', { intent: 'direct_read' }, 'completed')]
    renderRun(snapshot({ traceEvents: events, seq: 1 }), events)

    expect(screen.getByRole('status')).toHaveTextContent(/Connection lost; run status unknown/i)
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('requires reattachment before retrying a run whose PDF is no longer in browser memory', () => {
    const events = [
      event(1, 'router', 'step.updated', { intent: 'direct_read' }, 'completed'),
      event(2, 'read_paper', 'run.failed', undefined, 'error'),
    ]
    renderRun(snapshot({
      status: 'error',
      currentNode: 'read_paper',
      error: 'The supplied PDF could not be read.',
      input: { query: '', paperInputs: [], fileNames: ['paper.pdf'] },
      traceEvents: events,
      seq: 2,
    }), events)

    expect(screen.getByRole('button', { name: 'Edit request' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Start again' })).not.toBeInTheDocument()
  })

  it('does not enter Results when a terminal run has no usable report', () => {
    const events = [
      event(1, 'router', 'step.updated', { intent: 'direct_read' }, 'completed'),
      event(2, 'final_report', 'step.completed', undefined, 'completed'),
      event(3, 'run.completed', 'run.completed', undefined, 'completed'),
    ]
    renderRun(snapshot({ status: 'success', traceEvents: events, seq: 3 }, { papers: [] }), events)

    expect(screen.getByRole('heading', { name: 'Research stopped' })).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent(/workflow finished without a usable report/i)
    expect(screen.queryByRole('tab', { name: 'Summary' })).not.toBeInTheDocument()
  })

  it('opens Results only after a finalized report is present', () => {
    const events = [
      event(1, 'router', 'step.updated', { intent: 'direct_read' }, 'completed'),
      event(2, 'final_report', 'step.updated', { reportAvailable: true, reportId: 'report-1' }, 'completed'),
      event(3, 'run.completed', 'run.completed', undefined, 'completed'),
    ]
    renderRun(snapshot({ status: 'success', traceEvents: events, seq: 3 }, {
      report: '# Final report\n\nFindings.',
      reportId: 'report-1',
      papers: [{ paperId: '1706.03762', title: 'Attention Is All You Need', notes: { problem: 'Problem', method: 'Method', result: 'Result', limitation: 'Limitation' } }],
    }), events)

    expect(screen.getByRole('heading', { name: 'Attention Is All You Need' })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Summary' })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Report' })).toBeInTheDocument()
  })
})
