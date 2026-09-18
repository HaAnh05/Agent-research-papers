import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from './App'
import { ResearchProvider } from './context/ResearchContext'

const now = '2026-09-18T10:00:00.000Z'
const completedRun = {
  runId: 'run-results',
  input: { query: 'Explain attention', paperInputs: ['1706.03762'], fileNames: [] },
  provider: 'fixture',
  model: 'fixture-model',
  status: 'success',
  currentNode: 'final_report',
  progress: 100,
  seq: 4,
  traceEvents: [
    { runId: 'run-results', seq: 1, type: 'step.updated', node: 'router', label: 'Router', kind: 'routing', status: 'completed', summary: 'Routed direct input.', timestamp: now, progress: 20, facts: { intent: 'direct_read' } },
    { runId: 'run-results', seq: 2, type: 'step.updated', node: 'read_paper', label: 'Read paper', kind: 'paper', status: 'completed', summary: 'Read one paper.', timestamp: now, progress: 50, facts: { paperCount: 1 } },
    { runId: 'run-results', seq: 3, type: 'step.updated', node: 'final_report', label: 'Final report', kind: 'report', status: 'completed', summary: 'Report ready.', timestamp: now, progress: 90, facts: { reportAvailable: true, reportId: 'report-results' } },
    { runId: 'run-results', seq: 4, type: 'run.completed', node: 'final_report', label: 'Final report', kind: 'report', status: 'completed', summary: 'Completed.', timestamp: now, progress: 100 },
  ],
  result: {
    reportId: 'report-results',
    report: '# Attention report\n\nFinal findings.',
    papers: [{ paperId: '1706.03762', arxivId: '1706.03762', title: 'Attention Is All You Need', notes: { problem: 'Sequence transduction', method: 'Self attention', result: 'Strong results', limitation: 'Quadratic attention' } }],
  },
}

function json(value: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(value), { status, headers: { 'content-type': 'application/json' } }))
}

function renderApp(initialEntry: string) {
  vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
    const url = String(input)
    if (url.endsWith('/config')) return json({ provider: 'fixture', model: 'fixture-model', configured: true })
    if (url.endsWith('/threads')) return json([])
    if (url.endsWith('/runs/run-results')) return json(completedRun)
    return json([])
  }))
  return render(<MemoryRouter initialEntries={[initialEntry]}><ResearchProvider><App /></ResearchProvider></MemoryRouter>)
}

afterEach(() => vi.unstubAllGlobals())

describe('Research Scout application routing', () => {
  it('opens on the minimal Landing state with one obvious request action', async () => {
    renderApp('/')

    expect(await screen.findByRole('heading', { name: 'What do you want to understand?' })).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'Research request' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Start research' })).toBeInTheDocument()
    expect(screen.queryByText(/Library|Thư viện báo cáo/i)).not.toBeInTheDocument()
  })

  it('redirects unknown URLs to Landing instead of exposing a retired shell', async () => {
    renderApp('/retired-chat-route')

    expect(await screen.findByRole('heading', { name: 'What do you want to understand?' })).toBeInTheDocument()
  })

  it('opens a completed run directly in the Results workspace', async () => {
    renderApp('/run/run-results')

    expect(await screen.findByRole('heading', { name: 'Attention Is All You Need' })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Summary' })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Report' })).toBeInTheDocument()
    expect(screen.queryByText('Running research')).not.toBeInTheDocument()
  })

  it('does not leave a blank route while the run snapshot is loading', async () => {
    renderApp('/run/run-results')

    expect(screen.getByRole('heading', { name: /Opening research run|Loading the current run state/i })).toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Attention Is All You Need' })).toBeInTheDocument())
  })
})
