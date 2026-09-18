import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ResultsWorkspace } from './ResultsWorkspace'
import type { RunSnapshot, TraceEvent } from '../../types'

function snapshot(overrides: Partial<RunSnapshot['result']> = {}): RunSnapshot {
  return {
    runId: 'run-results-1',
    input: { query: 'attention methods', paperInputs: [], fileNames: [] },
    provider: 'test',
    model: 'test-model',
    status: 'success',
    progress: 100,
    traceEvents: [],
    result: {
      papers: [{
        paperId: '1706.03762',
        title: 'Attention Is All You Need',
        authors: ['Ashish Vaswani', 'Noam Shazeer'],
        published: '2017',
        arxivId: '1706.03762',
        pdfUrl: 'https://arxiv.org/pdf/1706.03762',
        bibtex: '@article{vaswani2017attention}',
        notes: { briefSummary: 'Attention replaces recurrence.\n- Parallel training.\n- Strong results.\n- Long sequences remain costly.', problem: 'Sequence transduction', method: 'Self attention', result: 'Strong results', limitation: 'Quadratic attention' },
      }],
      report: '# Attention report\n\nA finalized report.',
      ...overrides,
    },
  }
}

function event(overrides: Partial<TraceEvent> & { facts?: Record<string, unknown> } = {}): TraceEvent {
  return {
    runId: 'run-results-1',
    seq: 1,
    type: 'step.completed',
    node: 'final_report',
    label: 'Final report',
    kind: 'report',
    status: 'completed',
    summary: 'Prepared final research report',
    timestamp: '2026-09-18T10:00:00Z',
    progress: 100,
    ...overrides,
  } as TraceEvent
}

describe('ResultsWorkspace', () => {
  it('renders the finalized brief by default and keeps PMRL details behind View details', async () => {
    const user = userEvent.setup()
    render(<ResultsWorkspace snapshot={snapshot()} events={[event()]} onNewResearch={vi.fn()} />)

    expect(screen.getByRole('heading', { name: 'Attention Is All You Need' })).toBeInTheDocument()
    expect(screen.getByText('Attention replaces recurrence.')).toBeInTheDocument()
    expect(screen.queryByText('Sequence transduction')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'View details' }))
    expect(screen.getByText('Sequence transduction')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /Open arXiv page/ })).toHaveAttribute('href', 'https://arxiv.org/abs/1706.03762')
    expect(screen.getByText('Copy BibTeX')).toBeInTheDocument()
  })

  it('only exposes Comparison when benchmark evidence is explicitly available', async () => {
    const user = userEvent.setup()
    const papers = [
      { paperId: '1706.03762', title: 'Attention Is All You Need', notes: { problem: 'p1', method: 'm1', result: 'r1', limitation: 'l1' } },
      { paperId: 'paper-2', title: 'A second paper', notes: { problem: 'p2', method: 'm2', result: 'r2', limitation: 'l2' } },
    ]
    const view = render(<ResultsWorkspace snapshot={snapshot({ papers, benchmark: '| Paper | Result |\n| --- | --- |\n| A | good |' })} events={[event({ facts: { comparisonAvailable: false } })]} onNewResearch={vi.fn()} />)
    expect(screen.queryByRole('tab', { name: 'Comparison' })).not.toBeInTheDocument()

    view.rerender(<ResultsWorkspace snapshot={snapshot({ papers, benchmark: '| Paper | Result |\n| --- | --- |\n| A | good |' })} events={[event({ facts: { comparisonAvailable: true, comparedPaperIds: ['1706.03762', 'paper-2'] } })]} onNewResearch={vi.fn()} />)
    const comparison = screen.getByRole('tab', { name: 'Comparison' })
    await user.click(comparison)
    expect(screen.getAllByRole('columnheader', { name: 'Paper' })[0]).toBeInTheDocument()
    expect(screen.getByText('View benchmark Markdown')).toBeInTheDocument()
    expect(screen.getByText(/saved benchmark/)).toBeInTheDocument()
  })

  it('hides Comparison when fewer than two compared IDs match finalized papers', () => {
    const papers = [
      { paperId: 'paper-1', title: 'Paper one', notes: { problem: 'p1', method: 'm1', result: 'r1', limitation: 'l1' } },
      { paperId: 'paper-2', title: 'Paper two', notes: { problem: 'p2', method: 'm2', result: 'r2', limitation: 'l2' } },
    ]
    render(<ResultsWorkspace snapshot={snapshot({ papers, benchmark: 'A valid benchmark artifact' })} events={[event({ facts: { comparisonAvailable: true, comparedPaperIds: ['paper-1', 'missing-paper'] } })]} onNewResearch={vi.fn()} />)

    expect(screen.queryByRole('tab', { name: 'Comparison' })).not.toBeInTheDocument()
  })

  it('does not turn a local PDF paper key into an arXiv link', () => {
    const local = { paperId: 'local-notes-001', sourceType: 'local_pdf', title: 'Supplied notes', notes: { problem: 'A problem', method: 'A method', result: 'A result', limitation: 'A limitation' } }
    render(<ResultsWorkspace snapshot={snapshot({ papers: [local] })} events={[event()]} onNewResearch={vi.fn()} />)

    expect(screen.queryByRole('link', { name: /Open arXiv page/ })).not.toBeInTheDocument()
    expect(screen.getByText('Supplied PDF')).toBeInTheDocument()
  })

  it('suppresses fallback PMRL placeholders in the summary details', async () => {
    const user = userEvent.setup()
    render(<ResultsWorkspace snapshot={snapshot()} events={[event({ facts: { noteFallbackPaperIds: ['1706.03762'] } })]} onNewResearch={vi.fn()} />)

    expect(screen.getByText('PMRL fallback returned for this source.')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'View details' }))
    expect(screen.getAllByText('Not extracted')).toHaveLength(4)
    expect(screen.queryByText('Sequence transduction')).not.toBeInTheDocument()
  })

  it('does not rebuild a PMRL grid for a legacy Markdown comparison', async () => {
    const user = userEvent.setup()
    const papers = [
      { paperId: 'paper-1', title: 'Paper one', notes: { problem: 'fallback problem', method: 'fallback method', result: 'fallback result', limitation: 'fallback limitation' } },
      { paperId: 'paper-2', title: 'Paper two', notes: { problem: 'real problem', method: 'real method', result: 'real result', limitation: 'real limitation' } },
    ]
    render(<ResultsWorkspace snapshot={snapshot({ papers, benchmark: 'A valid benchmark artifact' })} events={[event({ facts: { comparisonAvailable: true, comparedPaperIds: ['paper-1', 'paper-2'], noteFallbackPaperIds: ['paper-1'] } })]} onNewResearch={vi.fn()} />)

    await user.click(screen.getByRole('tab', { name: 'Comparison' }))
    expect(screen.getByText(/saved benchmark/)).toBeInTheDocument()
    expect(screen.queryByText('fallback problem')).not.toBeInTheDocument()
    expect(screen.queryByText('real problem')).not.toBeInTheDocument()
  })

  it('renders structured comparison findings and keeps incomparable metrics explicit', async () => {
    const user = userEvent.setup()
    const papers = [
      { paperId: 'paper-1', title: 'Paper one', notes: { problem: 'p1', method: 'm1', result: 'r1', limitation: 'l1' } },
      { paperId: 'paper-2', title: 'Paper two', notes: { problem: 'p2', method: 'm2', result: 'r2', limitation: 'l2' } },
    ]
    const comparisonArtifact = {
      papers: papers.map((paper) => ({ paperId: paper.paperId, title: paper.title, findings: ['Short finding'], metrics: [] })),
      rows: [{ metric: 'Accuracy', dataset: 'Different datasets', unit: '%', values: {}, sourceQuotes: {}, comparable: false, reason: 'Not directly comparable' }],
      synthesis: ['Reported results use different tasks.'],
    }
    render(<ResultsWorkspace snapshot={snapshot({ papers, benchmark: 'Legacy benchmark', comparisonArtifact })} events={[event({ facts: { comparisonAvailable: true, comparedPaperIds: ['paper-1', 'paper-2'] } })]} onNewResearch={vi.fn()} />)
    await user.click(screen.getByRole('tab', { name: 'Comparison' }))
    expect(screen.getAllByText('Short finding')).toHaveLength(2)
    expect(screen.getByText('Not directly comparable')).toBeInTheDocument()
    expect(screen.queryByText('p1')).not.toBeInTheDocument()
  })

  it('warns when the verbatim report may include fallback PMRL notes', async () => {
    const user = userEvent.setup()
    render(<ResultsWorkspace snapshot={snapshot({ report: '# Report with fallback\n\nThe original Markdown stays intact.' })} events={[event({ facts: { noteFallbackPaperIds: ['1706.03762'] } })]} onNewResearch={vi.fn()} />)

    await user.click(screen.getByRole('tab', { name: 'Report' }))
    expect(screen.getByRole('note')).toHaveTextContent('may include fallback PMRL notes')
    expect(screen.getByText('The original Markdown stays intact.')).toBeInTheDocument()
  })

  it('renders Report only for actual markdown and keeps download absent without a report ID', () => {
    render(<ResultsWorkspace snapshot={snapshot({ report: '# Final report\n\nMarkdown' })} events={[event()]} onNewResearch={vi.fn()} />)
    expect(screen.getByRole('tab', { name: 'Report' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Download' })).not.toBeInTheDocument()
  })

  it('renders the report heading once while preserving the body Markdown', async () => {
    const user = userEvent.setup()
    render(<ResultsWorkspace snapshot={snapshot({ report: '# Final report\n\n## Findings\n\nMarkdown' })} events={[event()]} onNewResearch={vi.fn()} />)
    await user.click(screen.getByRole('tab', { name: 'Report' }))
    expect(screen.getAllByRole('heading', { name: 'Final report' })).toHaveLength(1)
    expect(screen.getByRole('heading', { name: 'Findings' })).toBeInTheDocument()
  })

  it('keeps Activity collapsed and omits assistant reasoning events', async () => {
    const user = userEvent.setup()
    render(<ResultsWorkspace snapshot={snapshot()} events={[event({ node: 'write_notes', type: 'step.completed', summary: 'Notes completed', durationMs: 1250 }), event({ seq: 2, node: 'write_notes', type: 'node.completed', summary: 'Notes completed again', durationMs: 1300 }), event({ seq: 3, type: 'assistant_delta', node: 'write_notes', kind: 'info', summary: 'private model reasoning' })]} onNewResearch={vi.fn()} />)
    const activity = screen.getByText('Activity')
    expect(screen.queryByText('private model reasoning')).not.toBeInTheDocument()
    await user.click(activity)
    expect(screen.queryByText('Notes completed')).not.toBeInTheDocument()
    expect(screen.getByText('Notes completed again')).toBeInTheDocument()
    expect(screen.getByText('Duration · 1.3s')).toBeInTheDocument()
    expect(screen.queryByText('private model reasoning')).not.toBeInTheDocument()
  })
})
