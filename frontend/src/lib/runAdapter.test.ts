import { describe, expect, it } from 'vitest'
import { deriveUiRun, orderedRunEvents } from './runAdapter'
import type { RunResult, RunSnapshot, TraceEvent, TraceFacts, TraceStatus } from '../types'

const RUN_ID = 'run-fixture-1'
const start = Date.parse('2026-09-18T10:00:00.000Z')

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
    timestamp: new Date(start + seq * 1000).toISOString(),
    progress: seq,
    facts,
  }
}

function snapshot(overrides: Partial<RunSnapshot> = {}, result: RunResult | null = null): RunSnapshot {
  return {
    runId: RUN_ID,
    input: { query: 'attention methods', paperInputs: [], fileNames: [] },
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

function searchEvents(refinements = 0): TraceEvent[] {
  const events: TraceEvent[] = [
    event(1, 'router', 'step.started'),
    event(2, 'router', 'step.updated', { intent: 'search' }, 'completed'),
    event(3, 'search_papers', 'step.started'),
    event(4, 'search_papers', 'step.updated', { resultCount: 5 }, 'completed'),
    event(5, 'eval_search', 'step.updated', { selectedCount: 2 }, 'completed'),
  ]
  for (let index = 1; index <= refinements; index += 1) {
    events.push(event(5 + index, 'refine_query', 'step.updated', { retryCount: index }, 'completed'))
  }
  return events
}

describe('runAdapter', () => {
  it('groups duplicate completion notices while keeping each search retry and its duration', () => {
    const events = [
      event(1, 'router', 'step.updated', { intent: 'search' }, 'completed'),
      event(2, 'search_papers', 'step.started'),
      event(3, 'search_papers', 'step.updated', { resultCount: 2 }, 'completed'),
      { ...event(4, 'search_papers', 'step.completed'), durationMs: 120 },
      event(5, 'refine_query', 'step.updated', { retryCount: 1 }, 'completed'),
      event(6, 'search_papers', 'step.started'),
      event(7, 'search_papers', 'step.updated', { resultCount: 1 }, 'completed'),
      { ...event(8, 'search_papers', 'step.completed'), durationMs: 220 },
    ]
    const view = deriveUiRun(snapshot({ traceEvents: events, seq: 8 }))
    const search = view.activity.filter((item) => item.text.startsWith('Search returned'))
    expect(search).toHaveLength(2)
    expect(search.map((item) => item.text)).toEqual(['Search returned 2 papers · 120ms', 'Search returned 1 paper · 220ms'])
    expect(view.activity.some((item) => item.text.includes('task finished'))).toBe(false)
    expect(view.activity.some((item) => item.text.includes('refined · 1 time'))).toBe(true)
  })

  it.each([0, 1, 2])('keeps %s search refinements inside one Discover row', (refinements) => {
    const view = deriveUiRun(snapshot({ traceEvents: searchEvents(refinements), seq: 5 + refinements }))

    expect(view.route).toBe('search')
    expect(view.stages.filter((stage) => stage.id === 'discover')).toHaveLength(1)
    const discover = view.stages.find((stage) => stage.id === 'discover')
    if (refinements === 0) expect(discover?.detail).toBeUndefined()
    else expect(discover?.detail).toBe(`Refined query · ${refinements} ${refinements === 1 ? 'time' : 'times'}`)
  })

  it('orders events by sequence and de-duplicates duplicate SSE frames', () => {
    const replacement = event(3, 'search_papers', 'step.completed', { resultCount: 4 }, 'completed')
    const duplicate = { ...replacement }
    const viewSnapshot = snapshot({ traceEvents: [event(1, 'router', 'step.started')], seq: 1 })

    const ordered = orderedRunEvents(viewSnapshot, [replacement, event(1, 'router', 'step.completed', undefined, 'completed'), duplicate])

    expect(ordered.map((item) => item.seq)).toEqual([1, 3])
    expect(ordered.find((item) => item.seq === 3)?.status).toBe('completed')
    expect(deriveUiRun(viewSnapshot, [replacement, duplicate]).events.map((item) => item.seq)).toEqual([1, 3])
  })

  it('does not let a stale duplicate sequence overwrite the authoritative snapshot event', () => {
    const authoritative = event(2, 'router', 'step.updated', { intent: 'direct_read' }, 'completed')
    const viewSnapshot = snapshot({ traceEvents: [event(1, 'router', 'step.started'), authoritative], seq: 2 })
    const staleSearch = event(2, 'search_papers', 'step.completed', { resultCount: 99 }, 'completed')

    const ordered = orderedRunEvents(viewSnapshot, [staleSearch])

    expect(ordered.map((item) => item.node)).toEqual(['router', 'router'])
    expect(deriveUiRun(viewSnapshot, [staleSearch]).route).toBe('direct_read')
  })

  it('ignores frames older than the snapshot cursor while retaining newer reconnect frames', () => {
    const authoritative = event(5, 'router', 'step.updated', { intent: 'search' }, 'completed')
    const viewSnapshot = snapshot({ traceEvents: [authoritative], seq: 5 })
    const outOfOrder = event(4, 'search_papers', 'step.updated', { resultCount: 99 }, 'completed')
    const newer = event(7, 'search_papers', 'step.updated', { resultCount: 5 }, 'completed')

    expect(orderedRunEvents(viewSnapshot, [newer, outOfOrder]).map((item) => item.seq)).toEqual([5, 7])
    expect(deriveUiRun(viewSnapshot, [newer, outOfOrder]).counts.candidates).toBe(5)
  })

  it('maps direct source input without a fabricated discovery stage or search count', () => {
    const directEvents = [
      event(1, 'router', 'step.updated', { intent: 'direct_read' }, 'completed'),
      event(2, 'read_paper', 'step.started'),
      event(3, 'read_paper', 'step.updated', { paperCount: 1 }, 'completed'),
    ]
    const view = deriveUiRun(snapshot({ traceEvents: directEvents, seq: 3 }))

    expect(view.route).toBe('direct_read')
    expect(view.stages.map((stage) => stage.id)).toEqual(['understand', 'read', 'enrich', 'notes', 'report'])
    expect(view.stages.find((stage) => stage.id === 'read')?.label).toBe('Read supplied source(s)')
    expect(view.counts).toEqual({ pdfsRead: 1 })
    expect(view.activity.some((item) => /Discovery skipped · supplied source/.test(item.text))).toBe(true)
    expect(view.activity.some((item) => /Search returned/.test(item.text))).toBe(false)
  })

  it('keeps requested direct comparison separate from discovery', () => {
    const events = [
      event(1, 'router', 'step.updated', { intent: 'direct_compare' }, 'completed'),
      event(2, 'read_paper', 'step.updated', { paperCount: 2 }, 'completed'),
      event(3, 'compare_benchmark', 'step.updated', { comparisonAvailable: true, comparedPaperIds: ['p1', 'p2'] }, 'completed'),
    ]
    const view = deriveUiRun(snapshot({ traceEvents: events, seq: 3 }, {
      benchmark: '| Paper | Method |\n| --- | --- |\n| A | Attention |',
      report: '# Report',
    }))

    expect(view.route).toBe('direct_compare')
    expect(view.stages.map((stage) => stage.id)).toEqual(['understand', 'read', 'enrich', 'notes', 'compare', 'report'])
    expect(view.stages.some((stage) => stage.id === 'discover')).toBe(false)
    expect(view.comparisonReady).toBe(true)
  })

  it('gates the completed results phase on a usable final report', () => {
    const terminal = [
      event(1, 'router', 'step.updated', { intent: 'direct_read' }, 'completed'),
      event(2, 'read_paper', 'step.completed', { paperCount: 1 }, 'completed'),
      event(3, 'write_notes', 'step.completed', undefined, 'completed'),
      event(4, 'final_report', 'step.completed', { reportAvailable: true, reportId: 'report-1' }, 'completed'),
      event(5, 'run.completed', 'run.completed', undefined, 'completed'),
    ]
    const missing = deriveUiRun(snapshot({ status: 'success', traceEvents: terminal, seq: 5 }, { report: '' }))
    const ready = deriveUiRun(snapshot({ status: 'success', traceEvents: terminal, seq: 5 }, { report: '# Final report\n\nFindings.' }))
    const errorReport = deriveUiRun(snapshot({ status: 'success', traceEvents: terminal, seq: 5 }, { report: '# Error Generating Final Report\n\nProvider unavailable.' }))

    expect(missing.phase).toBe('missing_report')
    expect(missing.reportReady).toBe(false)
    expect(ready.phase).toBe('completed')
    expect(ready.reportReady).toBe(true)
    expect(errorReport.phase).toBe('missing_report')
  })

  it('exposes comparison only when both the workflow fact and benchmark artifact are real', () => {
    const events = [
      event(1, 'router', 'step.updated', { intent: 'search' }, 'completed'),
      event(2, 'eval_search', 'step.updated', { selectedCount: 2 }, 'completed'),
      event(3, 'read_paper', 'step.updated', { paperCount: 2 }, 'completed'),
      event(4, 'compare_benchmark', 'step.updated', { comparisonAvailable: true, comparedPaperIds: ['p1', 'p2'] }, 'completed'),
    ]
    const papers = [{ paperId: 'p1', title: 'Paper One' }, { paperId: 'p2', title: 'Paper Two' }]
    const available = deriveUiRun(snapshot({ status: 'success', traceEvents: events, seq: 4 }, {
      benchmark: '| Paper | Result |\n| --- | --- |\n| A | Good |',
      report: '# Report',
      papers,
    }))
    const missingArtifact = deriveUiRun(snapshot({ status: 'success', traceEvents: events, seq: 4 }, { report: '# Report', papers }))
    const aliasedArtifact = deriveUiRun(snapshot({ status: 'success', traceEvents: events, seq: 4 }, {
      benchmarkMatrix: '| Paper | Result |\n| --- | --- |\n| A | Good |',
      report: '# Report',
      papers,
    }))
    const unavailableFact = deriveUiRun(snapshot({ status: 'success', traceEvents: events.map((item) => item.node === 'compare_benchmark' ? { ...item, facts: { comparisonAvailable: false } } : item), seq: 4 }, {
      benchmark: '| Paper | Result |\n| --- | --- |\n| A | Good |',
      report: '# Report',
      papers,
    }))

    expect(available.comparisonReady).toBe(true)
    expect(missingArtifact.comparisonReady).toBe(false)
    expect(aliasedArtifact.comparisonReady).toBe(true)
    expect(unavailableFact.comparisonReady).toBe(false)
  })

  it('keeps a partial PDF read visible as a warning while preserving measured counts', () => {
    const view = deriveUiRun(snapshot({
      status: 'degraded',
      traceEvents: [
        event(1, 'router', 'step.updated', { intent: 'search' }, 'completed'),
        event(2, 'read_paper', 'step.updated', { paperCount: 2, failedPdfCount: 1 }, 'completed'),
      ],
      seq: 2,
    }))

    expect(view.counts.pdfsRead).toBe(2)
    expect(view.warnings).toContain('1 PDF could not be read.')
    expect(view.phase).toBe('missing_report')
  })

  it('surfaces structured note fallbacks as unavailable fields instead of fabricated notes', () => {
    const view = deriveUiRun(snapshot({
      traceEvents: [
        event(1, 'router', 'step.updated', { intent: 'direct_read' }, 'completed'),
        event(2, 'write_notes', 'step.updated', { noteFallbackPaperIds: ['paper-1'] }, 'completed'),
      ],
      seq: 2,
    }))

    expect(view.warnings).toContain('Structured notes were unavailable for 1 paper(s).')
    expect(view.stages.find((item) => item.id === 'notes')?.detail).toBe('1 unavailable')
    expect(view.activity.some((item) => /Structured notes unavailable for 1 paper/.test(item.text))).toBe(true)
  })

  it.each([
    ['read_paper', 'read', 'Could not read source PDFs.'],
    ['write_notes', 'notes', 'Could not write structured notes.'],
    ['final_report', 'report', 'Could not prepare the research report.'],
  ] as const)('attaches a terminal failure to the %s stage', (node, stage, headline) => {
    const view = deriveUiRun(snapshot({
      status: 'error',
      currentNode: node,
      error: `Fixture failure at ${node}.`,
      traceEvents: [
        event(1, 'router', 'step.updated', { intent: 'direct_read' }, 'completed'),
        event(2, node, 'run.failed', undefined, 'error'),
      ],
      seq: 2,
    }))

    expect(view.phase).toBe('failed')
    expect(view.stages.find((item) => item.id === stage)?.status).toBe('failed')
    expect(view.headline).toBe(headline)
    expect(view.error).toBe(`Fixture failure at ${node}.`)
  })

  it('maps a search with no suitable result to a discover error and never invents a report', () => {
    const view = deriveUiRun(snapshot({
      status: 'error',
      currentNode: 'error_handler',
      error: 'No papers met the selection criteria.',
      traceEvents: [
        event(1, 'router', 'step.updated', { intent: 'search' }, 'completed'),
        event(2, 'search_papers', 'step.updated', { resultCount: 5 }, 'completed'),
        event(3, 'eval_search', 'step.updated', { selectedCount: 0 }, 'completed'),
        event(4, 'error_handler', 'run.failed', undefined, 'error'),
      ],
      seq: 4,
    }))

    expect(view.phase).toBe('failed')
    expect(view.stages.find((item) => item.id === 'discover')?.status).toBe('failed')
    expect(view.headline).toMatch(/find suitable papers/i)
    expect(view.reportReady).toBe(false)
  })

  it('uses the failed source stage when error_handler reports a provider failure', () => {
    const view = deriveUiRun(snapshot({
      status: 'error',
      currentNode: 'write_notes',
      error: 'The language-model provider did not respond.',
      traceEvents: [
        event(1, 'router', 'step.updated', { intent: 'direct_read' }, 'completed'),
        event(2, 'write_notes', 'step.started'),
        event(3, 'error_handler', 'run.failed', undefined, 'error'),
      ],
      seq: 3,
    }))

    expect(view.stages.find((item) => item.id === 'notes')?.status).toBe('failed')
    expect(view.error).toContain('provider did not respond')
  })
})
