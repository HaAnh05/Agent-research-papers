import { describe, expect, it } from 'vitest'
import { normalizeSnapshot, normalizeTraceEvent, reconcileEvents } from './api'
import type { TraceEvent } from '../types'

const event = (seq: number, status: TraceEvent['status'], summary = `event ${seq}`): TraceEvent => ({
  runId: 'run-1', seq, type: status === 'completed' ? 'step.completed' : 'step.updated', node: 'router', label: 'Router', kind: 'node', status, summary, timestamp: `2026-09-15T10:00:0${seq}.000Z`, progress: seq / 10,
})

describe('API DTO normalization', () => {
  it('normalizes backend snake/camel fields and 0..1 progress', () => {
    const normalized = normalizeTraceEvent({ run_id: 'run-1', seq: 4, type: 'step.updated', node: 'eval_search', status: 'running', summary: 'Scoring papers', details: { score: 0.81, selected: 2, links: [{ label: 'ArXiv', href: 'https://arxiv.org/abs/1706.03762' }] }, progress: 0.32, timestamp: '2026-09-15T10:00:00Z' })
    expect(normalized.runId).toBe('run-1')
    expect(normalized.progress).toBe(32)
    expect(normalized.details).toEqual({ score: 0.81, selected: 2, links: [{ label: 'ArXiv', href: 'https://arxiv.org/abs/1706.03762' }] })
    expect(normalized.links).toEqual([{ label: 'ArXiv', href: 'https://arxiv.org/abs/1706.03762' }])
    expect(normalizeTraceEvent({ seq: 5, type: 'step.completed', node: 'router', status: 'running' }).status).toBe('completed')
  })

  it('reconciles duplicate and out-of-order SSE events by sequence', () => {
    const merged = reconcileEvents([event(1, 'completed'), event(3, 'running')], [event(2, 'completed'), event(1, 'running', 'stale'), event(3, 'completed', 'new')])
    expect(merged.map((item) => item.seq)).toEqual([1, 2, 3])
    expect(merged[0].status).toBe('running')
    expect(merged[2].summary).toBe('new')
  })

  it('uses a snapshot as the authoritative reconnect source', () => {
    const snapshot = normalizeSnapshot({ runId: 'run-9', input: { query: 'graph memory', paperInputs: ['1706.03762'], pdfNames: ['notes.pdf'] }, provider: 'zai', model: 'glm-5.3-flash', status: 'running', currentNode: 'read_paper', seq: 3, traceEvents: [event(1, 'completed'), event(3, 'running')] })
    expect(snapshot.runId).toBe('run-9')
    expect(snapshot.input.fileNames).toEqual(['notes.pdf'])
    expect(snapshot.currentNode).toBe('read_paper')
    expect(snapshot.seq).toBe(3)
    expect(snapshot.traceEvents).toHaveLength(2)
  })
})
