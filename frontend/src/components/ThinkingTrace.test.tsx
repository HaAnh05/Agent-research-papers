import React from 'react'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { ThinkingTrace } from './ThinkingTrace'
import type { TraceEvent } from '../types'

const events: TraceEvent[] = [{ runId: 'run-1', seq: 1, type: 'step.completed', node: 'router', label: 'Router', kind: 'routing', status: 'completed', summary: 'Đã chọn direct_read.', details: { intent: 'direct_read' }, timestamp: '2026-09-15T10:00:00Z', progress: 8 }]

describe('ThinkingTrace', () => {
  it('is open while running and allows manual detail expansion', async () => {
    const user = userEvent.setup()
    render(<ThinkingTrace events={events} running />)
    expect(screen.getByRole('button', { name: /thu gọn/i })).toHaveAttribute('aria-expanded', 'true')
    await user.click(screen.getByRole('button', { name: /xem chi tiết/i }))
    expect(screen.getByText('intent: direct_read')).toBeInTheDocument()
  })

  it('keeps the trace available after the run settles', () => {
    render(<ThinkingTrace events={events} />)
    expect(screen.getAllByText('Đã chọn direct_read.')).toHaveLength(2)
    expect(screen.getByText('hoàn tất')).toBeInTheDocument()
  })
})
