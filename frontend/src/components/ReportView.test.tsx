import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ReportView } from './ReportView'

describe('ReportView', () => {
  it('does not repeat the report heading beside a page title', () => {
    render(<><h1>Final report</h1><ReportView title="Final report" markdown={'# Final report\n\n## Findings\n\nBody'} /></>)
    expect(screen.getAllByRole('heading', { name: 'Final report' })).toHaveLength(1)
    expect(screen.getByRole('heading', { name: 'Findings' })).toBeInTheDocument()
  })
})
