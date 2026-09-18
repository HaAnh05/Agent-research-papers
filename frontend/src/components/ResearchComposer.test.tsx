import React from 'react'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ResearchComposer, validateResearchInput } from './ResearchComposer'

describe('ResearchComposer', () => {
  it('requires at least one input source', async () => {
    expect(validateResearchInput('', [], [])).toMatch(/Thêm một câu hỏi/)
    const onSubmit = vi.fn().mockResolvedValue(undefined)
    const user = userEvent.setup()
    render(<ResearchComposer onSubmit={onSubmit} />)
    await user.click(screen.getByRole('button', { name: /bắt đầu scout/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/Thêm một câu hỏi/)
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('submits a query and paper inputs', async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined)
    const user = userEvent.setup()
    render(<ResearchComposer onSubmit={onSubmit} />)
    await user.type(screen.getByLabelText(/bạn muốn tìm hiểu/i), 'attention in vision')
    await user.type(screen.getByLabelText(/arxiv id/i), '1706.03762, https://arxiv.org/abs/2005.14165')
    await user.click(screen.getByRole('button', { name: /bắt đầu scout/i }))
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ query: 'attention in vision', paperInputs: ['1706.03762', 'https://arxiv.org/abs/2005.14165'], files: [] }))
  })

  it('rejects non-PDF files even when the browser omits the MIME type', () => {
    const textFile = new File(['notes'], 'notes.txt', { type: '' })
    expect(validateResearchInput('', [], [textFile])).toMatch(/không phải file PDF/)
    const pdf = new File(['pdf'], 'paper.pdf', { type: 'application/octet-stream' })
    expect(validateResearchInput('', [], [pdf])).toBeNull()
  })
})
