import React from 'react'
import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { MarkdownContent, titleFromMarkdown } from './MarkdownContent'
import { GAUS_SLAM_REPORT_EXCERPT } from '../test/fixtures/gausSlamReport'

describe('MarkdownContent', () => {
  it('renders GFM tables and math without enabling raw HTML', () => {
    const { container } = render(<MarkdownContent content={'# Scout report\n\n| Paper | Score |\n| --- | ---: |\n| A | 0.81 |\n\n$E = mc^2$\n\n<div>unsafe</div>'} />)
    expect(container.querySelector('table')).toBeInTheDocument()
    expect(container.querySelector('.katex')).toBeInTheDocument()
    expect(container.querySelector('.markdown-content > div:not(.markdown-table-wrap)')).not.toBeInTheDocument()
  })

  it('extracts the first heading for report cards', () => {
    expect(titleFromMarkdown('# A careful title\n\nBody')).toBe('A careful title')
    expect(titleFromMarkdown('No heading', 'Fallback')).toBe('Fallback')
  })

  it('keeps unverified Markdown destinations as text with a warning', () => {
    const { container } = render(<MarkdownContent content={'[valid](https://arxiv.org/abs/1706.03762) [placeholder](/tmp/report.md) [unknown](https://unverified.example/paper)'} />)
    expect(container.querySelector('a[href="https://arxiv.org/abs/1706.03762"]')).toBeInTheDocument()
    expect(container.querySelectorAll('.markdown-link-warning')).toHaveLength(2)
    expect(container.querySelector('a[href="https://unverified.example/paper"]')).not.toBeInTheDocument()
    expect(container.querySelector('.markdown-link-warning a')).not.toBeInTheDocument()
  })

  it('renders the current GauS-SLAM report math and table without rewriting TeX', () => {
    const { container } = render(<MarkdownContent content={GAUS_SLAM_REPORT_EXCERPT} />)
    const tex = [...container.querySelectorAll('annotation[encoding="application/x-tex"]')]
      .map((node) => node.textContent)

    expect(tex).toContain('B = 4')
    expect(tex).toContain('N_{\\text{gs}} > \\tau_l \\cdot H \\cdot W, \\qquad \\tau_l = 1.5')
    expect(tex).toContain('p_{\\text{new}} > \\tau_k, \\qquad \\tau_k = 0.01')
    expect(tex).toContain('\\mathcal{L} = \\lambda_1 \\, \\mathcal{L}_{\\text{rgb}} + \\lambda_2 \\, \\mathcal{L}_{\\text{depth}}')
    expect(container.querySelectorAll('.katex')).toHaveLength(4)
    expect(container.querySelector('.katex-error')).not.toBeInTheDocument()
    expect(container.querySelector('table')).toBeInTheDocument()
    expect(container).toHaveTextContent('có bước hiệu chỉnh độ sâu')
  })

  it('normalizes bracket delimiters to inline and display math only', () => {
    const content = String.raw`Inline \(B=4\).

\[N_{gs} > 1.5HW\]`
    const { container } = render(<MarkdownContent content={content} />)

    expect(container.querySelectorAll('.katex')).toHaveLength(2)
    expect(container.querySelectorAll('.katex-display')).toHaveLength(1)
  })

  it('keeps Vietnamese TeX and long parenthesized math inline', () => {
    const content = String.raw`Giữ $\text{mất mát}=\lambda_1L_{rgb}+\lambda_2L_{depth}$ ở inline và \(a_1+a_2+a_3+a_4+a_5+a_6+a_7+a_8+a_9+a_{10}+a_{11}+a_{12}+a_{13}+a_{14}+a_{15}\) trong câu.`
    const { container } = render(<MarkdownContent content={content} />)

    expect(container.querySelectorAll('.katex')).toHaveLength(2)
    expect(container.querySelector('.katex-display')).not.toBeInTheDocument()
  })
})
