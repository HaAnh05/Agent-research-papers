import React from 'react'
import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { MarkdownContent, titleFromMarkdown } from './MarkdownContent'

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
})
