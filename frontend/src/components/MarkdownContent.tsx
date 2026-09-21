import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

export function titleFromMarkdown(markdown: string, fallback = 'Báo cáo nghiên cứu'): string {
  const heading = markdown.match(/^#\s+(.+)$/m)?.[1]?.trim()
  return heading?.replace(/[*_`]/g, '') || fallback
}

/** Remove the first H1 only for the rendered view; downloads keep the source unchanged. */
export function withoutFirstMarkdownHeading(markdown: string): string {
  return markdown.replace(/^#\s+.+(?:\r?\n|$)/m, '').replace(/^\s+/, '')
}

function safeMarkdownHref(href: string | undefined): string | undefined {
  if (!href?.trim()) return undefined
  try {
    const url = new URL(href)
    const host = url.hostname.toLowerCase()
    const allowed = host === 'arxiv.org' || host === 'www.arxiv.org' || host === 'export.arxiv.org' || host === 'github.com' || host === 'www.github.com'
    return url.protocol === 'https:' && allowed && !url.username && !url.password ? url.toString() : undefined
  } catch {
    return undefined
  }
}

/** Normalize only equivalent TeX delimiters; remark-math owns math parsing. */
function normalizeMarkdown(content: string): string {
  return content
    .replace(/<br\s*\/?\s*>/gi, '\n')
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, expression: string) => `\n$$\n${expression.trim()}\n$$\n`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_match, expression: string) => `$${expression.trim()}$`)
}

export function MarkdownContent({ content, compact = false }: { content: string; compact?: boolean }) {
  if (!content.trim()) return <div className="markdown-empty">Chưa có nội dung để hiển thị.</div>
  const normalizedContent = normalizeMarkdown(content)

  return (
    <div className={`markdown-content ${compact ? 'markdown-content--compact' : ''}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        // rehypeRaw is intentionally not enabled: report text must not execute
        // arbitrary HTML supplied by a model or a paper.
        components={{
          a: ({ href, children, node: _node, ...props }) => {
            const safeHref = safeMarkdownHref(href)
            if (!safeHref) {
              return <span className="markdown-link-warning" role="note" {...props}>{children}<small> (link unavailable)</small></span>
            }
            return <a href={safeHref} target="_blank" rel="noreferrer" {...props}>{children}</a>
          },
          table: ({ children, node: _node, ...props }) => <div className="markdown-table-wrap"><table {...props}>{children}</table></div>,
          pre: ({ children, node: _node, ...props }) => <pre className="markdown-code" {...props}>{children}</pre>,
          blockquote: ({ children, node: _node, ...props }) => <blockquote className="markdown-quote" {...props}>{children}</blockquote>,
        }}
      >
        {normalizedContent}
      </ReactMarkdown>
    </div>
  )
}
