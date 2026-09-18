/*
 * Artifact links borrow the compact Tool Chip vocabulary from Beautiful UI's
 * Chat catalog (https://www.beautifului.dev/r/registry.json). The original
 * Tailwind demo is reduced to source-backed links only; there is no invented
 * tool output or loading state here. See ../NOTICE for the MIT notice.
 */
import { ExternalLink, FileCode2, FileText, Github, Link2 } from 'lucide-react'
import { Link } from 'react-router-dom'
import type { ChatArtifact } from './contracts'

function safeHref(href: string): string | null {
  if (/^#workspace:(papers|pmrl|compare|report|run|trace)$/.test(href)) return href
  if (href.startsWith('/')) return href
  try {
    const url = new URL(href)
    return url.protocol === 'http:' || url.protocol === 'https:' ? url.toString() : null
  } catch {
    return null
  }
}

function ArtifactIcon({ kind }: { kind?: string }) {
  const normalized = kind?.toLowerCase() ?? ''
  if (normalized.includes('github') || normalized.includes('repo')) return <Github size={13} aria-hidden="true" />
  if (normalized.includes('bib')) return <FileCode2 size={13} aria-hidden="true" />
  if (normalized.includes('report') || normalized.includes('paper') || normalized.includes('pdf')) return <FileText size={13} aria-hidden="true" />
  return <Link2 size={13} aria-hidden="true" />
}

export function ArtifactLinks({ artifacts }: { artifacts: ChatArtifact[] }) {
  const validArtifacts = artifacts
    .map((artifact) => ({ ...artifact, href: safeHref(artifact.href) }))
    .filter((artifact): artifact is ChatArtifact & { href: string } => Boolean(artifact.href))
  if (!validArtifacts.length) return null

  return (
    <div className="artifact-links" aria-label="Tài liệu và nguồn liên quan">
      {validArtifacts.map((artifact) => {
        if (artifact.href.startsWith('#workspace:')) return <button key={artifact.id} type="button" className="artifact-chip" onClick={() => window.dispatchEvent(new CustomEvent('research-scout:workspace', { detail: artifact.href.slice('#workspace:'.length) }))}><ArtifactIcon kind={artifact.kind} /><span>{artifact.label}</span></button>
        const internal = artifact.href.startsWith('/')
        const content = <><ArtifactIcon kind={artifact.kind} /><span>{artifact.label}</span>{internal ? null : <ExternalLink size={11} aria-hidden="true" />}</>
        return internal
          ? <Link key={artifact.id} className="artifact-chip" to={artifact.href}>{content}</Link>
          : <a key={artifact.id} className="artifact-chip" href={artifact.href} target="_blank" rel="noreferrer">{content}</a>
      })}
    </div>
  )
}
