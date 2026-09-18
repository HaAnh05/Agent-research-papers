import type { ReactNode } from 'react'
import { PanelRightClose, PanelRightOpen } from 'lucide-react'

/**
 * Stable seam for the inspector worker. Artifact panels intentionally do not
 * live here yet; consumers can provide children without changing AppShell.
 */
export interface ResearchInspectorProps {
  open: boolean
  onToggle: () => void
  threadId?: string
  runId?: string
  modal?: boolean
  children?: ReactNode
}

export function ResearchInspector({ open, onToggle, threadId, runId, modal = false, children }: ResearchInspectorProps) {
  return (
    <aside className={`research-inspector ${open ? 'is-open' : 'is-collapsed'}`} data-inspector-slot="true" data-thread-id={threadId} data-run-id={runId} aria-label="Inspector nghiên cứu" role={modal ? 'dialog' : undefined} aria-modal={modal ? true : undefined}>
      <div className="research-inspector__head">
        {open ? <span>Inspector</span> : null}
        <button type="button" className="inspector-toggle" onClick={onToggle} aria-label={open ? 'Thu gọn inspector' : 'Mở inspector'} aria-expanded={open}>
          {open ? <PanelRightClose size={16} /> : <PanelRightOpen size={16} />}
        </button>
      </div>
      {open ? children ?? <div className="research-inspector__slot"><span>Inspector slot</span><small>Artifact panels sẽ được gắn vào đây.</small></div> : null}
    </aside>
  )
}
