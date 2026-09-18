import { useEffect, useMemo, useRef, useState } from 'react'
import { BookOpen, ChevronLeft, FileText, LibraryBig, Menu, Moon, PanelRightClose, PanelRightOpen, Plus, Radio, Sun, X } from 'lucide-react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { useResearch } from '../context/ResearchContext'
import { formatThreadDate, normalizeThreads, readThreadId, type ThreadModel } from './chat/contracts'
import { ResearchInspector } from './chat/ResearchInspector'
import { InspectorWorkspace } from './inspector/InspectorWorkspace'

type Theme = 'light' | 'dark'

function ThemeToggle({ theme, onToggle }: { theme: Theme; onToggle: () => void }) {
  const nextTheme = theme === 'light' ? 'dark' : 'light'
  return <button type="button" className="theme-toggle" onClick={onToggle} aria-label={`Chuyển sang giao diện ${nextTheme === 'dark' ? 'tối' : 'sáng'}`} title={`Giao diện ${nextTheme === 'dark' ? 'tối' : 'sáng'}`}>{theme === 'light' ? <Moon size={15} /> : <Sun size={15} />}<span>{theme === 'light' ? 'Tối' : 'Sáng'}</span></button>
}

function ProviderStatus({ research }: { research: ReturnType<typeof useResearch> }) {
  const config = research.config
  const configured = config?.configured ?? false
  return <div className="provider-status" aria-label="Trạng thái provider"><span className={`status-dot ${configured ? 'is-ready' : 'is-muted'}`} aria-hidden="true" /><div className="provider-status__copy"><span className="provider-status__label">Provider</span><span className={`provider-status__value ${configured ? '' : 'is-muted'}`}>{config?.provider ?? 'Chưa cấu hình'}</span></div><code className="provider-status__model" title={config?.model}>{config?.model ?? '—'}</code></div>
}

function threadTitle(thread: ThreadModel): string {
  return thread.title || 'Research thread'
}

function ThreadRow({ thread, active, onSelect }: { thread: ThreadModel; active: boolean; onSelect: (thread: ThreadModel) => void }) {
  return <button type="button" className={`thread-row ${active ? 'is-active' : ''}`} onClick={() => onSelect(thread)} aria-current={active ? 'page' : undefined}><span className="thread-row__mark" aria-hidden="true"><FileText size={13} /></span><span className="thread-row__body"><strong>{threadTitle(thread)}</strong><small>{formatThreadDate(thread.updatedAt ?? thread.createdAt)}{thread.messageCount ? ` · ${thread.messageCount} tin nhắn` : ''}</small></span>{thread.status === 'running' ? <span className="thread-row__live" aria-label="đang chạy" /> : null}</button>
}

function SidebarContent({ research, theme, onThemeToggle, onNavigate }: { research: ReturnType<typeof useResearch>; theme: Theme; onThemeToggle: () => void; onNavigate?: () => void }) {
  const location = useLocation()
  const navigate = useNavigate()
  const currentThreadId = readThreadId(research.currentThread) ?? research.currentRun?.runId
  const threads = useMemo(() => normalizeThreads(research.threads), [research.threads])

  const handleNewThread = async () => {
    await research.newThread()
    onNavigate?.()
    navigate('/')
  }

  const handleSelect = async (thread: ThreadModel) => {
    research.selectThread(thread.id)
    await research.loadThread(thread.id)
    onNavigate?.()
    navigate(`/run/${encodeURIComponent(thread.id)}`)
  }

  return <>
    <div className="desk-brand"><span className="desk-brand__mark" aria-hidden="true"><BookOpen size={17} strokeWidth={1.8} /></span><div><strong>Research Scout</strong><small>Annotated Research Desk</small></div></div>
    <button type="button" className="new-thread-button" onClick={() => void handleNewThread()}><Plus size={16} /><span>Nghiên cứu mới</span></button>
    <div className="sidebar-section"><div className="sidebar-section__heading"><span>Trong phiên</span><span>{threads.length || ''}</span></div><div className="thread-list" aria-label="Các thread nghiên cứu">{threads.length ? threads.map((thread) => <ThreadRow key={thread.id} thread={thread} active={thread.id === currentThreadId} onSelect={(item) => void handleSelect(item)} />) : <p className="thread-list__empty">Chưa có thread. Bắt đầu một research mới.</p>}</div></div>
    <nav className="sidebar-nav" aria-label="Điều hướng workspace">
      <NavLink to="/run/current" className={({ isActive }) => `sidebar-nav__link ${isActive ? 'is-active' : ''}`} onClick={onNavigate}><Radio size={16} /><span>Current run</span>{research.activeRun ? <span className="sidebar-nav__pulse" aria-label="đang chạy" /> : null}</NavLink>
      <NavLink to="/library" className={({ isActive }) => `sidebar-nav__link ${isActive ? 'is-active' : ''}`} onClick={onNavigate}><LibraryBig size={16} /><span>Thư viện</span></NavLink>
    </nav>
    <div className="sidebar-spacer" />
    <div className="sidebar-note"><span className="sidebar-note__icon" aria-hidden="true"><FileText size={14} /></span><span><strong>Operational trace</strong><small>Chỉ hiển thị sự kiện workflow an toàn.</small></span></div>
    <ProviderStatus research={research} />
    <div className="sidebar-footer"><ThemeToggle theme={theme} onToggle={onThemeToggle} /><span className="sidebar-footer__version">v0.2 desk</span></div>
    <span className="sr-only" aria-live="polite">{location.pathname === '/' ? 'Đang ở workspace nghiên cứu mới' : ''}</span>
  </>
}

export function AppShell() {
  const research = useResearch()
  const location = useLocation()
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [inspectorOpen, setInspectorOpen] = useState(() => typeof window === 'undefined' || !window.matchMedia?.('(max-width: 900px)').matches)
  const [theme, setTheme] = useState<Theme>(() => typeof document !== 'undefined' && document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light')
  const mobileMenuButtonRef = useRef<HTMLButtonElement>(null)
  const mobileInspectorButtonRef = useRef<HTMLButtonElement>(null)
  const drawerCloseRef = useRef<HTMLButtonElement>(null)
  const drawerRef = useRef<HTMLElement>(null)
  const restoreFocusRef = useRef<HTMLElement | null>(null)
  const showInspector = !location.pathname.startsWith('/library')

  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  useEffect(() => {
    const openWorkspace = () => setInspectorOpen(true)
    window.addEventListener('research-scout:workspace', openWorkspace)
    return () => window.removeEventListener('research-scout:workspace', openWorkspace)
  }, [])

  useEffect(() => {
    if (!inspectorOpen || !window.matchMedia('(max-width: 900px)').matches) return
    const panel = document.querySelector<HTMLElement>('.research-inspector')
    requestAnimationFrame(() => panel?.querySelector<HTMLButtonElement>('button')?.focus())
    const handleKeys = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        setInspectorOpen(false)
        requestAnimationFrame(() => mobileInspectorButtonRef.current?.focus())
      }
      if (event.key !== 'Tab' || !panel) return
      const focusable = Array.from(panel.querySelectorAll<HTMLElement>('a[href], button:not([disabled]), input:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'))
      if (!focusable.length) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus() }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
    }
    document.addEventListener('keydown', handleKeys)
    return () => document.removeEventListener('keydown', handleKeys)
  }, [inspectorOpen])

  useEffect(() => {
    if (!drawerOpen) {
      const element = restoreFocusRef.current
      restoreFocusRef.current = null
      if (element && document.contains(element)) requestAnimationFrame(() => element.focus())
      return
    }

    requestAnimationFrame(() => drawerCloseRef.current?.focus())
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        setDrawerOpen(false)
        return
      }
      if (event.key !== 'Tab' || !drawerRef.current) return
      const focusable = Array.from(drawerRef.current.querySelectorAll<HTMLElement>('a[href], button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'))
      if (!focusable.length) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [drawerOpen])

  const openDrawer = () => {
    const active = document.activeElement
    restoreFocusRef.current = active instanceof HTMLElement ? active : mobileMenuButtonRef.current
    setDrawerOpen(true)
  }

  const closeDrawer = () => setDrawerOpen(false)
  const shellState = showInspector ? (inspectorOpen ? 'is-inspector-open' : 'is-inspector-collapsed') : 'app-shell--no-inspector'

  return <div className={`app-shell app-shell--${theme} ${shellState}`}>
    <aside className="app-sidebar app-sidebar--desktop"><SidebarContent research={research} theme={theme} onThemeToggle={() => setTheme((value) => value === 'light' ? 'dark' : 'light')} /></aside>
    <header className="app-mobile-header"><button ref={mobileMenuButtonRef} type="button" className="mobile-menu-button" onClick={openDrawer} aria-label="Mở menu" aria-expanded={drawerOpen} aria-controls="mobile-navigation"><Menu size={20} /></button><span className="mobile-brand"><span className="desk-brand__mark" aria-hidden="true"><BookOpen size={15} /></span>Research Scout</span><span className="mobile-header__rule" aria-hidden="true" />{showInspector ? <button ref={mobileInspectorButtonRef} type="button" className="mobile-inspector-button" onClick={() => setInspectorOpen((value) => !value)} aria-label={inspectorOpen ? 'Đóng inspector' : 'Mở inspector'} aria-expanded={inspectorOpen}>{inspectorOpen ? <PanelRightClose size={17} /> : <PanelRightOpen size={17} />}</button> : null}</header>
    {drawerOpen ? <><button type="button" className="drawer-scrim" onClick={closeDrawer} aria-label="Đóng menu" /><aside ref={drawerRef} id="mobile-navigation" className="app-sidebar app-sidebar--drawer" aria-label="Menu di động" role="dialog" aria-modal="true"><div className="drawer-top"><strong>Research Scout</strong><button ref={drawerCloseRef} type="button" className="mobile-menu-button" onClick={closeDrawer} aria-label="Đóng menu"><X size={19} /></button></div><SidebarContent research={research} theme={theme} onThemeToggle={() => setTheme((value) => value === 'light' ? 'dark' : 'light')} onNavigate={closeDrawer} /></aside></> : null}
    <div className="app-workspace"><main id="main-content" className="app-main"><Outlet /></main></div>
    {showInspector && inspectorOpen ? <button type="button" className="inspector-scrim" onClick={() => setInspectorOpen(false)} aria-label="Đóng research workspace" /> : null}
    {showInspector ? <ResearchInspector open={inspectorOpen} onToggle={() => setInspectorOpen((value) => !value)} threadId={readThreadId(research.currentThread)} runId={research.currentRun?.runId} modal={typeof window !== 'undefined' && window.matchMedia('(max-width: 900px)').matches && inspectorOpen}><InspectorWorkspace snapshot={research.currentRun ?? null} events={research.events ?? []} onAskSelection={(selection) => { if (window.matchMedia('(max-width: 900px)').matches) setInspectorOpen(false); window.dispatchEvent(new CustomEvent('research-scout:draft', { detail: `Giải thích đoạn sau dựa trên ngữ cảnh nghiên cứu hiện tại:\n\n“${selection}”` })) }} /></ResearchInspector> : null}
  </div>
}

export function PageIntro({ eyebrow, title, description, action }: { eyebrow?: string; title: string; description?: string; action?: React.ReactNode }) {
  return <div className="page-intro"><div>{eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}<h1>{title}</h1>{description ? <p className="page-intro__description">{description}</p> : null}</div>{action ? <div className="page-intro__action">{action}</div> : null}</div>
}

export function StatusBadge({ status }: { status: string }) {
  const labels: Record<string, string> = { running: 'Đang chạy', success: 'Hoàn tất', degraded: 'Một phần', error: 'Có lỗi', idle: 'Chưa chạy', completed: 'Hoàn tất' }
  return <span className={`status-badge status-badge--${status}`}><span className="status-badge__dot" aria-hidden="true" />{labels[status] ?? status}</span>
}

export function SectionHeading({ title, count, action }: { title: string; count?: number; action?: React.ReactNode }) {
  return <div className="section-heading"><h2>{title}{typeof count === 'number' ? <span className="count-chip">{count}</span> : null}</h2>{action}</div>
}

export function BackButton({ children = 'Quay lại' }: { children?: React.ReactNode }) {
  return <button type="button" className="text-button" onClick={() => window.history.back()}><ChevronLeft size={15} aria-hidden="true" />{children}</button>
}
