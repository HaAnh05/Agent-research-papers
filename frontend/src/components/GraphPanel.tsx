import { useEffect, useMemo, useRef, useState } from 'react'
import { fmtNum, shortTitle } from '../lib/format'
import type { GraphNode, SimilarityGraph } from '../types'
import { Close } from './icons'

type Tab = 'sim' | 'prior' | 'deriv'

interface Layout {
  px: number[]
  py: number[]
}

interface LabelBox {
  i: number
  x: number
  y: number
  left: boolean
  text: string
  origin: boolean
}

const isOrigin = (n: GraphNode) => n.role === 'origin'
const sim = (n: GraphNode) => n.similarity ?? 0.1
const radius = (n: GraphNode) => (isOrigin(n) ? 11 : 3.5 + Math.sqrt(Math.log10((n.citedByCount ?? 0) + 1)) * 6.5)
const cites = (n: GraphNode) => (n.citedByCount == null ? '—' : `${fmtNum(n.citedByCount)} tr.d.`)
const nodeMeta = (n: GraphNode) => [n.firstAuthor, n.year].filter(Boolean).join(', ')

function mulberry(seed: number) {
  let s = seed
  return () => ((s = Math.imul(s ^ (s >>> 15), s | 1) ^ (s + Math.imul(s ^ (s >>> 7), s | 61))) >>> 0) / 4294967296
}

/** Force layout ported from the mockup: seeded start by similarity, repulsion + weighted attraction, cooling. */
function layout(nodes: GraphNode[], edges: Array<{ a: number; b: number; w: number }>, W: number, H: number): Layout {
  const n = nodes.length
  const rand = mulberry(5)
  const cx = W / 2
  const cy = H / 2
  const kd = Math.sqrt((W * H) / n) * 0.8
  const px = new Array<number>(n)
  const py = new Array<number>(n)
  const origins = nodes.map((x, i) => (isOrigin(x) ? i : -1)).filter((i) => i >= 0)
  const fixed = new Map<number, [number, number]>()
  origins.forEach((i, k) => {
    if (origins.length === 1) fixed.set(i, [cx, cy])
    else {
      const a = (k / origins.length) * Math.PI * 2 - Math.PI / 2
      fixed.set(i, [cx + Math.cos(a) * Math.min(W, H) * 0.12, cy + Math.sin(a) * Math.min(W, H) * 0.12])
    }
  })
  nodes.forEach((x, i) => {
    const a = rand() * 6.283
    const d = (1 - Math.min(sim(x), 1)) * Math.min(W, H) * 0.5
    px[i] = cx + Math.cos(a) * d
    py[i] = cy + Math.sin(a) * d
  })
  let temp = Math.min(W, H) / 8
  const vx = new Array<number>(n)
  const vy = new Array<number>(n)
  for (let it = 0; it < 500; it++) {
    vx.fill(0)
    vy.fill(0)
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = px[i] - px[j]
        const dy = py[i] - py[j]
        const d = Math.hypot(dx, dy) || 0.1
        const f = (kd * kd) / d
        vx[i] += (dx / d) * f
        vy[i] += (dy / d) * f
        vx[j] -= (dx / d) * f
        vy[j] -= (dy / d) * f
      }
    }
    for (const e of edges) {
      const dx = px[e.a] - px[e.b]
      const dy = py[e.a] - py[e.b]
      const d = Math.hypot(dx, dy) || 0.1
      const f = ((d * d) / kd) * (0.35 + e.w * 2)
      vx[e.a] -= (dx / d) * f
      vy[e.a] -= (dy / d) * f
      vx[e.b] += (dx / d) * f
      vy[e.b] += (dy / d) * f
    }
    for (let i = 0; i < n; i++) {
      const pin = fixed.get(i)
      if (pin) {
        px[i] = pin[0]
        py[i] = pin[1]
        continue
      }
      vx[i] += (cx - px[i]) * 0.01 * (kd / 10)
      vy[i] += (cy - py[i]) * 0.01 * (kd / 10)
      const d = Math.hypot(vx[i], vy[i]) || 1
      px[i] += (vx[i] / d) * Math.min(d, temp)
      py[i] += (vy[i] / d) * Math.min(d, temp)
      px[i] = Math.max(22, Math.min(W - 22, px[i]))
      py[i] = Math.max(20, Math.min(H - 34, py[i]))
    }
    temp *= 0.985
  }
  return { px, py }
}

/** Label collision layer: only the 8 most cited-and-similar nodes (and origins) get a label, none may overlap. */
function placeLabels(nodes: GraphNode[], order: number[], pos: Layout, W: number): LabelBox[] {
  const { px, py } = pos
  const top = new Set(
    nodes
      .map((n, i) => [n, i] as const)
      .filter(([n]) => !isOrigin(n))
      .sort((a, b) => (b[0].citedByCount ?? 0) * sim(b[0]) - (a[0].citedByCount ?? 0) * sim(a[0]))
      .slice(0, 8)
      .map(([, i]) => i),
  )
  const originBoxes = nodes.flatMap((n, i) => (isOrigin(n) ? [[px[i] - 40, py[i] - 14, px[i] + 60, py[i] + 14]] : []))
  const placed: number[][] = []
  const out: LabelBox[] = []
  for (const i of order) {
    const n = nodes[i]
    const origin = isOrigin(n)
    const text = origin ? shortTitle(n.title) : n.year ? `${shortTitle(n.title)}, ${n.year}` : shortTitle(n.title)
    const lw = text.length * 5.9
    const left = px[i] > W * 0.62
    const r = radius(n)
    const bx = left ? px[i] - r - 4 - lw : px[i] + r + 4
    const box = [bx, py[i] - 8, bx + lw, py[i] + 6]
    const hits = (q: number[]) => !(box[2] < q[0] || box[0] > q[2] || box[3] < q[1] || box[1] > q[3])
    const free = origin || (top.has(i) && bx > 2 && box[2] < W - 2 && !placed.some(hits) && !originBoxes.some(hits))
    if (!free) continue
    placed.push(box)
    out.push({ i, x: left ? px[i] - r - 4 : px[i] + r + 4, y: py[i] + 3.5, left, text, origin })
  }
  return out
}

export function GraphPanel({
  graph,
  onClose,
  onAsk,
  onAnalyse,
}: {
  graph: SimilarityGraph
  onClose: () => void
  onAsk: (text: string) => void
  onAnalyse: (node: GraphNode) => void
}) {
  const nodes = graph.nodes
  const originIdx = useMemo(() => nodes.map((n, i) => (isOrigin(n) ? i : -1)).filter((i) => i >= 0), [nodes])
  const origin = originIdx.length ? nodes[originIdx[0]] : undefined
  const originName = originIdx.length === 1 && origin ? shortTitle(origin.title) : originIdx.length > 1 ? `${originIdx.length} bài đã chọn` : 'bài gốc'

  const wrapRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ W: 0, H: 0 })
  const [hover, setHover] = useState<number | null>(null)
  const [selected, setSelected] = useState<number>(-1)
  const [tab, setTab] = useState<Tab>('sim')

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const measure = () => {
      const r = el.getBoundingClientRect()
      setSize((prev) => (Math.abs(prev.W - r.width) < 2 && Math.abs(prev.H - r.height) < 2 ? prev : { W: Math.round(r.width), H: Math.round(r.height) }))
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const edges = useMemo(() => {
    const index = new Map(nodes.map((n, i) => [n.id, i]))
    return graph.edges
      .map((e) => ({ a: index.get(e.source), b: index.get(e.target), w: e.weight }))
      .filter((e): e is { a: number; b: number; w: number } => e.a != null && e.b != null)
  }, [graph.edges, nodes])

  const pos = useMemo(() => (size.W > 0 && size.H > 0 ? layout(nodes, edges, size.W, size.H) : null), [nodes, edges, size.W, size.H])

  const order = useMemo(() => {
    const idx = nodes.map((_, i) => i)
    idx.sort((a, b) => Number(isOrigin(nodes[b])) - Number(isOrigin(nodes[a])) || radius(nodes[b]) - radius(nodes[a]))
    // origin nodes last so they are drawn on top
    return [...idx.filter((i) => !isOrigin(nodes[i])), ...idx.filter((i) => isOrigin(nodes[i]))]
  }, [nodes])

  const labels = useMemo(() => {
    if (!pos) return []
    // labels are placed in the mockup order: origin first, then larger nodes first
    const placeOrder = [...order.filter((i) => isOrigin(nodes[i])), ...order.filter((i) => !isOrigin(nodes[i]))]
    return placeLabels(nodes, placeOrder, pos, size.W)
  }, [pos, order, nodes, size.W])

  const years = nodes.map((n) => n.year).filter((y): y is number => typeof y === 'number')
  const yLo = years.length ? Math.min(...years) : 2022
  const yHi = years.length && Math.max(...years) > yLo ? Math.max(...years) : yLo + 4
  const yearColor = (y?: number | null) => {
    const t = Math.max(0, Math.min(1, ((y ?? yLo) - yLo) / (yHi - yLo)))
    const a = [0xd9, 0xdc, 0xe0]
    const b = [0x3d, 0x63, 0xb5]
    return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(',')})`
  }

  const neighbors = useMemo(() => {
    if (hover == null) return null
    const set = new Set<number>([hover])
    edges.forEach((e) => {
      if (e.a === hover) set.add(e.b)
      if (e.b === hover) set.add(e.a)
    })
    return set
  }, [hover, edges])

  const similar = useMemo(
    () => nodes.map((n, i) => [n, i] as const).filter(([n]) => !isOrigin(n)).sort((a, b) => sim(b[0]) - sim(a[0])),
    [nodes],
  )
  const pick = (i: number) => {
    setSelected(i)
    setTab('sim')
  }
  const sel = selected >= 0 && !isOrigin(nodes[selected]) ? nodes[selected] : null

  const tabs: Array<[Tab, string, number]> = [
    ['sim', 'Tương tự', similar.length],
    ['prior', 'Nền tảng', graph.priorWorks.length],
    ['deriv', 'Kế thừa', graph.derivativeWorks.length],
  ]

  return (
    <aside className="panel" aria-label="Bản đồ liên quan">
      <div className="panel-inner">
        <div className="ph">
          <div>
            <h2>Bản đồ liên quan</h2>
            <p>
              {similar.length} bài dùng chung nhiều tài liệu tham khảo nhất với {originName}. Bài càng gần, nền tảng càng giống.
            </p>
          </div>
          <button type="button" className="icon-btn" aria-label="Đóng bản đồ" onClick={onClose}>
            <Close />
          </button>
        </div>
        <div ref={wrapRef} className={`gwrap${hover != null ? ' focus' : ''}`}>
          <svg role="group" aria-label="Đồ thị bài liên quan. Dùng phím Tab để duyệt từng bài." viewBox={`0 0 ${size.W || 1} ${size.H || 1}`}>
            {pos && (
              <>
                <g>
                  {edges.map((e, k) => (
                    <line
                      key={k}
                      className={`edge${hover != null && (e.a === hover || e.b === hover) ? ' hl' : ''}`}
                      x1={pos.px[e.a]}
                      y1={pos.py[e.a]}
                      x2={pos.px[e.b]}
                      y2={pos.py[e.b]}
                      strokeWidth={0.5 + e.w * 4}
                      style={{ opacity: Math.min(1, 0.1 + e.w) }}
                    />
                  ))}
                  {order.map((i) => {
                    const n = nodes[i]
                    const cls = ['node', isOrigin(n) && 'origin', i === selected && !isOrigin(n) && 'sel', neighbors?.has(i) && 'hl'].filter(Boolean).join(' ')
                    return (
                      <g
                        key={n.id}
                        className={cls}
                        tabIndex={0}
                        role="button"
                        aria-label={`${n.title}${n.year ? `, ${n.year}` : ''}, ${n.citedByCount ?? 'chưa rõ'} trích dẫn`}
                        onClick={() => pick(i)}
                        onKeyDown={(ev) => {
                          if (ev.key === 'Enter' || ev.key === ' ') {
                            ev.preventDefault()
                            pick(i)
                          }
                        }}
                        onMouseEnter={() => setHover(i)}
                        onFocus={() => setHover(i)}
                        onMouseLeave={() => setHover(null)}
                        onBlur={() => setHover(null)}
                      >
                        <circle cx={pos.px[i]} cy={pos.py[i]} r={radius(n)} fill={isOrigin(n) ? '#1B1B19' : yearColor(n.year)} />
                      </g>
                    )
                  })}
                </g>
                <g>
                  {labels.map((l) => (
                    <text
                      key={l.i}
                      className="lbl"
                      x={l.x}
                      y={l.y}
                      textAnchor={l.left ? 'end' : undefined}
                      style={l.origin ? { fill: '#1B1B19', fontWeight: 600 } : undefined}
                    >
                      {l.text}
                    </text>
                  ))}
                </g>
              </>
            )}
          </svg>
          <div className="legend" aria-hidden="true">
            <span className="dots">
              <i style={{ width: 5, height: 5 }} />
              <i style={{ width: 9, height: 9 }} />
              <i style={{ width: 14, height: 14 }} />
              số trích dẫn
            </span>
            <span>
              {yLo}
              <span className="ramp" />
              {yHi}
            </span>
          </div>
        </div>
        <div className="tabs" role="tablist">
          {tabs.map(([key, label, count]) => (
            <button key={key} type="button" role="tab" aria-selected={tab === key} onClick={() => setTab(key)}>
              {label}
              <span className="count">{count}</span>
            </button>
          ))}
        </div>
        <div className="plist">
          {tab === 'sim' && (
            <>
              {sel && (
                <div className="detail">
                  <h4>{sel.title}</h4>
                  <div className="dm tnum">
                    {[sel.firstAuthor, sel.year, sel.citedByCount != null ? `${fmtNum(sel.citedByCount)} trích dẫn` : null, sel.similarity != null ? `tương đồng ${sel.similarity.toFixed(2)}` : null]
                      .filter((x) => x != null && x !== '')
                      .join(', ')}
                  </div>
                  {sel.abstract && <p>{sel.abstract.slice(0, 260)}…</p>}
                  <div className="acts">
                    <button type="button" className="btn" onClick={() => onAsk(`So sánh ${shortTitle(sel.title)} với ${originName}`)}>
                      Hỏi về bài này
                    </button>
                    {sel.arxivId && (
                      <button type="button" className="btn primary" onClick={() => onAnalyse(sel)}>
                        Phân tích bài này
                      </button>
                    )}
                  </div>
                </div>
              )}
              <ol>
                {similar.map(([n, i]) => (
                  <li key={n.id}>
                    <button type="button" onClick={() => pick(i)} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
                      <span className="pt">{n.title}</span>
                      <span className="pc tnum">{cites(n)}</span>
                      <span className="pm">{nodeMeta(n)}</span>
                    </button>
                  </li>
                ))}
              </ol>
            </>
          )}
          {tab === 'prior' && (
            <>
              <p className="empty lead-in">Các bài mà nhiều bài trên bản đồ cùng trích dẫn. Đây thường là nền tảng của hướng nghiên cứu.</p>
              <WorkList items={graph.priorWorks} meta={(n) => `${n.year ?? ''}${n.support != null ? `, ${n.support} bài trên bản đồ trích dẫn` : ''}`} />
            </>
          )}
          {tab === 'deriv' && (
            <>
              {origin && origin.citedByCount === 0 ? (
                <p className="empty lead-in">
                  <b>Chưa có bài nào trích dẫn {originName}.</b>OpenAlex ghi nhận 0 trích dẫn. Danh sách dưới đây là các bài trích dẫn nhiều bài trên bản đồ.
                </p>
              ) : (
                <p className="empty lead-in">Các bài trích dẫn nhiều bài trên bản đồ. Đây thường là bài mới hơn hoặc bài tổng quan của hướng này.</p>
              )}
              <WorkList items={graph.derivativeWorks} meta={(n) => `${n.year ?? ''}${n.support != null ? `, trích dẫn ${n.support} bài trên bản đồ` : ''}`} />
            </>
          )}
        </div>
      </div>
    </aside>
  )
}

function WorkList({ items, meta }: { items: GraphNode[]; meta: (n: GraphNode) => string }) {
  if (!items.length) return <p className="empty">Chưa có dữ liệu.</p>
  return (
    <ol>
      {items.map((n) => (
        <li key={n.id}>
          <div className="row">
            <span className="pt">{n.title}</span>
            <span className="pc tnum">{cites(n)}</span>
            <span className="pm">{meta(n).replace(/^, /, '')}</span>
          </div>
        </li>
      ))}
    </ol>
  )
}
