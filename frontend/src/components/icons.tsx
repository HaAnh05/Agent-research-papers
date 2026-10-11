import type { SVGProps } from 'react'

type P = SVGProps<SVGSVGElement>
const base = { fill: 'none', stroke: 'currentColor', 'aria-hidden': true } as const

export const Chev = (p: P) => (
  <svg className="chev" viewBox="0 0 16 16" strokeWidth="1.6" {...base} {...p}>
    <path d="m6 4 4 4-4 4" />
  </svg>
)
export const Tick = (p: P) => (
  <svg className="tick" viewBox="0 0 16 16" strokeWidth="1.8" {...base} {...p}>
    <path d="m3.5 8.5 3 3 6-7" />
  </svg>
)
export const Warn = (p: P) => (
  <svg className="tick warn" viewBox="0 0 16 16" strokeWidth="1.6" {...base} {...p}>
    <circle cx="8" cy="8" r="6.2" />
    <path d="M8 4.8v3.8M8 10.9v.1" />
  </svg>
)
export const Download = (p: P) => (
  <svg viewBox="0 0 16 16" strokeWidth="1.5" {...base} {...p}>
    <path d="M8 2.5v8M4.5 7 8 10.5 11.5 7M3 13.5h10" />
  </svg>
)
export const Copy = (p: P) => (
  <svg viewBox="0 0 16 16" strokeWidth="1.5" {...base} {...p}>
    <rect x="5" y="5" width="8.5" height="8.5" rx="1.5" />
    <path d="M10.5 5V3.5A1 1 0 0 0 9.5 2.5h-6a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1H5" />
  </svg>
)
export const MapIcon = (p: P) => (
  <svg viewBox="0 0 16 16" strokeWidth="1.4" {...base} {...p}>
    <circle cx="4" cy="11" r="2" />
    <circle cx="11.5" cy="4.5" r="2.5" />
    <circle cx="12" cy="12" r="1.5" />
    <path d="M5.6 9.8 9.6 6M5.9 11.3l4.6.6" />
  </svg>
)
export const Ok = (p: P) => (
  <svg viewBox="0 0 12 12" strokeWidth="1.8" {...base} {...p}>
    <path d="M2 6.5 5 9l5-6" />
  </svg>
)
export const Clip = (p: P) => (
  <svg viewBox="0 0 20 20" strokeWidth="1.5" {...base} {...p}>
    <path d="M13.5 6.5 7.6 12.4a1.8 1.8 0 0 0 2.5 2.6l6-6a3.4 3.4 0 0 0-4.8-4.8l-6 6a5 5 0 0 0 7.1 7.1l4.4-4.4" />
  </svg>
)
export const Send = (p: P) => (
  <svg viewBox="0 0 16 16" strokeWidth="1.8" {...base} {...p}>
    <path d="M8 13V3M3.5 7.5 8 3l4.5 4.5" />
  </svg>
)
export const Close = (p: P) => (
  <svg viewBox="0 0 20 20" strokeWidth="1.6" {...base} {...p}>
    <path d="M5 5l10 10M15 5 5 15" />
  </svg>
)
