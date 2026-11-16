import { useEffect, useState, type ReactNode } from 'react'
import { scaleLinear } from 'd3-scale'
import { interpolateLab } from 'd3-interpolate'
import type { Resolved } from '../theme'

export function useWidth<T extends HTMLElement>(min = 200): [(el: T | null) => void, number] {
  const [el, setEl] = useState<T | null>(null)
  const [w, setW] = useState(0)
  useEffect(() => {
    if (!el) return
    const upd = () => setW(Math.max(min, Math.floor(el.clientWidth)))
    const ro = new ResizeObserver(upd)
    ro.observe(el); upd()
    return () => ro.disconnect()
  }, [el, min])
  return [setEl, w]
}

export interface Tip { x: number; y: number; body: ReactNode }
export function TipBox({ tip, w }: { tip: Tip | null; w: number }) {
  if (!tip) return null
  const right = tip.x > w * 0.55
  return <div className="tt" role="tooltip" style={{ top: Math.max(0, tip.y - 10), [right ? 'right' : 'left']: right ? w - tip.x + 12 : tip.x + 12 }}>{tip.body}</div>
}

type Lin = { (v: number): number; ticks: (n?: number) => number[] }
/** Horizontal gridlines and left tick labels. */
export function YGrid({ y, width, ticks = 4, fmt = (v: number) => String(+v.toFixed(3)), zero = false }: { y: Lin; width: number; ticks?: number; fmt?: (v: number) => string; zero?: boolean }) {
  return <g className="axis">{y.ticks(ticks).map(v => <g key={v}>
    <line x1={0} x2={width} y1={y(v)} y2={y(v)} className={zero && Math.abs(v) < 1e-12 ? 'zero' : 'grid'} />
    <text x={-6} y={y(v)} dy="0.32em" textAnchor="end">{fmt(v)}</text>
  </g>)}</g>
}
export function XTicks({ x, height, values, fmt = (v: number) => String(+v.toFixed(3)) }: { x: (v: number) => number; height: number; values: number[]; fmt?: (v: number) => string }) {
  return <g className="axis">{values.map(v => <g key={v}><line x1={x(v)} x2={x(v)} y1={height} y2={height + 4} className="tick" /><text x={x(v)} y={height + 15} textAnchor="middle">{fmt(v)}</text></g>)}</g>
}
export function AxisTitle({ x, y, children, rotate, anchor = 'middle' }: { x: number; y: number; children: ReactNode; rotate?: boolean; anchor?: 'start' | 'middle' | 'end' }) {
  return <text className="axis-title" x={rotate ? 0 : x} y={rotate ? 0 : y} textAnchor={anchor} transform={rotate ? `translate(${x},${y}) rotate(-90)` : undefined}>{children}</text>
}

/** Robust y-domain: ignores isolated extreme values so one outlier cannot flatten a chart. Returns [lo, hi, cut] where cut lists clipped values. */
export function robustDomain(vals: number[], pad = 0.06): { lo: number; hi: number } {
  const v = vals.filter(Number.isFinite).sort((a, b) => a - b)
  if (!v.length) return { lo: 0, hi: 1 }
  const q = (p: number) => v[Math.min(v.length - 1, Math.max(0, Math.round(p * (v.length - 1))))]
  const q1 = q(0.25), q3 = q(0.75), iqr = Math.max(q3 - q1, (Math.abs(q(0.5)) || 1) * 0.25, 1e-9)
  const lo0 = Math.max(v[0], q1 - 4 * iqr), hi0 = Math.min(v[v.length - 1], q3 + 4 * iqr)
  const span = Math.max(hi0 - lo0, Math.abs(hi0) * 0.1, 1e-6)
  return { lo: lo0 - span * pad, hi: hi0 + span * pad }
}

/** Diverging colour (blue, grey midpoint, red) for signed values; selected per theme, not flipped. */
const DIV: Record<Resolved, string[]> = {
  light: ['#2d5c8f', '#93b4d8', '#f4f2ee', '#efa47c', '#b4461b'],
  dark: ['#93b4d8', '#3a5d84', '#2c2b28', '#9a4c2a', '#f0a07a'],
}
export function divergingScale(maxAbs: number, theme: Resolved) {
  const m = maxAbs > 0 ? maxAbs : 1
  return scaleLinear<string>().domain([-m, -m / 2, 0, m / 2, m]).range(DIV[theme]).interpolate(interpolateLab).clamp(true)
}
/** Black or white text on a given hex fill. */
export function inkOn(c: string): string {
  let r = 0, g = 0, b = 0
  const h = c.match(/^#([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i)
  if (h) [r, g, b] = h.slice(1).map(s => parseInt(s, 16))
  else { const m = c.match(/rgba?\((\d+)[,\s]+(\d+)[,\s]+(\d+)/); if (!m) return '#111417'; [r, g, b] = m.slice(1).map(Number) }
  const lin = (c: number) => { const s = c / 255; return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4 }
  const L = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
  return L > 0.28 ? '#111417' : '#ffffff'
}

/** Diverging legend bar. */
export function DivLegend({ maxAbs, theme, fmt, label }: { maxAbs: number; theme: Resolved; fmt: (v: number) => string; label: string }) {
  const id = `dl-${theme}`
  const stops = DIV[theme]
  return (
    <div className="divlegend" aria-label={`Colour scale: ${label}`}>
      <span className="mono">{fmt(-maxAbs)}</span>
      <svg width={140} height={10} aria-hidden><defs><linearGradient id={id}>{stops.map((c, i) => <stop key={i} offset={i / (stops.length - 1)} stopColor={c} />)}</linearGradient></defs><rect width={140} height={10} rx={2} fill={`url(#${id})`} /></svg>
      <span className="mono">{fmt(maxAbs)}</span>
      <span className="note">{label}</span>
    </div>
  )
}

/** Spearman rank correlation, used to describe whether a dose curve is monotone. */
export function spearman(xs: number[], ys: number[]): number {
  const rank = (a: number[]) => { const idx = a.map((v, i) => [v, i] as const).sort((p, q) => p[0] - q[0]); const r = new Array(a.length); idx.forEach(([, i], k) => (r[i] = k)); return r as number[] }
  const n = xs.length; if (n < 3) return NaN
  const rx = rank(xs), ry = rank(ys)
  const mx = (n - 1) / 2
  let num = 0, dx = 0, dy = 0
  for (let i = 0; i < n; i++) { num += (rx[i] - mx) * (ry[i] - mx); dx += (rx[i] - mx) ** 2; dy += (ry[i] - mx) ** 2 }
  return num / Math.sqrt(dx * dy)
}

/** Linear interpolation of x for a target y along a piecewise-linear curve (points sorted by x). Returns null outside range. */
export function inverseInterp(pts: { x: number; y: number }[], target: number): number | null {
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i], b = pts[i + 1]
    if ((target - a.y) * (target - b.y) <= 0 && a.y !== b.y) return a.x + ((target - a.y) / (b.y - a.y)) * (b.x - a.x)
  }
  return null
}
