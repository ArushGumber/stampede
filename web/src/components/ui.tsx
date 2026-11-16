import { useEffect, useState, type ReactNode, type CSSProperties } from 'react'
import * as D from '../data'

/** Load a promise-returning source into a Loaded<T> state. */
export function useLoaded<T>(f: () => Promise<T>, deps: unknown[]): D.Loaded<T> {
  const [s, setS] = useState<D.Loaded<T>>({ state: 'loading' })
  useEffect(() => {
    let dead = false
    setS(p => (p.state === 'ok' ? p : { state: 'loading' }))
    f().then(d => { if (!dead) setS({ state: 'ok', data: d }) }).catch(e => { if (!dead) setS(D.describeError(e)) })
    return () => { dead = true }
  }, deps) // eslint-disable-line react-hooks/exhaustive-deps
  return s
}
export const useResult = <T,>(name: string) => useLoaded<T>(() => D.result<T>(name), [name])
export const useSource = () => useLoaded(() => D.init(), [])

/** Honest empty state: says what is missing and what will appear when it exists. */
export function Empty({ title, children, what }: { title: string; children?: ReactNode; what?: string }) {
  return (
    <div className="empty" role="status">
      <div className="empty-t">{title}</div>
      {children && <div className="empty-b">{children}</div>}
      {what && <div className="empty-f mono">{what}</div>}
    </div>
  )
}
export function LoadState<T>({ l, label, expect }: { l: D.Loaded<T>; label: string; expect?: ReactNode }) {
  if (l.state === 'loading') return <p className="note" role="status">Loading {label}.</p>
  if (l.state === 'missing') return <Empty title={`No ${label} yet`} what={`${l.what} is not on the server`}>{expect ?? 'This experiment has not been exported to the site yet. The page fills in when the file appears.'}</Empty>
  if (l.state === 'error') return <div className="banner err" role="alert">Could not load {label}: {l.error}</div>
  return null
}

export function Panel({ title, sub, right, children, takeaway, className, id, info }: { title?: ReactNode; sub?: ReactNode; right?: ReactNode; children: ReactNode; takeaway?: ReactNode; className?: string; id?: string; info?: ReactNode }) {
  return (
    <section className={`panel ${className ?? ''}`} id={id}>
      {(title || right) && <header className="ph"><div className="ph-l">{title && <div className="ph-t"><h2>{title}</h2>{info && <Info>{info}</Info>}</div>}{sub && <span className="ph-sub">{sub}</span>}</div>{right && <div className="ph-r">{right}</div>}</header>}
      {takeaway && <Takeaway>{takeaway}</Takeaway>}
      {children}
    </section>
  )
}

/** Small circled-i icon that shows its children in a popover on hover or keyboard focus. */
export function Info({ children, align, label = 'More detail' }: { children: ReactNode; align?: 'left' | 'right'; label?: string }) {
  return (
    <span className={`info ${align ?? ''}`} tabIndex={0} role="button" aria-label={label}>
      <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden><circle cx="8" cy="8" r="6.6" fill="none" stroke="currentColor" strokeWidth="1.3" /><path d="M8 7.2v4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" /><circle cx="8" cy="4.9" r=".9" fill="currentColor" /></svg>
      <span className="info-pop" role="tooltip">{children}</span>
    </span>
  )
}

const flat = (n: ReactNode): string | null => {
  if (typeof n === 'string' || typeof n === 'number') return String(n)
  if (Array.isArray(n)) { let out = ''; for (const x of n) { if (x === null || x === undefined || x === false || x === true) continue; const f = flat(x); if (f === null) return null; out += f } return out }
  if (n === null || n === undefined || n === false || n === true) return ''
  return null
}
/** A one-line summary under a chart, computed from the data. Longer summaries show their first sentence and keep the
 *  rest behind an info icon. */
export function Takeaway({ children, full }: { children: ReactNode; full?: boolean }) {
  const t = flat(children)
  if (t === null || full) return <p className="takeaway">{children}</p>
  const parts = t.trim().split(/(?<=[.])\s+(?=[A-Z])/)
  if (parts.length < 2) return <p className="takeaway">{t}</p>
  return <p className="takeaway">{parts[0]} <Info align="left">{parts.slice(1).join(' ')}</Info></p>
}

export function Seg<T extends string | number>({ value, options, onChange, label, small }: { value: T; options: { v: T; label: ReactNode; title?: string; disabled?: boolean }[]; onChange: (v: T) => void; label?: string; small?: boolean }) {
  return (
    <div className="field">
      {label && <div className="lab">{label}</div>}
      <div className={`seg ${small ? 'sm' : ''}`} role="group" aria-label={label}>
        {options.map(o => <button key={String(o.v)} type="button" aria-pressed={value === o.v} title={o.title} disabled={o.disabled} onClick={() => onChange(o.v)}>{o.label}</button>)}
      </div>
    </div>
  )
}
export function Sel<T extends string>({ label, value, options, onChange, id }: { label: string; value: T; options: { v: T; label: string }[]; onChange: (v: T) => void; id?: string }) {
  const fid = id ?? `sel-${label.replace(/\W+/g, '-').toLowerCase()}`
  return (
    <div className="field">
      <label className="lab" htmlFor={fid}>{label}</label>
      <select id={fid} value={value} onChange={e => onChange(e.target.value as T)}>{options.map(o => <option key={o.v} value={o.v}>{o.label}</option>)}</select>
    </div>
  )
}

export type GlyphShape = 'circle' | 'ring' | 'diamond' | 'square' | 'triangle'
/** Marker glyph in SVG coordinates (centre cx, cy). */
export function GlyphMark({ shape, cx, cy, r = 4.5, color, ring = true, style }: { shape: GlyphShape; cx: number; cy: number; r?: number; color: string; ring?: boolean; style?: CSSProperties }) {
  const st: CSSProperties = { fill: shape === 'ring' ? 'var(--surface)' : color, stroke: shape === 'ring' ? color : ring ? 'var(--surface)' : 'none', strokeWidth: shape === 'ring' ? 1.6 : ring ? 1.5 : 0, ...style }
  if (shape === 'diamond') { const d = r * 1.3; return <path d={`M${cx},${cy - d}L${cx + d},${cy}L${cx},${cy + d}L${cx - d},${cy}Z`} style={st} /> }
  if (shape === 'square') return <rect x={cx - r * 0.95} y={cy - r * 0.95} width={r * 1.9} height={r * 1.9} rx={1} style={st} />
  if (shape === 'triangle') { const d = r * 1.25; return <path d={`M${cx},${cy - d}L${cx + d},${cy + d * 0.8}L${cx - d},${cy + d * 0.8}Z`} style={st} /> }
  return <circle cx={cx} cy={cy} r={r} style={st} />
}
export function Glyph({ shape, color, size = 12, dash }: { shape: GlyphShape | 'line'; color: string; size?: number; dash?: string }) {
  if (shape === 'line') return <svg width={22} height={size} aria-hidden className="glyph"><line x1={1} x2={21} y1={size / 2} y2={size / 2} style={{ stroke: color }} strokeWidth={2} strokeDasharray={dash} /></svg>
  return <svg width={size} height={size} aria-hidden className="glyph"><GlyphMark shape={shape} cx={size / 2} cy={size / 2} r={size / 2 - 2} color={color} ring={false} /></svg>
}
export function LegendItem({ children, glyph }: { children: ReactNode; glyph: ReactNode }) {
  return <span className="lg-i">{glyph}<span>{children}</span></span>
}

export function Tile({ k, v, s, tone, info }: { k: ReactNode; v: ReactNode; s?: ReactNode; tone?: 'bad' | 'ok' | 'accent'; info?: ReactNode }) {
  return <div className={`tile ${tone ?? ''}`}><div className="k">{k}{info && <Info>{info}</Info>}</div><div className="v">{v}</div>{s && <div className="s">{s}</div>}</div>
}

export function PageHead({ title, children, extra }: { title: string; children?: ReactNode; extra?: ReactNode }) {
  return <div className="pagehead"><h1>{title}</h1>{children && <p>{children}</p>}{extra}</div>
}

export function SimTag() { return <span className="simtag" title="Generated inside the simulation; not real news">Simulated</span> }
