import { useMemo, useState } from 'react'
import { scaleLinear } from 'd3-scale'
import type { DoseRecord, YardRow } from '../types'
import { CLOSED, METRIC, TRAIT, metricLabel, modelLabel } from '../config'
import { useResult, LoadState, Panel, PageHead, Takeaway, GlyphMark, LegendItem, Glyph, Empty } from '../components/ui'
import { useWidth, TipBox, AxisTitle, robustDomain, inverseInterp, spearman, type Tip } from '../components/chart'
import { disposition, fin, num, uniq } from '../vocab'

const PANELS: { metric: string; trait: string }[] = [
  { metric: 'crra_gamma', trait: 'risk' }, { metric: 'kelly_fraction', trait: 'risk' }, { metric: 'credulity', trait: 'credulity' },
  { metric: 'unethical_rate', trait: 'integrity' }, { metric: 'mean_exposure', trait: 'risk' },
]
const REF_MODEL = 'gemma-4-12b'
const fmtM = (m: string, v: number) => (METRIC[m]?.frac ? `${(v * 100).toFixed(1)}%` : num(v, 2))
const sa = (a: number) => (a > 0 ? `+${+a.toFixed(2)}` : a < 0 ? `−${+(-a).toFixed(2)}` : '0')

function agentStyle(r: YardRow) {
  if (CLOSED[r.agent]) { const d = disposition(r.agent); return { color: d.color, glyph: d.glyph, label: d.label, closed: true } }
  return { color: r.agent === REF_MODEL ? 'var(--d-base)' : 'var(--muted)', glyph: (r.agent === REF_MODEL ? 'circle' : 'ring') as 'circle' | 'ring', label: modelLabel(r.agent), closed: false }
}

interface Curve { pts: { x: number; y: number }[]; trait: string; model: string; monotone: boolean }
function curveFor(dose: DoseRecord[], metric: string, trait: string): Curve | null {
  const rec = dose.find(d => d.model === REF_MODEL && d.trait === trait) ?? dose.find(d => d.trait === trait)
  if (!rec) return null
  const st = rec.series.filter(s => s.kind === 'steer' && s.metric === metric && fin(s.value)).sort((a, b) => a.alpha - b.alpha)
  if (st.length < 3) return null
  const rd = robustDomain(st.map(s => s.value), 0.5)
  const pts = st.filter(s => s.value >= rd.lo && s.value <= rd.hi).map(s => ({ x: s.alpha, y: s.value }))
  const rho = spearman(pts.map(p => p.x), pts.map(p => p.y))
  return { pts, trait, model: rec.model, monotone: Math.abs(rho) >= 0.85 }
}

function YardPanel({ metric, trait, rows, curve }: { metric: string; trait: string; rows: YardRow[]; curve: Curve | null }) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const [tip, setTip] = useState<Tip | null>(null)
  const narrow = w < 460
  const ml = narrow ? 112 : 176, mr = 16, rh = 26, mt = 6
  const sorted = [...rows].sort((a, b) => a.value - b.value)
  const rulerH = curve ? 52 : 0
  const H = mt + rh * sorted.length + 40 + rulerH + 6
  const xs = [...sorted.map(r => r.value).filter(fin), ...(curve?.pts.map(p => p.y) ?? [])] as number[]
  const lo0 = Math.min(...xs), hi0 = Math.max(...xs), pad = (hi0 - lo0) * 0.06 || 0.05
  const x = scaleLinear().domain([lo0 - pad, hi0 + pad]).range([0, Math.max(10, w - ml - mr)]).nice()
  const iw = Math.max(10, w - ml - mr)
  const eq = (r: YardRow): { a: number; read: boolean } | null => {
    if (fin(r.alpha_equivalent) && CLOSED[r.agent]) return { a: r.alpha_equivalent, read: false }
    if (!curve || !curve.monotone || !CLOSED[r.agent]) return null
    const a = inverseInterp(curve.pts, r.value)
    return a === null ? null : { a, read: true }
  }
  // Ruler ticks: thin out labels that would collide.
  // Keep alpha = 0 and the extremes labelled first, then add others where they do not collide.
  const pts0 = (curve?.pts ?? []).map(p => ({ ...p, px: x(p.y), show: false }))
  const order = [...pts0].sort((a, b) => (a.x === 0 ? -1 : b.x === 0 ? 1 : Math.abs(b.x) - Math.abs(a.x)))
  const shown: number[] = []
  for (const p of order) if (shown.every(q => Math.abs(q - p.px) >= 26)) { p.show = true; shown.push(p.px) }
  const ruler = pts0
  const span = Math.abs(x.domain()[1] - x.domain()[0])
  const tf = (v: number) => (METRIC[metric]?.frac ? (span * 100 < 10 ? (v * 100).toFixed(1) : String(Math.round(v * 100))) : String(+v.toFixed(2)))
  const yAx = mt + rh * sorted.length + 6
  return (
    <div className="chartbox" ref={ref}>
      {w > 0 && <svg width={w} height={H} role="img" aria-label={`${metricLabel(metric)} for each agent`}>
        <g transform={`translate(${ml},0)`}>
          {x.ticks(Math.max(3, Math.floor(iw / 60))).map(v => <g key={v} className="axis"><line x1={x(v)} x2={x(v)} y1={mt} y2={yAx} className={Math.abs(v) < 1e-12 ? 'zero' : 'grid'} /><text x={x(v)} y={yAx + 13} textAnchor="middle">{tf(v)}</text></g>)}
          <AxisTitle x={iw / 2} y={yAx + 28}>{narrow ? METRIC[metric]?.short ?? metric : METRIC[metric]?.unit ?? metric}</AxisTitle>
          {sorted.map((r, i) => {
            const st = agentStyle(r), cy = mt + i * rh + rh / 2, e = eq(r)
            return <g key={r.agent} onPointerMove={() => setTip({ x: ml + x(r.value), y: cy - 8, body: <><b>{st.label}</b><br />{metricLabel(metric)}: {fmtM(metric, r.value)}{e ? <><br />Gemma {TRAIT[trait]?.label.toLowerCase()} dial equivalent: alpha {sa(e.a)}{!e.read && fin(r.lo) && fin(r.hi) ? ` [${sa(r.lo)}, ${sa(r.hi)}]` : ''}{e.read ? ' (read off the curve on this page)' : ''}</> : null}<br /><span className="muted">{r.harness ?? ''}{r.date ? `, ${r.date}` : ''}</span></> })} onPointerLeave={() => setTip(null)}>
              <rect x={-ml} y={cy - rh / 2} width={w} height={rh} fill="transparent" />
              <text x={-10} y={cy - 4} dy="0.32em" textAnchor="end" fontSize={narrow ? 11.5 : 12} fill="var(--ink)">{st.label}</text>
              <text x={-10} y={cy + 8} dy="0.32em" textAnchor="end" fontSize={10} fill="var(--muted)">{st.closed ? '' : 'unsteered, '}{r.harness ?? ''}{r.date && !narrow ? `, ${r.date}` : ''}</text>
              <GlyphMark shape={st.glyph} cx={x(r.value)} cy={cy} r={5} color={st.color} />
            </g>
          })}
          {curve && <g transform={`translate(0,${yAx + 56})`}>
            <line x1={x(Math.min(...curve.pts.map(p => p.y)))} x2={x(Math.max(...curve.pts.map(p => p.y)))} y1={0} y2={0} stroke="var(--ink)" strokeWidth={2} />
            {ruler.map(p => <g key={p.x}><line x1={p.px} x2={p.px} y1={-4} y2={4} stroke="var(--ink)" strokeWidth={1.5} />{p.show && <text x={p.px} y={-7} textAnchor="middle" fontSize={10} fill="var(--ink2)">{sa(p.x)}</text>}</g>)}
            <text x={0} y={17} fontSize={10.5} fill="var(--muted)">{narrow ? `ruler: ${TRAIT[curve.trait]?.label.toLowerCase()} dial alpha` : `ruler: ${modelLabel(curve.model)} steered to each alpha on the ${TRAIT[curve.trait]?.label.toLowerCase()} dial`}</text>
          </g>}
        </g>
      </svg>}
      <TipBox tip={tip} w={w} />
    </div>
  )
}

function panelTake(metric: string, trait: string, rows: YardRow[], curve: Curve | null): string {
  const closed = rows.filter(r => CLOSED[r.agent]).sort((a, b) => a.value - b.value)
  const open = rows.filter(r => !CLOSED[r.agent])
  if (!closed.length) return 'No closed-model measurements for this metric yet.'
  const f = (v: number) => fmtM(metric, v), L = (r: YardRow) => CLOSED[r.agent]?.label ?? r.agent
  const lowW = metric === 'crra_gamma' ? 'least risk-averse' : metric === 'credulity' ? 'most sceptical' : 'lowest'
  const highW = metric === 'crra_gamma' ? 'most risk-averse' : metric === 'credulity' ? 'most credulous' : 'highest'
  let s = `Among closed models, ${L(closed[closed.length - 1])} is ${highW} (${f(closed[closed.length - 1].value)}) and ${L(closed[0])} ${lowW} (${f(closed[0].value)})`
  const g = open.find(r => r.agent === REF_MODEL)
  if (g) s += `; unsteered ${modelLabel(REF_MODEL)} is at ${f(g.value)}`
  if (curve?.monotone || closed.some(r => fin(r.alpha_equivalent))) {
    const eqs = closed.map(r => (fin(r.alpha_equivalent) ? r.alpha_equivalent : curve?.monotone ? inverseInterp(curve.pts, r.value) : null)).filter(fin) as number[]
    const k = closed.length - eqs.length
    if (eqs.length) s += `. On the ${TRAIT[trait]?.label.toLowerCase()} dial they sit between alpha ${sa(Math.min(...eqs))} and ${sa(Math.max(...eqs))}${k ? ` (${k} ${k > 1 ? 'fall' : 'falls'} outside the steered range)` : ''}`
  }
  return s + '.'
}

export default function Yardstick() {
  const yd = useResult<YardRow[]>('yardstick')
  const dr = useResult<DoseRecord[]>('dose_response')
  const rows = useMemo(() => (yd.state === 'ok' && Array.isArray(yd.data) ? yd.data.filter(r => fin(r.value)) : []), [yd])
  const dose = useMemo(() => (dr.state === 'ok' ? (Array.isArray(dr.data) ? dr.data : [dr.data]) : []), [dr])
  const agents = uniq(rows.map(r => r.agent))
  const closedAgents = Object.keys(CLOSED).filter(a => agents.includes(a))
  const openAgents = agents.filter(a => !CLOSED[a])
  return (
    <div>
      <PageHead title="Yardstick">Closed models cannot be steered, so they are measured on the same batteries and placed on the same economic axes: CRRA risk aversion from lottery choices, bet size relative to Kelly, credulity toward unverified news and the rate of unethical choices. The steered Gemma curve under each axis is the ruler.</PageHead>
      {yd.state !== 'ok' ? <LoadState l={yd} label="yardstick measurements" /> : !rows.length ? <Empty title="No yardstick rows yet" what="results/yardstick.json is empty" /> : <>
        <div className="legend">
          {closedAgents.map(a => { const d = disposition(a); return <LegendItem key={a} glyph={<Glyph shape={d.glyph} color={d.color} size={12} />}>{d.label}</LegendItem> })}
          {openAgents.map(a => <LegendItem key={a} glyph={<Glyph shape={a === REF_MODEL ? 'circle' : 'ring'} color={a === REF_MODEL ? 'var(--d-base)' : 'var(--muted)'} size={12} />}>{modelLabel(a)} unsteered</LegendItem>)}
          <LegendItem glyph={<Glyph shape="line" color="var(--ink)" />}>Gemma dial ruler, ticks labelled with alpha</LegendItem>
        </div>
        <div className="smgrid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(min(100%, 480px), 1fr))' }}>
          {PANELS.map(({ metric, trait }) => {
            const rs = rows.filter(r => r.metric === metric)
            const curve = curveFor(dose, metric, trait)
            return <Panel key={metric} id={`yard-${metric}`} title={METRIC[metric]?.label ?? metric} sub={TRAIT[trait] ? `${TRAIT[trait].label} dial` : undefined}>
              {rs.length ? <>
                <Takeaway>{panelTake(metric, trait, rs, curve)}</Takeaway>
                <YardPanel metric={metric} trait={trait} rows={rs} curve={curve} />
                {!curve && <p className="note">The {modelLabel(REF_MODEL)} {TRAIT[trait]?.label.toLowerCase()} dial for this metric is not exported yet, so no ruler is drawn.</p>}
              </> : <Empty title="Not measured yet" what={`no ${metric} rows in results/yardstick.json`} />}
            </Panel>
          })}
        </div>
        <details><summary>Table view</summary><div className="tablewrap"><table className="t"><thead><tr><th>agent</th><th className="l">vendor</th><th className="l">harness</th><th className="l">date</th>{PANELS.map(p => <th key={p.metric}>{metricLabel(p.metric)}</th>)}</tr></thead><tbody>
          {[...closedAgents, ...openAgents].map(a => { const r0 = rows.find(r => r.agent === a)!; return <tr key={a}><td>{agentStyle(r0).label}</td><td className="l">{r0.vendor}</td><td className="l">{r0.harness ?? '–'}</td><td className="l">{r0.date ?? '–'}</td>
            {PANELS.map(p => { const r = rows.find(x => x.agent === a && x.metric === p.metric); return <td key={p.metric}>{r ? fmtM(p.metric, r.value) : '–'}{r && fin(r.alpha_equivalent) && CLOSED[a] ? ` (alpha ${sa(r.alpha_equivalent)}${fin(r.lo) && fin(r.hi) ? ` [${sa(r.lo)}, ${sa(r.hi)}]` : ''})` : ''}</td> })}</tr> })}
        </tbody></table></div></details>
        <p className="note" style={{ marginTop: 8 }}>Positions describe measured behaviour on these fixed batteries, through the named harness on the named date. They are not a statement about any vendor's safety. Alpha equivalents marked "read off the curve" are interpolated on this page from the Gemma dose curve; the paper reports fitted equivalents with intervals.</p>
      </>}
    </div>
  )
}
