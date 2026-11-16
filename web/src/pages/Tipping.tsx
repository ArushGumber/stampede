import { useEffect, useMemo, useState } from 'react'
import { scaleLinear, scaleLog } from 'd3-scale'
import { line as d3line, area as d3area } from 'd3-shape'
import * as D from '../data'
import type { Cell } from '../types'
import { HCODE, HCODE_ORDER, hcodeShort, regimeLabel, sgLabel } from '../config'
import { useLoaded, useResult, useSource, LoadState, Panel, PageHead, Seg, Sel, Takeaway, GlyphMark, Empty, Info } from '../components/ui'
import { useWidth, TipBox, YGrid, AxisTitle, divergingScale, inkOn, DivLegend, type Tip } from '../components/chart'
import { useResolvedTheme } from '../theme'
import { capLabel, disposition, fin, num, pct, popInfo, popOrder, uniq } from '../vocab'
import { params } from '../modes'

type Measure = { key: string; label: string; unit: string; fmt: (v: number) => string; signed: boolean }
const MEASURES: Measure[] = [
  { key: 'excess_overshoot', label: 'Excess overshoot', unit: '% of price below fundamental value, beyond the no-headline twin', fmt: v => pct(v, 2), signed: true },
  { key: 'excess_overshoot_up', label: 'Excess overshoot above fair value', unit: '% of price above fundamental value, beyond the no-headline twin', fmt: v => pct(v, 2), signed: true },
  { key: 'excess_crash_depth', label: 'Excess crash depth', unit: '% fall from the pre-headline price, beyond the twin', fmt: v => pct(v, 2), signed: true },
  { key: 'excess_n_liquidations', label: 'Extra liquidations', unit: 'forced liquidations per episode, beyond the twin', fmt: v => num(v, 1), signed: true },
  { key: 'p_crash_5pct', label: 'Chance of a 5% crash', unit: '% of seeds whose price fell more than 5%', fmt: v => pct(v, 0), signed: false },
]
const sgOf = (c: Cell) => String(c.safeguard_name ?? c.safeguard ?? 'none')
const val = (c: Cell, k: string) => (fin(c[k]) ? (c[k] as number) : NaN)

export default function Tipping() {
  const tp = useResult<Cell[]>('tipping_point')
  const src = useSource()
  const idx = useLoaded(() => D.episodes().catch(() => []), [])
  const have = useMemo(() => new Set((idx.state === 'ok' ? idx.data : []).map(e => e.id)), [idx])
  const meta = src.state === 'ok' ? src.data.meta : {}
  const theme = useResolvedTheme()
  const rows = useMemo(() => (tp.state === 'ok' && Array.isArray(tp.data) ? tp.data : []), [tp])
  const tags = uniq(rows.map(r => r.tag ?? 'sweep'))
  const q = params()
  const TAG_PREF = ['models', 'reckless_share', 'risk_dial']
  const [tag, setTag] = useState(q.get('tag') ?? ''); const tg = tags.includes(tag) ? tag : TAG_PREF.find(t => tags.includes(t)) ?? tags[tags.length - 1]
  const inTag = rows.filter(r => (r.tag ?? 'sweep') === tg)
  const regimes = uniq(inTag.map(r => r.regime)).sort()
  const [regime, setRegime] = useState(q.get('regime') ?? ''); const rg = regimes.includes(regime) ? regime : regimes.includes('crypto') ? 'crypto' : regimes[0]
  const hcodes = uniq(inTag.map(r => r.hcode)).sort((a, b) => HCODE_ORDER.indexOf(a) - HCODE_ORDER.indexOf(b))
  const [hcode, setHcode] = useState(q.get('hcode') ?? ''); const hc = hcodes.includes(hcode) ? hcode : hcodes[0]
  const sgs = uniq(inTag.map(sgOf))
  const [sg, setSg] = useState('none'); const sgv = sgs.includes(sg) ? sg : sgs[0]
  const measures = MEASURES.filter(m => inTag.some(r => fin(r[m.key])))
  const [mk, setMk] = useState(q.get('measure') ?? '')
  // A false bullish headline pushes the price above fair value, so its default measure is the upward overshoot.
  const defKey = hc === 'fake_unv_bull' ? 'excess_overshoot_up' : 'excess_overshoot'
  const ms = measures.find(m => m.key === (mk || defKey)) ?? measures.find(m => m.key === defKey) ?? measures[0]
  const cells = inTag.filter(r => r.regime === rg && r.hcode === hc && sgOf(r) === sgv)
  const pops = uniq(cells.map(r => r.pop)).sort((a, b) => popOrder(a) - popOrder(b) || a.localeCompare(b))
  const caps = uniq(cells.map(r => r.capital)).sort((a, b) => a - b)
  const [hover, setHover] = useState<Cell | null>(null)
  useEffect(() => setHover(null), [tg, rg, hc, sgv])

  if (tp.state !== 'ok') return <div><Head /><LoadState l={tp} label="tipping-point sweeps" /></div>
  if (!rows.length) return <div><Head /><Empty title="No sweep cells yet" what="results/tipping_point.json is empty" /></div>

  const lo = (c: Cell) => val(c, `${ms.key}_lo`), hi = (c: Cell) => val(c, `${ms.key}_hi`)
  const vs = cells.map(c => val(c, ms.key)).filter(fin)
  const maxAbs = Math.max(1e-9, ...vs.map(Math.abs))
  const top = cells.filter(c => fin(val(c, ms.key))).sort((a, b) => val(b, ms.key) - val(a, ms.key))[0]
  const minCap = caps[0]
  const atMin = cells.filter(c => c.capital === minCap).map(c => val(c, ms.key)).filter(fin)
  const clears = pops.map(p => ({ p, c: cells.filter(c => c.pop === p && fin(lo(c)) && lo(c) > 0).sort((a, b) => a.capital - b.capital)[0] })).filter(z => z.c)
  const firstClear = clears.sort((a, b) => a.c!.capital - b.c!.capital)[0]
  const label = (p: string) => popInfo(p, meta.populations).label
  const take = top ? `${ms.label} is largest for ${label(top.pop)} at ${capLabel(top.capital)} book depth: ${ms.fmt(val(top, ms.key))}${fin(lo(top)) ? ` [${ms.fmt(lo(top))}, ${ms.fmt(hi(top))}]` : ''}.`
    + (atMin.length ? ` At ${capLabel(minCap)} no population exceeds ${ms.fmt(Math.max(...atMin))}.` : '')
    + (ms.signed && firstClear ? ` The lowest capital where a population's interval lies wholly above zero is ${capLabel(firstClear.c!.capital)} (${label(firstClear.p)}).` : ms.signed && fin(lo(top)) ? ' No interval lies wholly above zero.' : '')
    : 'No values in this slice.'

  return (
    <div>
      <Head />
      <div className="filters">
        {tags.length > 1 && <Sel label="Sweep" value={tg} options={tags.map(t => ({ v: t, label: t }))} onChange={setTag} />}
        <Seg label="Regime" value={rg} options={regimes.map(v => ({ v, label: regimeLabel(v) }))} onChange={setRegime} />
        <Seg label="Headline" value={hc} options={hcodes.map(v => ({ v, label: hcodeShort(v), title: HCODE[v]?.long }))} onChange={v => { setHcode(v); setMk('') }} />
        {sgs.length > 1 && <Sel label="Safeguard" value={sgv} options={sgs.map(s => ({ v: s, label: sgLabel(s) }))} onChange={setSg} />}
        <Sel label="Measure" value={ms.key} options={measures.map(m => ({ v: m.key, label: m.label }))} onChange={setMk} />
      </div>
      <p className="note" style={{ marginTop: -8, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 5 }}>{pops.length} populations, {caps.length} capital levels, {cells[0]?.n ?? '?'} seeds per cell.<Info align="left">Sweep {tg}. {HCODE[hc]?.long ?? ''}. Excess means the headline episode minus its twin run with the same seed and no headline.</Info></p>
      {!cells.length ? <Empty title="No cells match these switches" /> : <>
        <Takeaway>{take}</Takeaway>
        <div className="grid21">
          <Panel title={`${ms.label} by population and capital`} sub={ms.unit}>
            <HeatMap cells={cells} pops={pops} caps={caps} ms={ms} maxAbs={maxAbs} theme={theme} have={have} label={label} onHover={setHover} hover={hover} />
            <DivLegend maxAbs={maxAbs} theme={theme} fmt={ms.fmt} label={ms.signed ? 'below the twin to above it' : ''} />
            <p className="note" style={{ marginTop: 8 }}>Cells with a corner mark have a recorded run. Click one to replay it in the Arena.</p>
          </Panel>
          <Panel title="Cell detail" sub={(hover ?? top) ? `${label((hover ?? top)!.pop)}, ${capLabel((hover ?? top)!.capital)}${hover ? '' : ' (largest; hover another cell)'}` : 'Hover, focus or tap a cell'}>
            {(hover ?? top) ? <CellDetail c={(hover ?? top)!} have={have} /> : <p className="note">Each cell is {cells[0]?.n ?? 'several'} seeded episodes of 24 agents.</p>}
          </Panel>
        </div>
        <div style={{ marginTop: 12 }}>
          <Panel title={`${ms.label} against agent capital`} sub="One line per population; bands are intervals across seeds">
            <Lines cells={cells} pops={pops} caps={caps} ms={ms} label={label} />
          </Panel>
        </div>
        <details><summary>Table view of this slice</summary><div className="tablewrap"><table className="t"><thead><tr><th>population</th><th>capital</th><th>n</th>{measures.map(m => <th key={m.key}>{m.label}</th>)}<th>crash depth</th><th>recovery (min)</th><th>replays</th></tr></thead><tbody>
          {[...cells].sort((a, b) => popOrder(a.pop) - popOrder(b.pop) || a.capital - b.capital).map((c, i) => <tr key={i}><td>{label(c.pop)}</td><td>{capLabel(c.capital)}</td><td>{c.n}</td>{measures.map(m => <td key={m.key}>{fin(c[m.key]) ? m.fmt(c[m.key] as number) : '–'}{fin(c[`${m.key}_lo`]) ? ` [${m.fmt(c[`${m.key}_lo`] as number)}, ${m.fmt(c[`${m.key}_hi`] as number)}]` : ''}</td>)}<td>{pct(c.crash_depth, 2)}</td><td>{fin(c.recovery_ticks) ? num(c.recovery_ticks, 1) : '–'}</td><td>{(c.episode_ids ?? []).filter(id => have.has(id)).length}</td></tr>)}</tbody></table></div></details>
      </>}
    </div>
  )
}

function Head() {
  return <PageHead title="Tipping point">How far does the price overshoot after one shared headline as AI agents manage more of the market? Each cell is 24 agents of one kind at one capital level, compared with the same run without the headline.</PageHead>
}

function HeatMap({ cells, pops, caps, ms, maxAbs, theme, have, label, onHover, hover }: { cells: Cell[]; pops: string[]; caps: number[]; ms: Measure; maxAbs: number; theme: 'light' | 'dark'; have: Set<string>; label: (p: string) => string; onHover: (c: Cell | null) => void; hover: Cell | null }) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const narrow = w < 480
  const ml = narrow ? 104 : 190, mt = 4, mb = 40, mr = 4
  const cw = (w - ml - mr) / Math.max(caps.length, 1), ch = narrow ? 34 : 38
  const H = mt + ch * pops.length + mb
  const col = divergingScale(maxAbs, theme)
  const open = (c: Cell) => { const id = (c.episode_ids ?? []).find(e => have.has(e)); if (id) location.hash = `#/arena/${encodeURIComponent(id)}` }
  return (
    <div className="chartbox" ref={ref}>
      {w > 0 && <svg width={w} height={H} role="grid" aria-label={`Heat map of ${ms.label}`}>
        {pops.map((p, j) => { const d = disposition(p); return <g key={p} transform={`translate(0,${mt + ch * j + ch / 2})`}>
          <GlyphMark shape={d.glyph} cx={8} cy={0} r={4.2} color={d.color} ring={false} />
          <text x={18} dy="0.32em" fontSize={narrow ? 11 : 12} fill="var(--ink)">{narrow ? d.short : d.mid}</text>
        </g> })}
        {caps.map((c, i) => <text key={c} x={ml + cw * i + cw / 2} y={mt + ch * pops.length + 15} textAnchor="middle" fontSize={11} fill="var(--muted)">{capLabel(c)}</text>)}
        <text className="axis-title" x={ml + (w - ml - mr) / 2} y={H - 6} textAnchor="middle">{narrow ? 'agent capital (x book depth)' : 'agent capital (multiple of book depth within 1% of mid)'}</text>
        {cells.map(c => {
          const i = caps.indexOf(c.capital), j = pops.indexOf(c.pop); if (i < 0 || j < 0) return null
          const v = val(c, ms.key), X = ml + cw * i, Y = mt + ch * j
          const fill = fin(v) ? col(v) : 'var(--surface2)'
          const rep = (c.episode_ids ?? []).some(e => have.has(e))
          return <g key={`${c.pop}-${c.capital}`} className={`hm-cell ${rep ? '' : 'noep'}`} tabIndex={0} role="gridcell" data-cell={`${c.pop}|${c.capital}`}
            aria-label={`${label(c.pop)}, capital ${capLabel(c.capital)}: ${fin(v) ? ms.fmt(v) : 'no value'}${rep ? '. Press Enter to replay.' : ''}`}
            onPointerEnter={() => onHover(c)} onFocus={() => onHover(c)} onClick={() => { onHover(c); open(c) }} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(c) } }}>
            <rect className="hm-r" x={X + 1} y={Y + 1} width={cw - 2} height={ch - 2} rx={3} style={{ fill }} stroke={hover === c ? 'var(--ink)' : 'none'} strokeWidth={2} />
            {cw > 32 && fin(v) && <text x={X + cw / 2} y={Y + ch / 2} dy="0.32em" textAnchor="middle" fontSize={cw > 52 ? 11.5 : 10} style={{ fill: inkOn(fill), pointerEvents: 'none' }}>{cw > 52 ? ms.fmt(v) : ms.fmt(v).replace('%', '')}</text>}
            {rep && <path d={`M${X + cw - 9},${Y + 3}h6v6z`} style={{ fill: inkOn(fill), opacity: 0.55, pointerEvents: 'none' }} />}
          </g>
        })}
      </svg>}
    </div>
  )
}

function CellDetail({ c, have }: { c: Cell; have: Set<string> }) {
  const reps = (c.episode_ids ?? []).filter(id => have.has(id))
  const NAMES: Record<string, string> = {
    excess_overshoot: 'Excess overshoot', excess_crash_depth: 'Excess crash depth', crash_depth: 'Crash depth', overshoot: 'Overshoot past fundamental',
    excess_n_liquidations: 'Extra liquidations', n_liquidations: 'Liquidations per episode', recovery_ticks: 'Recovery (engine metric)',
    p_crash_5pct: 'Chance of a 5% crash', excess_action_correlation: 'Extra action correlation',
  }
  // Intervals are shown only for excess measures; plain crash_depth_lo/hi in the file are not intervals of crash_depth.
  const row = (k: string, f: (v: number) => string) => fin(c[k]) ? <><dt>{NAMES[k] ?? k.replace(/_/g, ' ')}</dt><dd>{f(c[k] as number)}{k.startsWith('excess_') && fin(c[`${k}_lo`]) ? ` [${f(c[`${k}_lo`] as number)}, ${f(c[`${k}_hi`] as number)}]` : ''}</dd></> : null
  return (
    <div>
      <dl className="kv">
        {row('excess_overshoot', v => pct(v, 2))}{row('excess_crash_depth', v => pct(v, 2))}{row('crash_depth', v => pct(v, 2))}{row('overshoot', v => pct(v, 2))}
        {row('excess_n_liquidations', v => num(v, 1))}{row('n_liquidations', v => num(v, 1))}{row('recovery_ticks', v => `${num(v, 1)} min`)}{row('p_crash_5pct', v => pct(v, 0))}
        {row('excess_action_correlation', v => num(v, 2))}
        <dt>seeds</dt><dd>{c.n}</dd>
      </dl>
      <div style={{ display: 'flex', gap: 6, marginTop: 10, flexWrap: 'wrap' }}>
        {reps.map((id, i) => <a key={id} className="btn ghost sm" href={`#/arena/${encodeURIComponent(id)}`}>Replay run {i + 1}</a>)}
      </div>
      {!reps.length && <p className="note" style={{ marginTop: 8 }}>{c.episode_ids?.length ? `${c.episode_ids.length} episodes were recorded for this cell, but their traces are not exported to this server.` : 'No episode ids listed for this cell.'}</p>}
    </div>
  )
}

function Lines({ cells, pops, caps, ms, label }: { cells: Cell[]; pops: string[]; caps: number[]; ms: Measure; label: (p: string) => string }) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const [tip, setTip] = useState<Tip | null>(null)
  const narrow = w < 560
  const H = 300, m = { l: 52, r: narrow ? 12 : 176, t: 10, b: 40 }
  const iw = Math.max(10, w - m.l - m.r), ih = H - m.t - m.b
  const x = scaleLog().base(2).domain([caps[0] ?? 0.25, caps[caps.length - 1] ?? 5]).range([0, iw])
  const all = cells.flatMap(c => [val(c, ms.key), val(c, `${ms.key}_lo`), val(c, `${ms.key}_hi`)]).filter(fin)
  const y = scaleLinear().domain([Math.min(0, ...all), Math.max(1e-9, ...all)]).range([ih, 0]).nice(5)
  const series = pops.map(p => ({ p, d: disposition(p), pts: cells.filter(c => c.pop === p && fin(val(c, ms.key))).sort((a, b) => a.capital - b.capital) }))
  const ln = d3line<Cell>().x(c => x(c.capital)).y(c => y(val(c, ms.key)))
  const ar = d3area<Cell>().x(c => x(c.capital)).y0(c => y(fin(val(c, `${ms.key}_lo`)) ? val(c, `${ms.key}_lo`) : val(c, ms.key))).y1(c => y(fin(val(c, `${ms.key}_hi`)) ? val(c, `${ms.key}_hi`) : val(c, ms.key)))
  // End labels, pushed apart so they never overlap.
  const ends = series.filter(s => s.pts.length).map(s => ({ s, y: y(val(s.pts[s.pts.length - 1], ms.key)) })).sort((a, b) => a.y - b.y)
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13
  const tickFmt = (v: number) => ms.key.startsWith('excess_n') ? num(v, 1) : `${+(v * 100).toFixed(2)}`
  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const cx = x.invert(e.clientX - r.left)
    const cap = caps.reduce((a, b) => (Math.abs(Math.log2(b / cx)) < Math.abs(Math.log2(a / cx)) ? b : a), caps[0])
    const at = series.map(s => ({ s, c: s.pts.find(c => c.capital === cap) })).filter(z => z.c).sort((a, b) => val(b.c!, ms.key) - val(a.c!, ms.key))
    setTip({ x: m.l + x(cap), y: m.t, body: <><b>capital {capLabel(cap)}</b>{at.map(z => <div key={z.s.p}><svg width={10} height={10} aria-hidden><GlyphMark shape={z.s.d.glyph} cx={5} cy={5} r={3.5} color={z.s.d.color} ring={false} /></svg> {z.s.d.mid}: <b>{ms.fmt(val(z.c!, ms.key))}</b></div>)}</> })
  }
  const slopes = series.filter(z => z.pts.length >= 2).map(z => ({ z, a: val(z.pts[0], ms.key), b: val(z.pts[z.pts.length - 1], ms.key) }))
  const rising = slopes.filter(q => q.b > q.a)
  const steep = [...slopes].sort((p, q) => (q.b - q.a) - (p.b - p.a))[0]
  const fall = slopes.filter(q => q.b < q.a)
  const take = slopes.length ? `From ${capLabel(caps[0])} to ${capLabel(caps[caps.length - 1])} capital, ${ms.label.toLowerCase()} rises for ${rising.length} of ${slopes.length} populations${steep && steep.b > steep.a ? `, most for ${steep.z.d.mid} (${ms.fmt(steep.a)} to ${ms.fmt(steep.b)})` : ''}${fall.length ? `; it ends lower for ${fall.map(q => q.z.d.mid).join(', ')}` : ''}.` : ''
  return (<>
    {take && <Takeaway>{take}</Takeaway>}
    <div className="chartbox" ref={ref}>
      {w > 0 && <svg width={w} height={H} role="img" aria-label={`${ms.label} against capital, one line per population`}>
        <g transform={`translate(${m.l},${m.t})`}>
          <YGrid y={y} width={iw} ticks={5} fmt={tickFmt} zero />
          <g className="axis">{caps.map(c => <g key={c}><line x1={x(c)} x2={x(c)} y1={ih} y2={ih + 4} className="tick" /><text x={x(c)} y={ih + 15} textAnchor="middle">{capLabel(c)}</text></g>)}</g>
          <AxisTitle x={iw / 2} y={ih + 32}>agent capital (multiple of book depth, log scale)</AxisTitle>
          <AxisTitle x={-40} y={ih / 2} rotate>{ms.key.startsWith('excess_n') ? 'liquidations' : `${ms.label.toLowerCase()} (%)`}</AxisTitle>
          {series.map(s => <path key={'b' + s.p} d={ar(s.pts) ?? ''} className="band" style={{ fill: s.d.color }} />)}
          {series.map(s => <g key={s.p}>
            <path d={ln(s.pts) ?? ''} fill="none" style={{ stroke: s.d.color }} strokeWidth={2} />
            {s.pts.map(c => <GlyphMark key={c.capital} shape={s.d.glyph} cx={x(c.capital)} cy={y(val(c, ms.key))} r={4} color={s.d.color} />)}
          </g>)}
          {!narrow && ends.map(e => <g key={'l' + e.s.p}><GlyphMark shape={e.s.d.glyph} cx={iw + 12} cy={e.y} r={3.8} color={e.s.d.color} ring={false} /><text x={iw + 20} y={e.y} dy="0.32em" fontSize={11.5} fill="var(--ink)">{e.s.d.mid}</text></g>)}
          <rect x={0} y={0} width={iw} height={ih} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setTip(null)} />
        </g>
      </svg>}
      {narrow && <div className="legend">{series.map(s => <span key={s.p} className="lg-i"><svg width={12} height={12} aria-hidden><GlyphMark shape={s.d.glyph} cx={6} cy={6} r={4} color={s.d.color} ring={false} /></svg>{s.d.mid}</span>)}</div>}
      <TipBox tip={tip} w={w} />
    </div>
  </>)
}
