import { useMemo, useState } from 'react'
import { scaleLinear } from 'd3-scale'
import type { EarlyRow, SgRow } from '../types'
import { regimeLabel, sgLabel } from '../config'
import { useResult, useSource, LoadState, Panel, PageHead, Seg, Takeaway, Empty, GlyphMark, LegendItem, Glyph } from '../components/ui'
import { useWidth, TipBox, AxisTitle, type Tip } from '../components/chart'
import { capLabel, fin, num, pct, popInfo, popOrder, uniq } from '../vocab'

const SG_ORDER = ['none', 'hide_unverified', 'delay_unverified', 'provenance_filter', 'circuit_breaker', 'order_cap', 'leverage_cap', 'approval_delay', 'cooldown', 'stop_loss', 'account_stop_loss', 'steer_cautious', 'steer_sceptical']
const sgRank = (s: string) => { const i = SG_ORDER.indexOf(s); return i < 0 ? 99 : i }
const isProv = (s: string) => /unverified|provenance/.test(s)
// Short row labels for the chart; full names stay in the table and tooltips.
const SG_SHORT: Record<string, string> = {
  none: 'No safeguard', hide_unverified: 'Hide unverified posts', delay_unverified: 'Delay unverified posts', steer_cautious: 'Steer all cautious',
  steer_sceptical: 'Steer all sceptical', account_stop_loss: 'Account stop-loss', stop_loss: 'Account stop-loss', cooldown: 'Rebalance cooldown',
}
const short = (s: string) => SG_SHORT[s] ?? sgLabel(s)
const lc = (t: string) => t.charAt(0).toLowerCase() + t.slice(1)
type H = 'fake_unv' | 'fake_ver'
const HL: Record<H, { label: string; color: string; glyph: 'circle' | 'diamond' }> = {
  fake_unv: { label: 'False headline, unverified social post', color: 'var(--h-unv)', glyph: 'circle' },
  fake_ver: { label: 'False headline, verified-looking source', color: 'var(--h-ver)', glyph: 'diamond' },
}
const ov = (r: SgRow | undefined, h: H) => (r && fin(r[`${h}_overshoot`]) ? (r[`${h}_overshoot`] as number) : NaN)
const lo = (r: SgRow, h: H) => (fin(r[`${h}_lo`]) ? (r[`${h}_lo`] as number) : NaN)
const hi = (r: SgRow, h: H) => (fin(r[`${h}_hi`]) ? (r[`${h}_hi`] as number) : NaN)
/** Significantly different from the unprotected run: intervals do not overlap. */
const worse = (r: SgRow, base: SgRow, h: H) => fin(lo(r, h)) && fin(hi(base, h)) && lo(r, h) > hi(base, h)
const better = (r: SgRow, base: SgRow, h: H) => fin(hi(r, h)) && fin(lo(base, h)) && hi(r, h) < lo(base, h)
const pm = (v: number) => `${v >= 0 ? '' : '−'}${Math.abs(v * 100).toFixed(1)}`

function Dumbbell({ rows, base, domain, label }: { rows: SgRow[]; base: SgRow; domain: [number, number]; label: string }) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const [tip, setTip] = useState<Tip | null>(null)
  const narrow = w < 520
  const ml = narrow ? 118 : 176, costW = narrow ? 92 : 132, mr = 6, rh = 24, mt = 22
  const pw = Math.max(60, w - ml - costW - mr)
  const x = scaleLinear().domain(domain).range([0, pw]).nice()
  const H = mt + rows.length * rh + 34
  const td0 = base.true_dislocation
  const ticks = x.ticks(narrow ? 3 : 5)
  return (
    <div className="chartbox" ref={ref}>
      {w > 0 && <svg width={w} height={H} role="img" aria-label={`Excess overshoot under each safeguard, ${label}`}>
        <g transform={`translate(${ml},${mt})`}>
          {ticks.map(v => <g key={v} className="axis"><line x1={x(v)} x2={x(v)} y1={0} y2={rows.length * rh} className={Math.abs(v) < 1e-12 ? 'zero' : 'grid'} /><text x={x(v)} y={rows.length * rh + 13} textAnchor="middle">{+(v * 100).toFixed(2)}</text></g>)}
          <AxisTitle x={pw / 2} y={rows.length * rh + 28}>excess overshoot (% below fair value)</AxisTitle>
          {(['fake_unv', 'fake_ver'] as H[]).map(h => fin(ov(base, h)) && <line key={h} x1={x(ov(base, h))} x2={x(ov(base, h))} y1={-4} y2={rows.length * rh} style={{ stroke: HL[h].color }} strokeDasharray="3 3" opacity={0.8} />)}
          <text x={pw + 8} y={-8} fontSize={10.5} fill="var(--muted)">{narrow ? 'true news' : 'true-news error'}</text>
          <text x={w - ml - mr} y={-8} fontSize={10.5} fill="var(--muted)" textAnchor="end">volume</text>
          {rows.map((r, i) => {
            const cy = i * rh + rh / 2, isBase = r.safeguard === 'none'
            const bad = !isBase && (worse(r, base, 'fake_unv') || worse(r, base, 'fake_ver'))
            const good = !isBase && !bad && (better(r, base, 'fake_unv') || better(r, base, 'fake_ver'))
            // Negative dislocation (closer to fair value than the twin) is shown as 0x.
            const tdr = fin(r.true_dislocation) && fin(td0) && td0 > 0 ? Math.max(0, r.true_dislocation as number) / td0 : NaN
            const vr = fin(r.volume_ratio) ? r.volume_ratio : NaN
            const pts = (['fake_unv', 'fake_ver'] as H[]).map(h => ({ h, v: ov(r, h), dy: h === 'fake_unv' ? -4 : 4 })).filter(p => fin(p.v))
            return <g key={r.safeguard} onPointerMove={() => setTip({ x: ml + x(Math.max(...pts.map(p => p.v))), y: mt + cy - 6, body: <><b>{sgLabel(r.safeguard)}</b>{pts.map(p => <div key={p.h}>{HL[p.h].label.replace('False headline, ', '')}: <b>{pct(p.v, 2)}</b> [{pct(lo(r, p.h), 2)}, {pct(hi(r, p.h), 2)}]{!isBase ? ` (none: ${pct(ov(base, p.h), 2)})` : ''}</div>)}<div>true-news mispricing {fin(tdr) ? `${num(tdr, 2)} x unprotected` : 'n/a'}; volume kept {fin(vr) ? pct(vr, 0) : 'n/a'}</div></> })} onPointerLeave={() => setTip(null)}>
              <rect x={-ml} y={cy - rh / 2} width={w} height={rh} fill={isBase ? 'var(--surface2)' : 'transparent'} />
              <text x={-8} y={cy} dy="0.32em" textAnchor="end" fontSize={narrow ? 11 : 12} fontWeight={isBase ? 600 : 400} style={{ fill: bad ? 'var(--bad)' : 'var(--ink)' }}>{short(r.safeguard) + (bad ? (narrow ? ' *' : ' (worse)') : '')}</text>
              {pts.length === 2 && <line x1={x(pts[0].v)} x2={x(pts[1].v)} y1={cy} y2={cy} stroke="var(--axis)" strokeWidth={2} />}
              {pts.map(p => <g key={p.h}>
                {fin(lo(r, p.h)) && <line x1={x(lo(r, p.h))} x2={x(hi(r, p.h))} y1={cy + p.dy} y2={cy + p.dy} style={{ stroke: HL[p.h].color }} strokeWidth={1.3} opacity={0.75} />}
                <GlyphMark shape={HL[p.h].glyph} cx={x(p.v)} cy={cy + p.dy * 0.25} r={4.6} color={HL[p.h].color} />
              </g>)}
              <text x={pw + 8} y={cy} dy="0.32em" fontSize={11.5} className="mono" style={{ fill: fin(tdr) && tdr > 1.25 ? 'var(--warn)' : 'var(--ink2)' }}>{isBase ? '1.00x' : fin(tdr) ? `${num(tdr, 2)}x` : '–'}</text>
              <text x={w - ml - mr} y={cy} dy="0.32em" textAnchor="end" fontSize={11.5} className="mono" style={{ fill: fin(vr) && vr < 0.8 ? 'var(--warn)' : 'var(--ink2)' }}>{fin(vr) ? pct(vr, 0) : '–'}</text>
              {good && <title>significantly lower than no safeguard</title>}
            </g>
          })}
        </g>
      </svg>}
      <TipBox tip={tip} w={w} />
    </div>
  )
}

function findings(rows: SgRow[], cap: number, label: (p: string) => string): string[] {
  const at = rows.filter(r => r.capital === cap)
  const groups = uniq(at.map(r => `${r.pop}|${r.regime}`))
  const out: string[] = []
  // Provenance filters: share of the unprotected overshoot removed, per headline type, where there was a crash to remove.
  const share = (pred: (s: string) => boolean, h: H) => {
    const v: number[] = []
    for (const g of groups) {
      const [pop, regime] = g.split('|')
      const base = at.find(r => r.pop === pop && r.regime === regime && r.safeguard === 'none')
      if (!base || !(ov(base, h) > 0.002)) continue
      for (const r of at.filter(x => x.pop === pop && x.regime === regime && pred(x.safeguard))) if (fin(ov(r, h))) v.push(1 - ov(r, h) / ov(base, h))
    }
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN
  }
  const pu = share(isProv, 'fake_unv'), pv = share(isProv, 'fake_ver')
  if (fin(pu) && fin(pv)) out.push(`Provenance filters remove on average ${Math.round(pu * 100)}% of the overshoot caused by an unverified rumour, but ${Math.round(pv * 100)}% when the same false headline comes from a verified-looking source.`)
  // Safeguards that make a population significantly worse.
  for (const pop of uniq(at.map(r => r.pop)).sort((a, b) => popOrder(a) - popOrder(b))) {
    const hits: { sg: string; regime: string; from: number; to: number }[] = []
    for (const regime of uniq(at.map(r => r.regime))) {
      const base = at.find(r => r.pop === pop && r.regime === regime && r.safeguard === 'none'); if (!base) continue
      for (const r of at.filter(x => x.pop === pop && x.regime === regime && x.safeguard !== 'none')) if (worse(r, base, 'fake_unv')) hits.push({ sg: r.safeguard, regime, from: ov(base, 'fake_unv'), to: ov(r, 'fake_unv') })
    }
    if (!hits.length) continue
    const bySg = uniq(hits.map(h => h.sg))
    const top = [...hits].sort((a, b) => (b.to - b.from) - (a.to - a.from))[0]
    const names = bySg.map((x, i) => (i ? lc(short(x)) : short(x)))
    out.push(`${names.length > 1 ? `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}` : names[0]} make${bySg.length > 1 ? '' : 's'} the population ${label(pop).replace(/^Everyone runs /, '')} worse: in the ${regimeLabel(top.regime).toLowerCase()} market, ${lc(short(top.sg))} raises its rumour overshoot from ${pct(top.from, 2)} to ${pct(top.to, 2)}.`)
  }
  // Best average reduction against the rumour, with its costs.
  const sgs = uniq(at.map(r => r.safeguard)).filter(s => s !== 'none')
  const red = sgs.map(sg => {
    const v: number[] = [], vol: number[] = []
    for (const g of groups) {
      const [pop, regime] = g.split('|')
      const base = at.find(r => r.pop === pop && r.regime === regime && r.safeguard === 'none'), r = at.find(x => x.pop === pop && x.regime === regime && x.safeguard === sg)
      if (base && r && fin(ov(base, 'fake_unv')) && fin(ov(r, 'fake_unv'))) { v.push(ov(base, 'fake_unv') - ov(r, 'fake_unv')); if (fin(r.volume_ratio)) vol.push(r.volume_ratio) }
    }
    return { sg, d: v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN, vol: vol.length ? vol.reduce((a, b) => a + b, 0) / vol.length : NaN }
  }).filter(z => fin(z.d)).sort((a, b) => b.d - a.d)
  if (red[0]) out.push(`Largest average cut in rumour overshoot: ${lc(short(red[0].sg))}, ${pm(red[0].d)} percentage points, keeping ${pct(red[0].vol, 0)} of normal volume.`)
  return out
}

function EarlyWarning() {
  const ew = useResult<EarlyRow[]>('early_warning')
  const [ref, w] = useWidth<HTMLDivElement>()
  if (ew.state !== 'ok') return <LoadState l={ew} label="early-warning results" expect="Whether the agents' internal states warn of a crash before prices do: lead time and AUROC of the mean trait projection, compared with price and order-flow indicators. Appears when results/early_warning.json is exported." />
  const rows = (Array.isArray(ew.data) ? ew.data : []).filter(r => fin(r.auroc)).sort((a, b) => b.auroc - a.auroc)
  if (!rows.length) return <Empty title="No early-warning rows yet" />
  const ml = Math.min(220, Math.max(120, w * 0.38)), rh = 26, H = rows.length * rh + 40
  const lo0 = Math.min(0.4, Math.floor(Math.min(...rows.map(r => (fin(r.lo) ? (r.lo as number) : r.auroc))) * 10) / 10)
  const x = scaleLinear().domain([lo0, 1]).range([0, Math.max(10, w - ml - 90)]).clamp(true)
  const best = rows[0], chance = rows.filter(r => fin(r.lo) && (r.lo as number) <= 0.5)
  return (
    <div>
      <Takeaway>{`${best.indicator.replace(/_/g, ' ')} warns best (AUROC ${num(best.auroc, 2)}, ${num(best.lead_ticks, 1)} min ahead)${chance.length ? `; ${chance.length} indicator${chance.length > 1 ? 's are' : ' is'} not distinguishable from chance` : ''}.`}</Takeaway>
      <div className="chartbox" ref={ref}>
        {w > 0 && <svg width={w} height={H} role="img" aria-label="AUROC of each early-warning indicator">
          <g transform={`translate(${ml},4)`}>
            {x.ticks(6).map(v => <g key={v} className="axis"><line x1={x(v)} x2={x(v)} y1={0} y2={rows.length * rh} className={v === 0.5 ? 'zero' : 'grid'} /><text x={x(v)} y={rows.length * rh + 13} textAnchor="middle">{v.toFixed(1)}</text></g>)}
            <AxisTitle x={(w - ml - 90) / 2} y={rows.length * rh + 30}>AUROC for predicting a crash (0.5 is chance)</AxisTitle>
            {rows.map((r, i) => <g key={r.indicator} transform={`translate(0,${i * rh + rh / 2})`}>
              <text x={-8} dy="0.32em" textAnchor="end" fontSize={12} fill="var(--ink)">{r.indicator.replace(/_/g, ' ')}</text>
              {fin(r.lo) && fin(r.hi) && <line x1={x(r.lo as number)} x2={x(r.hi as number)} stroke="var(--ink2)" strokeWidth={1.5} />}
              <circle cx={x(r.auroc)} r={5} style={{ fill: 'var(--accent)', stroke: 'var(--surface)', strokeWidth: 2 }} />
              <text x={x(fin(r.hi) ? Math.max(r.hi as number, r.auroc) : r.auroc) + 9} dy="0.32em" fontSize={10.5} fill="var(--muted)">{num(r.lead_ticks, 1)} min ahead</text>
            </g>)}
          </g>
        </svg>}
      </div>
    </div>
  )
}

export default function Safeguards() {
  const sg = useResult<SgRow[]>('safeguards')
  const src = useSource()
  const meta = src.state === 'ok' ? src.data.meta : {}
  const rows = useMemo(() => (sg.state === 'ok' && Array.isArray(sg.data) ? sg.data.filter(r => r && typeof r.safeguard === 'string') : []), [sg])
  const caps = uniq(rows.map(r => r.capital)).filter(fin).sort((a, b) => a - b)
  const [cap, setCap] = useState(NaN)
  const cp = caps.includes(cap) ? cap : caps.includes(2) ? 2 : caps[caps.length - 1]
  const at = rows.filter(r => r.capital === cp)
  const pops = uniq(at.map(r => r.pop)).sort((a, b) => popOrder(a) - popOrder(b) || a.localeCompare(b))
  const regimes = uniq(at.map(r => r.regime)).sort()
  const label = (p: string) => popInfo(p, meta.populations).label
  const shapeOk = rows.some(r => fin(r.fake_unv_overshoot))
  const all = at.flatMap(r => [r.fake_unv_lo, r.fake_unv_hi, r.fake_ver_lo, r.fake_ver_hi, r.fake_unv_overshoot, r.fake_ver_overshoot]).filter(fin) as number[]
  const domain: [number, number] = [Math.min(0, ...all), Math.max(0.001, ...all)]
  const facts = shapeOk ? findings(rows, cp, label) : []
  return (
    <div>
      <PageHead title="Safeguards">Each panel is one population in one market at a fixed amount of agent capital. Every row is a safeguard: the circle is the excess overshoot after a false rumour from an unverified post, the diamond after the same false headline from a verified-looking source. Dashed lines mark the unprotected run. The two columns on the right are costs: how much worse true news gets priced, and how much trading volume is left.</PageHead>
      {sg.state !== 'ok' ? <LoadState l={sg} label="safeguard results" expect="The safeguard sweep has not been exported yet. When results/safeguards.json appears, this page compares excess overshoot with and without each safeguard, and its costs." />
        : !rows.length ? <Empty title="No safeguard rows yet" what="results/safeguards.json is empty" />
        : !shapeOk ? <div className="banner err">results/safeguards.json does not have the expected fields (fake_unv_overshoot, fake_ver_overshoot, true_dislocation, volume_ratio).</div>
        : <>
          <div className="filters">
            <Seg small label="Agent capital, multiple of book depth" value={cp} options={caps.map(v => ({ v, label: capLabel(v) }))} onChange={setCap} />
            <div className="legend" style={{ marginBottom: 10 }}>
              {(['fake_unv', 'fake_ver'] as H[]).map(h => <LegendItem key={h} glyph={<Glyph shape={HL[h].glyph} color={HL[h].color} size={12} />}>{HL[h].label}</LegendItem>)}
              <LegendItem glyph={<Glyph shape="line" color="var(--muted)" dash="3 3" />}>unprotected run</LegendItem>
              <span className="note">True-news error: mispricing after a true headline relative to no safeguard (above 1x is slower pricing; mispricing below the twin shows as 0x). Volume: share of unprotected trading volume. Rows marked worse: interval lies wholly above the unprotected run.</span>
            </div>
          </div>
          {facts.length > 0 && <Panel id="sg-facts" title="What the sweep shows" sub={`capital ${capLabel(cp)}, all populations and markets`}>
            <ul className="facts">{facts.map((f, i) => <li key={i}>{f}</li>)}</ul>
          </Panel>}
          <div className="sg-grid">
            {pops.flatMap(pop => regimes.map(regime => {
              const rs = at.filter(r => r.pop === pop && r.regime === regime).sort((a, b) => sgRank(a.safeguard) - sgRank(b.safeguard))
              const base = rs.find(r => r.safeguard === 'none')
              if (!rs.length) return null
              const others = rs.filter(r => r.safeguard !== 'none')
              const bestU = base ? [...others].sort((a, b) => ov(a, 'fake_unv') - ov(b, 'fake_unv'))[0] : undefined
              const worst = base ? [...others].sort((a, b) => ov(b, 'fake_unv') - ov(a, 'fake_unv'))[0] : undefined
              return <Panel key={pop + regime} id={`sg-${pop}-${regime}`} title={label(pop)} sub={`${regimeLabel(regime)}, capital ${capLabel(cp)}`}>
                {base ? <>
                  <Takeaway>{`Unprotected: ${pct(ov(base, 'fake_unv'), 2)} after the rumour, ${pct(ov(base, 'fake_ver'), 2)} after the verified-looking fake.`}{bestU ? ` Lowest with ${lc(short(bestU.safeguard))} (${pct(ov(bestU, 'fake_unv'), 2)}).` : ''}{worst && worse(worst, base, 'fake_unv') ? ` Highest with ${lc(short(worst.safeguard))} (${pct(ov(worst, 'fake_unv'), 2)}).` : ''}</Takeaway>
                  <Dumbbell rows={rs} base={base} domain={domain} label={`${label(pop)}, ${regimeLabel(regime)}`} />
                </> : <Empty title="No unprotected run in this panel" />}
              </Panel>
            }))}
          </div>
          <details><summary>Table view</summary><div className="tablewrap"><table className="t"><thead><tr><th>population</th><th className="l">market</th><th className="l">safeguard</th><th>rumour (unverified)</th><th>verified-looking fake</th><th>true-news mispricing</th><th>volume kept</th></tr></thead><tbody>
            {[...at].sort((a, b) => popOrder(a.pop) - popOrder(b.pop) || a.regime.localeCompare(b.regime) || sgRank(a.safeguard) - sgRank(b.safeguard)).map((r, i) => <tr key={i} className={r.safeguard === 'none' ? 'hl' : ''}><td>{label(r.pop)}</td><td className="l">{regimeLabel(r.regime)}</td><td className="l">{sgLabel(r.safeguard)}</td>
              <td>{pct(r.fake_unv_overshoot, 2)} [{pct(r.fake_unv_lo, 2)}, {pct(r.fake_unv_hi, 2)}]</td><td>{pct(r.fake_ver_overshoot, 2)} [{pct(r.fake_ver_lo, 2)}, {pct(r.fake_ver_hi, 2)}]</td>
              <td>{num(r.true_dislocation, 2)}{fin(r.true_dislocation_base) ? ` (none ${num(r.true_dislocation_base, 2)})` : ''}</td><td>{fin(r.volume_ratio) ? pct(r.volume_ratio, 0) : '–'}</td></tr>)}</tbody></table></div>
            <p className="note">Overshoot values are excess over the no-headline twin, with bootstrap intervals. True-news mispricing is the dislocation area after a true headline (sum over minutes of the gap to fundamental value as a fraction).</p></details>
        </>}
      <div style={{ marginTop: 14 }}><Panel title="Early warning" sub="do the agents' internal states move before the price does?"><EarlyWarning /></Panel></div>
    </div>
  )
}
