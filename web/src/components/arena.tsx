import { useMemo, useState } from 'react'
import { scaleLinear } from 'd3-scale'
import { line as d3line, area as d3area } from 'd3-shape'
import type { Frame, AgentInfo, NewsItem } from '../types'
import { useWidth, TipBox, YGrid, XTicks, AxisTitle, type Tip } from './chart'
import { Takeaway, SimTag } from './ui'
import { disposition, pct, num, big, priceDigits, type Disposition } from '../vocab'

/* ------------------------------------------------------------------ price and volume */
export function PriceChart({ frames, cursor, nTicks, shockT, corrT, symbol, troughT, p0, height, peak }: {
  frames: Frame[]; cursor: number; nTicks: number; shockT: number | null; corrT: number | null; symbol: string; troughT?: number; p0?: number; height?: number; peak?: boolean
}) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const [tip, setTip] = useState<Tip | null>(null)
  const narrow = w < 520
  const m = { l: narrow ? 46 : 56, r: 12, t: 26, b: 4 }
  const H = height ?? (narrow ? 220 : 270), VH = 54, gap = 22
  const iw = Math.max(10, w - m.l - m.r), ih = H - m.t - m.b
  const vis = frames.filter(f => f.t <= cursor)
  const all = frames
  const lo = all.length ? Math.min(...all.map(f => Math.min(f.price, f.fundamental))) : 99
  const hi = all.length ? Math.max(...all.map(f => Math.max(f.price, f.fundamental))) : 101
  const padv = (hi - lo) * 0.1 || hi * 0.005
  const x = scaleLinear().domain([0, Math.max(nTicks - 1, 1)]).range([0, iw])
  const y = scaleLinear().domain([lo - padv, hi + padv]).range([ih, 0]).nice(5)
  const vmax = Math.max(1e-9, ...all.map(f => f.volume || 0))
  const yv = scaleLinear().domain([0, vmax]).range([VH, 0]).nice(2)
  const ln = (k: 'price' | 'fundamental') => d3line<Frame>().x(f => x(f.t)).y(f => y(f[k]))(vis) ?? ''
  const ar = d3area<Frame>().x(f => x(f.t)).y0(ih).y1(f => y(f.price))(vis) ?? ''
  const cur = vis[vis.length - 1]
  const dg = priceDigits(hi)
  const bw = Math.max(1, iw / Math.max(nTicks, 1) - 1.5)
  // Mark the trough after a bearish headline, or the peak after a bullish one.
  const post = shockT === null ? [] : vis.filter(f => f.t >= shockT)
  const pk = peak && post.length ? post.reduce((a, f) => (f.price > a.price ? f : a)) : undefined
  const tr = pk ?? (troughT !== undefined && troughT <= cursor ? frames.find(f => f.t === troughT) : undefined)
  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const t = Math.round(x.invert(e.clientX - r.left))
    const f = vis.find(fr => fr.t === Math.min(Math.max(t, 0), cursor))
    if (f) setTip({ x: m.l + x(f.t), y: m.t + y(f.price), body: <><b>minute {f.t}</b><br />price {num(f.price, dg)}<br />fundamental {num(f.fundamental, dg)} ({pct((f.price - f.fundamental) / f.fundamental, 2)})<br />volume {big(f.volume)} units{f.liquidations ? <><br /><span className="bad">{f.liquidations} liquidation{f.liquidations > 1 ? 's' : ''}</span></> : null}{f.halted ? <><br />trading halted</> : null}</> })
  }
  const totalH = m.t + ih + gap + VH + 34
  return (
    <div className="chartbox" ref={ref}>
      {w > 0 && <svg width={w} height={totalH} role="img" aria-label={`${symbol} price and fundamental value by simulated minute, with volume`}>
        <defs><linearGradient id="pricefill" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="var(--accent)" stopOpacity="0.10" /><stop offset="1" stopColor="var(--accent)" stopOpacity="0" /></linearGradient></defs>
        <g transform={`translate(${m.l},${m.t})`}>
          <YGrid y={y} width={iw} ticks={5} fmt={v => v.toFixed(dg > 1 ? 1 : 0)} />
          <text className="axis-title" x={-m.l + 4} y={-14}>price ({symbol})</text>
          {vis.filter(f => f.halted).map(f => <rect key={'h' + f.t} x={x(f.t) - bw / 2} y={0} width={bw + 1.5} height={ih} className="halt" />)}
          {shockT !== null && <g className="marker"><line x1={x(shockT)} x2={x(shockT)} y1={-6} y2={ih} className="m-head" /><text x={x(shockT) + 4} y={-4}>headline</text></g>}
          {corrT !== null && corrT <= cursor && <g className="marker"><line x1={x(corrT)} x2={x(corrT)} y1={-6} y2={ih} className="m-corr" /><text x={x(corrT) + 4} y={-4}>correction</text></g>}
          <path d={ar} className="a-price" />
          <path d={ln('fundamental')} className="l-fund" />
          <path d={ln('price')} className="l-price" />
          {tr && p0 && <g className="marker"><circle cx={x(tr.t)} cy={y(tr.price)} r={4} className="m-trough" /><text x={x(tr.t) + (x(tr.t) > iw * 0.8 ? -6 : 6)} y={y(tr.price) + 14} textAnchor={x(tr.t) > iw * 0.8 ? 'end' : 'start'}>{pk ? 'peak +' : 'trough '}{pct((tr.price - p0) / p0, 1).replace(/^\+/, '')}</text></g>}
          {cur && <circle cx={x(cur.t)} cy={y(cur.price)} r={3.5} className="m-cur" />}
          <rect x={0} y={0} width={iw} height={ih + gap + VH} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setTip(null)} />
        </g>
        <g transform={`translate(${m.l},${m.t + ih + gap})`}>
          <YGrid y={yv} width={iw} ticks={2} fmt={v => big(v)} />
          {vis.map(f => <g key={f.t}>
            <rect x={x(f.t) - bw / 2} y={yv(f.volume || 0)} width={bw} height={VH - yv(f.volume || 0)} className="vbar" />
            {f.liquidation_volume > 0 && <rect x={x(f.t) - bw / 2} y={yv(f.liquidation_volume)} width={bw} height={VH - yv(f.liquidation_volume)} className="vliq" />}
          </g>)}
          <XTicks x={x} height={VH} values={x.ticks(Math.max(2, Math.min(10, Math.floor(iw / 55))))} />
          <AxisTitle x={iw} y={VH + 30} anchor="end">simulated minute</AxisTitle>
          <text className="axis-title" x={0} y={-6}>volume (units / min){all.some(f => f.liquidation_volume > 0) ? ', forced sales in red' : ''}</text>
        </g>
      </svg>}
      <TipBox tip={tip} w={w} />
    </div>
  )
}

/* ------------------------------------------------------------------ order book */
export function DepthLadder({ frame, unit }: { frame: Frame | undefined; unit: string }) {
  if (!frame) return <p className="note">No book yet.</p>
  const bids = (frame.bids ?? []).slice(0, 10), asks = (frame.asks ?? []).slice(0, 10)
  const mx = Math.max(1e-9, ...bids.map(b => b[1]), ...asks.map(a => a[1]))
  const dg = priceDigits(frame.price)
  const sb = bids.reduce((s, b) => s + b[1], 0), sa = asks.reduce((s, a) => s + a[1], 0)
  const spread = frame.best_ask !== null && frame.best_bid !== null ? ((frame.best_ask - frame.best_bid) / frame.price) * 1e4 : null
  const gap = (frame.price - frame.fundamental) / frame.fundamental
  const row = (p: number, q: number, side: 'bid' | 'ask') => (
    <div className={`lad-row ${side}`} key={side + p}>
      <span className="px r">{p.toFixed(dg)}</span>
      <span className="lad-bar"><i style={{ width: `${(q / mx) * 100}%` }} /></span>
      <span className="mono r muted">{big(q)}</span>
    </div>
  )
  const lean = sb + sa > 0 ? sb / (sb + sa) : 0.5
  return (
    <div className="ladder">
      <div className="lad-row head"><span className="r">Price</span><span>Resting size</span><span className="r">{unit}</span></div>
      <div className="lad-side">{[...asks].reverse().map(a => row(a[0], a[1], 'ask'))}{!asks.length && <div className="note">No asks</div>}</div>
      <div className="lad-mid"><span>Mid <b>{frame.price.toFixed(dg)}</b></span><span>{frame.halted ? 'Halted' : <>Spread <b>{spread === null ? 'n/a' : `${spread.toFixed(1)} bp`}</b></>}</span></div>
      <div className="lad-side">{bids.map(b => row(b[0], b[1], 'bid'))}{!bids.length && <div className="note">No bids</div>}</div>
      <div className="lad-foot" title={`Top ${Math.max(bids.length, asks.length)} levels. ${lean < 0.4 ? 'Sellers dominate.' : lean > 0.6 ? 'Buyers dominate.' : 'Roughly balanced.'}`}>
        <span>Bid <b>{big(sb)}</b> / Ask <b>{big(sa)}</b></span>
        <span>vs fair value <b>{gap >= 0 ? '+' : '−'}{pct(Math.abs(gap), 2)}</b></span>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ agent grid */
export function AgentGrid({ frames, cursor, agents, shockT, maxLev, compact }: { frames: Frame[]; cursor: number; agents: AgentInfo[]; shockT: number | null; maxLev: number; compact?: boolean }) {
  const [sel, setSel] = useState<string | null>(null)
  const frame = useMemo(() => { let f: Frame | undefined; for (const fr of frames) if (fr.t <= cursor) f = fr; return f }, [frames, cursor])
  const prevMap = useMemo(() => {
    const liqBefore = new Set<string>()
    for (const fr of frames) if (fr.t < cursor) for (const a of fr.agents) if (a.liquidated) liqBefore.add(a.id)
    return liqBefore
  }, [frames, cursor])
  const preShock = useMemo(() => (shockT === null ? undefined : [...frames].reverse().find(f => f.t < shockT)), [frames, shockT])
  const sinceShock = useMemo(() => {
    const lo = new Map<string, number>(), hi = new Map<string, number>()
    if (shockT !== null) for (const fr of frames) if (fr.t >= shockT && fr.t <= cursor) for (const a of fr.agents) {
      lo.set(a.id, Math.min(lo.get(a.id) ?? Infinity, a.exposure)); hi.set(a.id, Math.max(hi.get(a.id) ?? -Infinity, a.exposure))
    }
    return { lo, hi }
  }, [frames, shockT, cursor])
  const byId = useMemo(() => new Map(agents.map(a => [a.id, a])), [agents])
  const disp = useMemo(() => { const m = new Map<string, Disposition>(); for (const a of agents) m.set(a.id, disposition(a.group, a.alphas, a.model)); return m }, [agents])
  const groups = useMemo(() => {
    const g = new Map<string, { d: Disposition; ids: string[] }>()
    for (const a of agents) { const d = disp.get(a.id)!; const k = d.id + '|' + a.group; if (!g.has(k)) g.set(k, { d, ids: [] }); g.get(k)!.ids.push(a.id) }
    return [...g.values()].sort((a, b) => a.d.order - b.d.order)
  }, [agents, disp])
  if (!frame) return <p className="note">No agents yet.</p>
  const fa = new Map(frame.agents.map(a => [a.id, a]))
  const pre = preShock ? new Map(preShock.agents.map(a => [a.id, a.exposure])) : null
  let cut = 0, lev = 0, nAfter = 0, sumPre = 0, sumNow = 0
  if (pre && shockT !== null && frame.t >= shockT) for (const [id, e0] of pre) {
    const e = fa.get(id)?.exposure; if (e === undefined) continue
    nAfter++; sumPre += e0; sumNow += e
    if ((sinceShock.lo.get(id) ?? e) < e0 - 0.05) cut++
    if ((sinceShock.hi.get(id) ?? e) > Math.max(e0, 1) + 0.5) lev++
  }
  const liqNow = frame.agents.filter(a => a.liquidated).length
  const liqEver = new Set([...prevMap, ...frame.agents.filter(a => a.liquidated).map(a => a.id)]).size
  const s = sel ? fa.get(sel) : undefined, si = sel ? byId.get(sel) : undefined
  return (
    <div>
      <div className="agrid-wrap">
        {groups.map(g => (
          <div className="agrp" key={g.d.id + g.ids[0]}>
            <div className="agrp-h"><span className={`sw ${g.d.dashed ? 'dashed' : ''}`} style={{ ['--gc' as string]: g.d.color }} />{g.d.label}<span className="muted"> × {g.ids.length}</span></div>
            <div className="agrid">
              {g.ids.map(id => {
                const a = fa.get(id); if (!a) return <div key={id} className="cell gone" />
                const flash = a.liquidated
                const hatched = prevMap.has(id) || a.liquidated
                const e = Math.sqrt(Math.min(1, Math.abs(a.exposure) / maxLev))
                return <button type="button" key={flash ? `${id}-${frame.t}` : id} className={`cell ${g.d.dashed ? 'dashed' : ''} ${hatched ? 'liq' : ''} ${flash ? 'flash' : ''} ${sel === id ? 'sel' : ''} ${e > 0.45 ? 'hi' : ''}`}
                  style={{ ['--gc' as string]: g.d.color, ['--e' as string]: e }} onMouseEnter={() => setSel(id)} onFocus={() => setSel(id)} onClick={() => setSel(id)}
                  aria-label={`${id}, ${g.d.label}, exposure ${num(a.exposure, 2)}x${hatched ? ', liquidated' : ''}`}>
                  <span className="cv">{Math.abs(a.exposure) >= 9.95 ? Math.round(a.exposure) : a.exposure.toFixed(1)}</span>
                </button>
              })}
            </div>
          </div>
        ))}
      </div>
      <div className="agrid-legend">
        <span className="lg-i">0x<span className="expo-ramp" aria-hidden />{maxLev}x {compact ? 'exposure' : 'exposure (position / equity)'}</span>
        <span className="lg-i"><span className="sw liqsw" />liquidated</span>
        {!compact && <span className="lg-i"><span className="sw dashed" style={{ ['--gc' as string]: 'var(--muted)' }} />closed-model policy</span>}
      </div>
      {!compact && <div className="agent-detail" aria-live="polite">
        {s && si ? <><b>{s.id}</b>, {disp.get(s.id)?.label}: exposure {num(s.exposure, 2)}x, equity {big(s.equity)} ({pct(s.equity / si.cash0 - 1, 2)}){s.action !== null && s.action !== undefined ? `, asked for ${num(s.action, 2)}x` : ''}{s.rejections?.length ? `; rejected: ${s.rejections.map(r => r.reason.replace(/_/g, ' ')).join(', ')}` : ''}{s.reason ? `; stated reason: "${s.reason}"` : ''}</> : 'Hover or tap a cell to see that agent.'}
      </div>}
      <Takeaway>{shockT === null || frame.t < shockT
        ? `Before the headline: mean exposure ${num(mean(frame.agents.map(a => a.exposure)), 2)}x across ${frame.agents.length} agents.`
        : `Since the headline, ${cut} of ${nAfter} agents cut exposure at some point${lev ? ` and ${lev} levered up` : ''}; mean exposure is ${num(sumNow / Math.max(nAfter, 1), 2)}x now against ${num(sumPre / Math.max(nAfter, 1), 2)}x before; ${liqEver} liquidated${liqNow ? `, ${liqNow} this minute` : ''}.`}</Takeaway>
    </div>
  )
}
const mean = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN)

/* ------------------------------------------------------------------ news */
export function NewsFeed({ items, truth, done }: { items: (NewsItem & { t: number })[]; truth?: 'true' | 'false'; done: boolean }) {
  return (
    <div className="feed" aria-live="polite">
      {items.length === 0 && <p className="note">No news yet. The headline arrives at the marked minute.</p>}
      {items.map(n => (
        <article key={n.id + n.t} className={`news ${n.verified ? 'ver' : 'unv'}`}>
          <div className="src"><b className={`vtag ${n.verified ? 'ver' : 'unv'}`}>{n.verified ? 'Verified' : 'Unverified'}</b><span>{n.source}{n.author_agent ? `, posted by ${n.author_agent}` : ''}</span><span className="t">min {n.t}</span></div>
          <p>{n.text} <SimTag /></p>
        </article>
      ))}
      {done && truth && <p className="note truth">{truth === 'false' ? 'This headline was false. The fundamental value never moved.' : 'This headline was true. The fundamental value moved with it.'}</p>}
    </div>
  )
}

/* ------------------------------------------------------------------ profit and loss by group */
export function GroupPnl({ pnl, agents }: { pnl: Record<string, number | null | undefined>; agents: AgentInfo[] }) {
  const groups = Object.entries(pnl).filter(([, v]) => typeof v === 'number' && Number.isFinite(v)) as [string, number][]
  if (!groups.length) return <p className="note">No positions yet.</p>
  const info = new Map(agents.map(a => [a.group, a]))
  const mx = Math.max(0.005, ...groups.map(([, v]) => Math.abs(v)))
  const rows = groups.map(([g, v]) => {
    const a = info.get(g)
    const d = g === 'background' ? null : disposition(g, a?.alphas, a?.model)
    return { g, v, label: d ? d.label : 'Background traders', color: d ? d.color : 'var(--muted)', order: d ? d.order : 999 }
  }).sort((a, b) => a.order - b.order)
  return (
    <div>
      {rows.map(r => <div className="pnlrow" key={r.g}>
        <span className="pl" title={r.label}>{r.label}</span>
        <span className="pnlbar"><i style={{ background: r.color, width: `${(Math.abs(r.v) / mx) * 50}%`, left: r.v >= 0 ? '50%' : `${50 - (Math.abs(r.v) / mx) * 50}%` }} /></span>
        <span className="mono r">{(r.v >= 0 ? '+' : '') + pct(r.v, 2)}</span>
      </div>)}
      <div className="pnl-axis"><span>loss</span><span>0</span><span>gain</span></div>
    </div>
  )
}
