import { useEffect, useMemo, useState } from 'react'
import { scaleLinear } from 'd3-scale'
import { line as d3line, area as d3area } from 'd3-shape'
import type { DoseRecord, DoseSeries, DecideTable } from '../types'
import { KIND, KIND_ORDER, METRIC, TRAIT, metricLabel, modelLabel } from '../config'
import { useResult, LoadState, Empty, Panel, PageHead, Seg, Sel, Takeaway, Glyph, GlyphMark, LegendItem } from '../components/ui'
import { useWidth, TipBox, YGrid, XTicks, AxisTitle, robustDomain, spearman, type Tip } from '../components/chart'
import { fin, num, uniq } from '../vocab'
import { on } from '../modes'

const METRIC_ORDER = ['crra_gamma', 'kelly_fraction', 'mean_exposure', 'leverage_rate', 'credulity', 'react_unverified', 'react_verified', 'unethical_rate', 'safe_choices', 'capability']
const fmtM = (m: string, v: number) => (METRIC[m]?.frac ? `${(v * 100).toFixed(METRIC[m]?.fmt ?? 0)}%` : num(v, METRIC[m]?.fmt ?? 2))
const lc = (t: string) => (/^[A-Z][a-z]/.test(t) ? t[0].toLowerCase() + t.slice(1) : t)
const sa = (a: number) => (a > 0 ? `+${a}` : a < 0 ? `−${-a}` : '0')

function DoseChart({ series, metric, cursorA }: { series: DoseSeries[]; metric: string; cursorA?: number | null }) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const [tip, setTip] = useState<Tip | null>(null)
  const H = 210, m = { l: 44, r: 12, t: 12, b: 38 }
  const iw = Math.max(10, w - m.l - m.r), ih = H - m.t - m.b
  const kinds = KIND_ORDER.filter(k => series.some(s => s.kind === k))
  const by = (k: string) => series.filter(s => s.kind === k).sort((a, b) => a.alpha - b.alpha)
  const steer = by('steer')
  const allA = series.map(s => s.alpha)
  const x = scaleLinear().domain([Math.min(-0.5, ...allA), Math.max(0.5, ...allA)]).range([0, iw]).nice()
  const vals = series.flatMap(s => [s.value, s.lo, s.hi]).filter(fin)
  const rd = robustDomain(vals)
  const frac = !!METRIC[metric]?.frac
  const y = frac ? scaleLinear().domain([0, 1]).range([ih, 0]) : scaleLinear().domain([rd.lo, rd.hi]).range([ih, 0]).nice(4)
  const [ylo, yhi] = y.domain()
  const cy = (v: number) => y(Math.max(ylo, Math.min(yhi, v)))
  const off = (v: number) => v < ylo || v > yhi
  const line = d3line<DoseSeries>().x(d => x(d.alpha)).y(d => cy(d.value))
  const band = d3area<DoseSeries>().x(d => x(d.alpha)).y0(d => cy(d.lo)).y1(d => cy(d.hi))
  const v0 = steer.find(s => s.alpha === 0)?.value
  const capRef = metric === 'capability' && fin(v0) ? 0.95 * v0 : null
  const yfmt = (v: number) => (METRIC[metric]?.frac ? `${Math.round(v * 100)}` : String(+v.toFixed(2)))
  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const a = x.invert(e.clientX - r.left)
    const as = uniq(series.map(s => s.alpha)).sort((p, q) => Math.abs(p - a) - Math.abs(q - a))
    const near = as[0]; if (near === undefined) return
    const at = KIND_ORDER.map(k => ({ k, s: series.find(s => s.kind === k && s.alpha === near) })).filter(z => z.s)
    setTip({ x: m.l + x(near), y: m.t + 10, body: <><b>alpha {sa(near)}</b>{at.map(z => <div key={z.k}>{KIND[z.k]?.label.split(',')[0] ?? z.k}: <b>{fmtM(metric, z.s!.value)}</b>{z.s!.lo !== z.s!.hi ? ` [${fmtM(metric, z.s!.lo)}, ${fmtM(metric, z.s!.hi)}]` : ''}{z.s!.n ? ` n=${z.s!.n}` : ''}</div>)}</> })
  }
  return (
    <div className="chartbox" ref={ref}>
      {w > 0 && <svg width={w} height={H} role="img" aria-label={`${metricLabel(metric)} against steering strength`}>
        <g transform={`translate(${m.l},${m.t})`}>
          <YGrid y={y} width={iw} ticks={4} fmt={yfmt} zero />
          <line x1={x(0)} x2={x(0)} y1={0} y2={ih} className="zero-v" stroke="var(--axis)" />
          {capRef !== null && capRef >= ylo && <g><line x1={0} x2={iw} y1={y(capRef)} y2={y(capRef)} stroke="var(--bad)" strokeDasharray="2 3" /><text x={4} y={y(capRef) + 13} fontSize={10.5} fill="var(--bad)">95% of unsteered</text></g>}
          {kinds.includes('random') && <>
            <path d={line(by('random')) ?? ''} fill="none" style={{ stroke: KIND.random.color }} strokeWidth={1.3} strokeDasharray={KIND.random.dash} />
            {by('random').map(d => <g key={'r' + d.alpha}>{d.lo !== d.hi && <line x1={x(d.alpha)} x2={x(d.alpha)} y1={cy(d.lo)} y2={cy(d.hi)} style={{ stroke: KIND.random.color }} strokeWidth={1.2} />}<GlyphMark shape="ring" cx={x(d.alpha)} cy={cy(d.value)} r={3.6} color={KIND.random.color} /></g>)}
          </>}
          {kinds.includes('unrelated') && <>
            <path d={line(by('unrelated')) ?? ''} fill="none" style={{ stroke: KIND.unrelated.color }} strokeWidth={1.4} strokeDasharray={KIND.unrelated.dash} />
            {by('unrelated').map(d => <GlyphMark key={'u' + d.alpha} shape="triangle" cx={x(d.alpha)} cy={cy(d.value)} r={3.6} color={KIND.unrelated.color} />)}
          </>}
          {steer.some(d => d.lo !== d.hi) && <path d={band(steer) ?? ''} className="band" style={{ fill: 'var(--ink)' }} />}
          <path d={line(steer) ?? ''} fill="none" style={{ stroke: KIND.steer.color }} strokeWidth={2} />
          {steer.map(d => <GlyphMark key={'s' + d.alpha} shape="circle" cx={x(d.alpha)} cy={cy(d.value)} r={3.4} color={KIND.steer.color} />)}
          {(['prompt', 'lora'] as const).flatMap(k => by(k).map(d => <g key={k + d.alpha}>
            {d.lo !== d.hi && <line x1={x(d.alpha)} x2={x(d.alpha)} y1={cy(d.lo)} y2={cy(d.hi)} style={{ stroke: KIND[k].color }} strokeWidth={1.5} />}
            <GlyphMark shape={KIND[k].glyph as 'diamond' | 'square'} cx={x(d.alpha)} cy={cy(d.value)} r={4.4} color={KIND[k].color} />
          </g>))}
          {series.filter(s => off(s.value)).map((s, i) => <g key={'o' + i}>
            <path d={s.value > yhi ? `M${x(s.alpha)},${-2}l-4,7h8z` : `M${x(s.alpha)},${ih + 2}l-4,-7h8z`} style={{ fill: KIND[s.kind]?.color ?? 'var(--ink)' }} />
            <text x={x(s.alpha) + (x(s.alpha) > iw * 0.75 ? -6 : 6)} y={s.value > yhi ? 8 : ih - 4} textAnchor={x(s.alpha) > iw * 0.75 ? 'end' : 'start'} fontSize={10.5} fill="var(--ink2)">off scale: {fmtM(metric, s.value)}</text>
          </g>)}
          {fin(cursorA) && (() => {
            const pts = steer.map(d => ({ x: d.alpha, y: d.value }))
            let v = NaN
            for (let i = 0; i + 1 < pts.length; i++) if (cursorA >= pts[i].x && cursorA <= pts[i + 1].x) v = pts[i].y + (pts[i + 1].y - pts[i].y) * (cursorA - pts[i].x) / (pts[i + 1].x - pts[i].x || 1)
            const right = x(cursorA) > iw * 0.6
            return <g className="dial-cursor">
              <line x1={x(cursorA)} x2={x(cursorA)} y1={0} y2={ih} stroke="var(--accent)" strokeWidth={2} />
              {fin(v) && <><circle cx={x(cursorA)} cy={cy(v)} r={6} style={{ fill: 'var(--accent)', stroke: 'var(--surface)', strokeWidth: 2 }} />
                <text x={x(cursorA) + (right ? -8 : 8)} y={Math.max(12, cy(v) - 10)} textAnchor={right ? 'end' : 'start'} fontSize={12} fontWeight={600} fill="var(--ink)">{fmtM(metric, v)}</text></>}
            </g>
          })()}
          <XTicks x={x} height={ih} values={x.ticks(Math.min(9, Math.max(3, Math.floor(iw / 42))))} fmt={sa} />
          <AxisTitle x={iw / 2} y={ih + 31}>dial strength alpha (mean-difference units)</AxisTitle>
          <rect x={0} y={0} width={iw} height={ih} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setTip(null)} />
        </g>
      </svg>}
      <TipBox tip={tip} w={w} />
    </div>
  )
}

function metricTake(metric: string, s: DoseSeries[], trait: string): string {
  const TRAIT_NEG = `${TRAIT[trait]?.neg ?? 'negative'} prompt`, TRAIT_POS = `${TRAIT[trait]?.pos ?? 'positive'} prompt`
  const st = s.filter(d => d.kind === 'steer').sort((a, b) => a.alpha - b.alpha)
  if (st.length < 2) return st.length ? `One steered point: ${fmtM(metric, st[0].value)} at alpha ${sa(st[0].alpha)}.` : 'No steered measurements for this metric.'
  const f = (v: number) => fmtM(metric, v)
  const a = st[0], b = st[st.length - 1]
  const rho = spearman(st.map(d => d.alpha), st.map(d => d.value))
  const rd = robustDomain(s.flatMap(d => [d.value, d.lo, d.hi]).filter(fin))
  const outl = st.filter(d => d.value > rd.hi || d.value < rd.lo)
  const rnd = s.filter(d => d.kind === 'random').map(d => d.value)
  const rtxt = rnd.length ? ` Random directions of the same size give ${f(Math.min(...rnd))} to ${f(Math.max(...rnd))}.` : ''
  if (metric === 'capability') {
    const v = st.map(d => d.value), v0 = st.find(d => d.alpha === 0)?.value
    const bad = fin(v0) ? st.filter(d => d.value < 0.95 * v0) : []
    return `Steered capability stays between ${f(Math.min(...v))} and ${f(Math.max(...v))}${fin(v0) ? ` (unsteered ${f(v0)})` : ''}${bad.length ? `; below 95% of unsteered at alpha ${bad.map(d => sa(d.alpha)).join(', ')}` : ', never below 95% of unsteered'}.${rtxt}`
  }
  const pr = s.filter(d => d.kind === 'prompt').sort((p, q) => p.alpha - q.alpha)
  const ptxt = pr.length ? ` Prompt personas: ${pr.map(d => `${d.alpha < 0 ? TRAIT_NEG : TRAIT_POS} ${f(d.value)}`).join(', ')}.` : ''
  const trend = !fin(rho) ? '' : Math.abs(rho) >= 0.8 ? ', a steady trend' : Math.abs(rho) < 0.4 ? ', with no steady trend' : ', a loose trend'
  return `Steering moves it from ${f(a.value)} at alpha ${sa(a.alpha)} to ${f(b.value)} at ${sa(b.alpha)}${trend}.${outl.length ? ` Unstable reading at alpha ${outl.map(d => sa(d.alpha)).join(', ')} (${outl.map(d => f(d.value)).join(', ')}).` : ''}${rtxt}${ptxt}`
}

function settingLabel(al: Record<string, number>, own: string): string {
  const nz = Object.entries(al).filter(([, v]) => v)
  if (!nz.length) return 'unsteered'
  return nz.map(([t, v]) => (t === own ? sa(v) : `${TRAIT[t]?.label.toLowerCase() ?? t} ${sa(v)}`)).join(', ')
}

function TryIt() {
  const dec = useResult<DecideTable>('decide')
  const tab = dec.state === 'ok' ? dec.data : null
  const vids = tab ? Object.keys(tab.vignettes ?? {}) : []
  const [vid, setVid] = useState('')
  const [setting, setSetting] = useState('{}')
  useEffect(() => { if (vids.length && !vids.includes(vid)) setVid(vids[0]) }, [vids.join(','), vid]) // eslint-disable-line
  const v = tab?.vignettes?.[vid]
  const model = tab?.default_model ?? tab?.records?.[0]?.model
  const recs = useMemo(() => (tab?.records ?? []).filter(r => r.vignette_id === vid && (!model || r.model === model)), [tab, vid, model])
  const own = v?.trait ?? 'risk'
  const settings = useMemo(() => uniq(recs.map(r => JSON.stringify(Object.fromEntries(Object.entries(r.alphas ?? {}).filter(([, x]) => x))))).sort((a, b) => {
    const pa = JSON.parse(a), pb = JSON.parse(b)
    const ka = Object.keys(pa)[0], kb = Object.keys(pb)[0]
    const oa = !ka ? 0 : ka === own ? 0 : 1, ob = !kb ? 0 : kb === own ? 0 : 1
    return oa - ob || (pa[ka] ?? 0) - (pb[kb] ?? 0)
  }), [recs, own])
  useEffect(() => { if (settings.length && !settings.includes(setting)) setSetting(settings.includes('{}') ? '{}' : settings[0]) }, [settings, setting])
  const norm = (al: Record<string, number>) => JSON.stringify(Object.fromEntries(Object.entries(al ?? {}).filter(([, x]) => x)))
  const rec = recs.find(r => norm(r.alphas) === setting)
  const base = recs.find(r => norm(r.alphas) === '{}')
  if (dec.state !== 'ok') return <Panel title="Try it: one decision at one dial setting"><LoadState l={dec} label="precomputed decisions" expect="This panel replays decisions precomputed on fixed vignettes (results/decide.json): the model's choice distribution at each dial setting and a short stated reason. Nothing is generated live and there is no free-text input. It appears when the file is exported." /></Panel>
  const letters = v?.letters ?? v?.options.map((_, i) => String.fromCharCode(65 + i)) ?? []
  const top = (p?: number[]) => (p && p.length ? p.indexOf(Math.max(...p)) : -1)
  const ti = top(rec?.probs), bi = top(base?.probs)
  const take = rec && base && rec !== base && ti >= 0 && bi >= 0
    ? (ti === bi ? `At this setting the most likely answer is still ${letters[ti]} (${Math.round(rec.probs[ti] * 100)}%, unsteered ${Math.round(base.probs[bi] * 100)}%).` : `The dial moves the most likely answer from ${letters[bi]} (${Math.round(base.probs[bi] * 100)}% unsteered) to ${letters[ti]} (${Math.round(rec.probs[ti] * 100)}%).`)
    : rec && ti >= 0 ? `Unsteered, the most likely answer is ${letters[ti]} at ${Math.round(rec.probs[ti] * 100)}%.` : 'Pick a setting.'
  return (
    <Panel title="Try it: one decision at one dial setting" sub={`${modelLabel(model)}, precomputed; fixed vignettes only`}>
      <div className="grid2">
        <div>
          <Sel label="Vignette" value={vid} options={vids.map(id => ({ v: id, label: `${tab!.vignettes[id].title} (${tab!.vignettes[id].trait})` }))} onChange={setVid} />
          <div className="lab">What the model reads</div>
          <div className="quote">{v?.user ?? v?.text ?? ''}</div>
          <div style={{ marginTop: 10 }}>
            <Seg small label={`Dial setting (${TRAIT[own]?.label ?? own} dial unless named)`} value={setting} options={settings.map(s => ({ v: s, label: settingLabel(JSON.parse(s), own) }))} onChange={setSetting} />
          </div>
        </div>
        <div aria-live="polite">
          <Takeaway>{take}</Takeaway>
          <h3>Choice distribution</h3>
          {rec ? v?.options.map((o, i) => <div className={`prob ${i === ti ? 'top' : ''}`} key={i}><span className="ltr">{letters[i]}</span><div>{o}<div className="bar"><i style={{ width: `${(rec.probs[i] ?? 0) * 100}%` }} /></div></div><span className="mono r">{((rec.probs[i] ?? 0) * 100).toFixed(1)}%</span></div>) : <p className="note">No record for this setting.</p>}
          {rec && Object.entries(rec.alphas ?? {}).some(([t, a]) => t === 'integrity' && a) ? <p className="note stated">Stated reasons are withheld for integrity-steered settings; only the choice distribution is shown.</p> : rec?.reason && <div className="stated"><div className="k">Stated reason (model output)</div><p style={{ margin: '4px 0 2px' }}>&ldquo;{rec.reason}&rdquo;</p><span className="note">Generated separately after the choice. A stated reason may not reflect how the choice was actually computed.</span></div>}
          {tab?.note && <p className="note" style={{ marginTop: 8 }}>{tab.note}</p>}
        </div>
      </div>
    </Panel>
  )
}

export default function Dials() {
  const dr = useResult<DoseRecord[] | DoseRecord>('dose_response')
  const recs = useMemo<DoseRecord[]>(() => (dr.state === 'ok' ? (Array.isArray(dr.data) ? dr.data : [dr.data]) : []), [dr])
  const traits = ['risk', 'credulity', 'integrity']
  const have = uniq(recs.map(r => r.trait))
  const [trait, setTrait] = useState('risk')
  const models = uniq(recs.filter(r => r.trait === trait).map(r => r.model))
  const [model, setModel] = useState('')
  const [cursorA, setCursorA] = useState<number | null>(null)
  useEffect(() => on('dialAlpha', v => setCursorA(typeof v === 'number' ? v : null)), [])
  useEffect(() => { if (have.length && !have.includes(trait)) setTrait(have[0]) }, [have.join(',')]) // eslint-disable-line
  const mdl = models.includes(model) ? model : models[0]
  const rec = recs.find(r => r.trait === trait && r.model === mdl)
  const metrics = METRIC_ORDER.filter(mt => rec?.series.some(s => s.metric === mt && s.kind === 'steer'))
  const by = (mt: string) => (rec?.series ?? []).filter(s => s.metric === mt)
  const kinds = KIND_ORDER.filter(k => rec?.series.some(s => s.kind === k))
  const summary = (() => {
    if (!rec) return ''
    const end = (mt: string) => { const st = by(mt).filter(s => s.kind === 'steer').sort((a, b) => a.alpha - b.alpha); return st.length > 1 ? st : null }
    const parts: string[] = []
    for (const mt of metrics.filter(x => x !== 'capability').slice(0, 2)) { const st = end(mt); if (st) parts.push(`${lc(metricLabel(mt))} goes from ${fmtM(mt, st[0].value)} to ${fmtM(mt, st[st.length - 1].value)}`) }
    const cap = end('capability')
    const as = (rec.series.filter(s => s.kind === 'steer').map(s => s.alpha))
    return `Across alpha ${sa(Math.min(...as))} to ${sa(Math.max(...as))}, ${parts.join(' and ')}${cap ? `, while capability stays between ${fmtM('capability', Math.min(...cap.map(c => c.value)))} and ${fmtM('capability', Math.max(...cap.map(c => c.value)))}` : ''}.`
  })()
  return (
    <div>
      <PageHead title="Dials">Each dial adds one direction to the model's internal activations with a continuous strength, alpha. Here every behavioural measurement is plotted against alpha, next to controls that push the model by the same amount in a random or unrelated direction, and next to prompt-written personas where measured.</PageHead>
      {dr.state !== 'ok' ? <LoadState l={dr} label="dose-response results" /> : <>
        <div className="filters">
          <Seg label="Dial" value={trait} options={traits.map(t => ({ v: t, label: `${TRAIT[t].label}${have.includes(t) ? '' : ' (not yet)'}`, disabled: !have.includes(t), title: have.includes(t) ? `${TRAIT[t].neg} to ${TRAIT[t].pos}` : 'Not exported yet' }))} onChange={setTrait} />
          {models.length > 1 && <Sel label="Model" value={mdl} options={models.map(x => ({ v: x, label: modelLabel(x) }))} onChange={setModel} />}
          {rec && <p className="note" style={{ marginBottom: 10 }}>{modelLabel(rec.model)}, layer {rec.layer ?? '?'}, {rec.method ?? ''} vector{fin(rec.vector_norm) ? `, norm ${num(rec.vector_norm, 1)}` : ''}. Negative alpha is {TRAIT[trait]?.neg}, positive is {TRAIT[trait]?.pos}.</p>}
        </div>
        {have.length < 3 && <p className="note">Dials not yet exported: {traits.filter(t => !have.includes(t)).map(t => TRAIT[t].label).join(', ')}. They appear here when their runs finish.</p>}
        {!rec ? <Empty title={`No ${TRAIT[trait]?.label.toLowerCase()} dial results yet`} what="results/dose_response.json has no record for this dial" /> : <>
          <Takeaway>{summary}</Takeaway>
          <div className="legend">{kinds.map(k => <LegendItem key={k} glyph={<span className="lg-i">{['steer', 'random', 'unrelated'].includes(k) && <Glyph shape="line" color={KIND[k].color} dash={KIND[k].dash} />}<Glyph shape={KIND[k].glyph} color={KIND[k].color} size={11} /></span>}>{KIND[k].label}</LegendItem>)}
            {kinds.includes('prompt') ? <span className="note">Prompt personas have no strength; they are drawn at alpha {'\u2212'}1 ({TRAIT[trait]?.neg} prompt) and +1 ({TRAIT[trait]?.pos} prompt).</span> : <span className="note">Prompt personas: not exported for this dial.</span>}</div>
          <div className="smgrid">
            {metrics.map(mt => <Panel key={mt} id={`dose-${mt}`} title={METRIC[mt]?.label ?? mt} sub={METRIC[mt]?.unit}>
              <Takeaway>{metricTake(mt, by(mt), trait)}</Takeaway>
              <DoseChart series={by(mt)} metric={mt} cursorA={cursorA} />
              {METRIC[mt]?.hint && <p className="note" style={{ margin: '2px 0 0' }}>{METRIC[mt].hint}</p>}
            </Panel>)}
          </div>
          <details><summary>Table view of every measurement</summary><div className="tablewrap"><table className="t"><thead><tr><th>metric</th><th className="l">kind</th><th>alpha</th><th>value</th><th>low</th><th>high</th><th>n</th></tr></thead><tbody>
            {[...rec.series].sort((a, b) => METRIC_ORDER.indexOf(a.metric) - METRIC_ORDER.indexOf(b.metric) || KIND_ORDER.indexOf(a.kind) - KIND_ORDER.indexOf(b.kind) || a.alpha - b.alpha).map((s, i) => <tr key={i}><td>{metricLabel(s.metric)}</td><td className="l">{s.kind}</td><td>{sa(s.alpha)}</td><td>{num(s.value, 3)}</td><td>{num(s.lo, 3)}</td><td>{num(s.hi, 3)}</td><td>{s.n ?? ''}</td></tr>)}</tbody></table></div></details>
        </>}
      </>}
      <div style={{ marginTop: 14 }}><TryIt /></div>
    </div>
  )
}
