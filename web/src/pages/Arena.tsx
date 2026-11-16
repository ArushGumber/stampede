import { useEffect, useMemo, useRef, useState } from 'react'
import * as D from '../data'
import type { Cell, Episode, EpisodeSummary, Metrics, SgRow } from '../types'
import { HCODE, HCODE_ORDER, REGIME_NOTE, hcodeShort, regimeLabel, sgLabel } from '../config'
import { computeMetrics, correctionTick, epCapital, epHcode, epSafeguard, shockTick } from '../metrics'
import { useLoaded, useResult, useSource, Empty, Info, Panel, Seg, Sel, SimTag, Tile } from '../components/ui'
import { AgentGrid, DepthLadder, GroupPnl, NewsFeed, PriceChart } from '../components/arena'
import { capLabel, disposition, fin, num, pct, popInfo, popOrder, uniq } from '../vocab'
import { api, isTour } from '../modes'
import { onSimStatus, simStatus, type SimStatus } from '../sim/client'

interface Choice { regime: string; pop: string; capital: number; hcode: string; safeguard: string }
// The steering safeguard is recorded as a separate population ("r50_shift"); present it as r50 + steer_cautious.
const keyOf = (e: EpisodeSummary): Choice => {
  const steer = e.config.tag === 'safeguards_steer'
  const pop = String(e.config.pop ?? e.config.groups?.[0]?.name ?? 'unknown')
  return {
    regime: e.config.regime, pop: steer ? pop.replace(/_shift$/, '') : pop, capital: epCapital(e.config) ?? NaN,
    hcode: epHcode(e.config) ?? 'unknown', safeguard: steer ? 'steer_cautious' : epSafeguard(e.config),
  }
}

function nearest(list: EpisodeSummary[], c: Choice): { ep: EpisodeSummary; diffs: string[] } | null {
  let best: EpisodeSummary | null = null, bd = Infinity
  for (const e of list) {
    const k = keyOf(e)
    const d = (k.regime === c.regime ? 0 : 1000) + (k.pop === c.pop ? 0 : 400) + (k.hcode === c.hcode ? 0 : 200) + (k.safeguard === c.safeguard ? 0 : 100)
      + (fin(k.capital) && fin(c.capital) ? 10 * Math.abs(Math.log2(k.capital / c.capital)) : 50) + (e.config.seed ?? 0) * 1e-3
    if (d < bd) { bd = d; best = e }
  }
  if (!best) return null
  const k = keyOf(best), diffs: string[] = []
  if (k.regime !== c.regime) diffs.push(`${regimeLabel(k.regime)} instead of ${regimeLabel(c.regime)}`)
  if (k.pop !== c.pop) diffs.push(`population ${popInfo(k.pop).label} instead of ${popInfo(c.pop).label}`)
  if (k.hcode !== c.hcode) diffs.push(`${hcodeShort(k.hcode)} headline instead of ${hcodeShort(c.hcode)}`)
  if (k.safeguard !== c.safeguard) diffs.push(`${sgLabel(k.safeguard)} instead of ${sgLabel(c.safeguard)}`)
  if (k.capital !== c.capital) diffs.push(`capital ${capLabel(k.capital)} instead of ${capLabel(c.capital)}`)
  return { ep: best, diffs }
}

const RATE = 3 // simulated minutes per second at 1x
const FEATURED = 'ep_07a9edc0bbaf'
const CAP_MIN = 0.25, CAP_MAX = 3

interface LiveRun { res: D.SimResult; key: Choice; at: number }
const SHIFTED: Record<string, string> = { base: 'base_shift', 'gemma-base': 'base_shift', r50: 'r50_shift' }

export default function Arena({ epId, speed: speed0, shot }: { epId?: string; speed?: number; shot?: boolean }) {
  const src = useSource()
  const idx = useLoaded(() => D.episodes(), [])
  const tip = useResult<Cell[]>('tipping_point')
  const sg = useResult<SgRow[]>('safeguards')
  const list = idx.state === 'ok' ? idx.data : []
  const meta = src.state === 'ok' ? src.data.meta : {}
  // Markets run on the server when there is one, otherwise in the browser; only the mock and empty modes cannot.
  const canLive = src.state === 'ok' && (src.data.source === 'live' || src.data.source === 'static')
  const [simSt, setSimSt] = useState<SimStatus>(simStatus())
  useEffect(() => onSimStatus(setSimSt), [])
  const cells = useMemo(() => (tip.state === 'ok' && Array.isArray(tip.data) ? tip.data : []), [tip])
  const sgRows = useMemo(() => (sg.state === 'ok' && Array.isArray(sg.data) ? sg.data : []), [sg])

  const opts = useMemo(() => {
    const ks = list.map(keyOf)
    return {
      regimes: uniq(ks.map(k => k.regime)).sort(),
      pops: uniq(ks.map(k => k.pop)).sort((a, b) => popOrder(a) - popOrder(b) || a.localeCompare(b)),
      capitals: uniq(ks.map(k => k.capital).filter(fin)).sort((a, b) => a - b),
      hcodes: uniq(ks.map(k => k.hcode)).sort((a, b) => (HCODE_ORDER.indexOf(a) + 99) % 99 - (HCODE_ORDER.indexOf(b) + 99) % 99),
      safeguards: uniq(ks.map(k => k.safeguard)).sort((a, b) => (a === 'none' ? -1 : b === 'none' ? 1 : a.localeCompare(b))),
    }
  }, [list])

  const [choice, setChoice] = useState<Choice | null>(null)
  const [forced, setForced] = useState<string | undefined>(epId)
  const [live, setLive] = useState<LiveRun | null>(null)
  const [busy, setBusy] = useState(false)
  const [simErr, setSimErr] = useState('')
  // Default: the paper's featured episode (unsteered Gemma, crypto, 3x, false post, no safeguard), whose crash starts
  // at the headline; otherwise the deepest false-headline episode without a safeguard.
  useEffect(() => {
    if (choice || !list.length) return
    if (epId) { const e = list.find(x => x.id === epId); if (e) { setChoice(keyOf(e)); return } }
    const feat = list.find(x => x.id === FEATURED)
    if (feat) { setChoice(keyOf(feat)); setForced(FEATURED); return }
    const fakes = list.filter(e => (epHcode(e.config) ?? '').startsWith('fake') && ((e.config as { safeguard?: string }).safeguard ?? 'none') === 'none')
    const pick = (fakes.length ? fakes : list).reduce((a, b) => ((b.metrics.overshoot ?? 0) > (a.metrics.overshoot ?? 0) ? b : a))
    setChoice(keyOf(pick))
  }, [list, epId, choice])

  const match = useMemo(() => {
    if (forced) { const e = list.find(x => x.id === forced); if (e) return { ep: e, diffs: [] as string[] } }
    return choice ? nearest(list, choice) : null
  }, [list, choice, forced])
  const set = (k: keyof Choice, v: string | number) => { setForced(undefined); setLive(null); setSimErr(''); setChoice(c => (c ? { ...c, [k]: v } : c)) }

  const directId = forced && idx.state === 'ok' && !list.some(x => x.id === forced) ? forced : undefined
  const loadId = directId ?? match?.ep.id
  const epL = useLoaded<Episode | null>(() => (loadId ? D.episode(loadId) : Promise.resolve(null)), [loadId])
  const recEp = epL.state === 'ok' ? epL.data : null
  const ep: Episode | null = live?.res.episode ?? recEp
  useEffect(() => { if (recEp && !choice) setChoice(keyOf(recEp)) }, [recEp, choice])
  const shown: EpisodeSummary | null = live ? { id: live.res.episode.id, config: live.res.episode.config, metrics: live.res.episode.metrics } : match?.ep ?? (recEp ? { id: recEp.id, config: recEp.config, metrics: recEp.metrics } : null)

  const [cursor, setCursor] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(speed0 ?? 1)
  const loadedFor = useRef<Episode | null>(null)
  const autoNext = useRef(false)
  const frames = ep?.frames ?? []
  const lastT = frames.length ? frames[frames.length - 1].t : 0
  const shockT = ep ? shockTick(ep.config, frames) : null
  const corrT = ep ? correctionTick(ep.config) : null
  useEffect(() => {
    if (!ep || loadedFor.current === ep) return
    const auto = (loadedFor.current === null && !!epId) || autoNext.current
    autoNext.current = false
    loadedFor.current = ep
    const st = shockTick(ep.config, ep.frames)
    if (shot) { setCursor(ep.frames.length ? ep.frames[ep.frames.length - 1].t : 0); setPlaying(false); return }
    setCursor(Math.max(0, st - (auto ? 4 : 1)))
    setPlaying(auto)
  }, [ep, epId, shot])
  // Scripted capture: window.__stampede.seek(t) jumps the replay; .play(speed) starts it; .arena reports state.
  useEffect(() => {
    api.seek = (t: number) => { setPlaying(false); setCursor(Math.max(0, Math.min(Math.round(t), lastT))) }
    api.play = (sp?: number) => { if (sp) setSpeed(sp); setPlaying(true) }
    return () => { delete api.seek; delete api.play }
  }, [lastT])
  const statsReady = tip.state !== 'loading' && sg.state !== 'loading'
  useEffect(() => { api.arena = { episodeId: ep?.id ?? '', t: cursor, lastT, shockT, ready: !!ep && statsReady, live: !!live } }, [ep, cursor, lastT, shockT, statsReady, live])
  useEffect(() => () => { api.arena = undefined }, [])
  useEffect(() => {
    if (!playing) return
    const id = setInterval(() => setCursor(c => { if (c >= lastT) { setPlaying(false); return c } return c + 1 }), 1000 / (RATE * speed))
    return () => clearInterval(id)
  }, [playing, speed, lastT])

  const playFromHeadline = () => { setCursor(Math.max(0, (shockT ?? 1) - 4)); setPlaying(true) }
  const release = () => {
    if (!ep) return
    if (playing) { setPlaying(false); return }
    if (cursor >= lastT) setCursor(Math.max(0, (shockT ?? 1) - 1))
    setPlaying(true)
  }
  const run = async () => {
    if (!choice || busy) return
    setSimErr(''); setPlaying(false)
    if (!canLive) { playFromHeadline(); return }
    setBusy(true)
    try {
      // "Steer everyone cautious" is a population, not an exchange rule: the cautious-shifted version of the same mix.
      const steer = choice.safeguard === 'steer_cautious'
      const pop = steer ? SHIFTED[choice.pop] ?? choice.pop : choice.pop
      const res = await D.simulate({ regime: choice.regime, pop, capital: choice.capital, hcode: choice.hcode, safeguard: steer ? 'none' : choice.safeguard })
      autoNext.current = true
      setForced(undefined)
      setLive({ res, key: choice, at: Date.now() })
    } catch (e) {
      if (e instanceof D.SimUnavailable) { setLive(null); playFromHeadline() }
      else setSimErr(e instanceof Error ? e.message : String(e))
    }
    setBusy(false)
  }

  const done = !!ep && cursor >= lastT
  const preShock = shockT !== null && cursor < shockT
  const vis = useMemo(() => frames.filter(f => f.t <= cursor), [frames, cursor])
  const running = useMemo(() => (ep ? computeMetrics(vis, ep.config, ep.agents) : {}), [ep, vis])
  const fin_ = done && ep ? { ...running, ...ep.metrics, recovery_ticks: running.recovery_ticks } : running
  const frame = vis[vis.length - 1]
  const news = useMemo(() => vis.flatMap(f => (f.news ?? []).map(n => ({ ...n, t: f.t }))).reverse(), [vis])
  const cfg = ep?.config
  const hcode = live ? live.key.hcode : cfg ? epHcode(cfg) : undefined
  const truth = hcode ? HCODE[hcode]?.truth : cfg?.headline_kind === 'true' ? 'true' : cfg?.headline_kind ? 'false' : undefined
  const pinfo = cfg ? popInfo(String(live ? live.key.pop : cfg.pop ?? ''), meta.populations, cfg.groups) : null
  // The excess measure of a recorded run comes from the sweep cell it belongs to (twins are not saved); a live run
  // brings its own twin.
  const cellFor = (k: Choice, epid?: string, tag?: unknown) => {
    const up = k.hcode === 'fake_unv_bull'
    if (tag === 'safeguards' || tag === 'safeguards_steer' || (!epid && k.safeguard !== 'none')) {
      const r = sgRows.find(x => x.pop === k.pop && x.regime === k.regime && x.capital === k.capital && x.safeguard === k.safeguard)
      const h = k.hcode === 'fake_unv' || k.hcode === 'fake_ver' ? k.hcode : null
      if (!r || !h || !fin(r[`${h}_overshoot`])) return null
      return { v: r[`${h}_overshoot`] as number, lo: r[`${h}_lo`] as number | undefined, hi: r[`${h}_hi`] as number | undefined, n: fin(r.n) ? r.n : undefined, up: false }
    }
    const c = (epid ? cells.find(x => x.episode_ids?.includes(epid)) : undefined) ?? cells.find(x => (!tag || x.tag === tag) && x.pop === k.pop && x.regime === k.regime && x.hcode === k.hcode && x.capital === k.capital && (x.safeguard_name ?? x.safeguard ?? 'none') === k.safeguard)
    if (!c) return null
    const key = up ? 'excess_overshoot_up' : 'excess_overshoot'
    if (!fin(c[key])) return null
    return { v: c[key] as number, lo: c[`${key}_lo`] as number | undefined, hi: c[`${key}_hi`] as number | undefined, n: c.n, up }
  }
  const cellStat = useMemo(() => (ep && !live ? cellFor(keyOf(ep), ep.id, cfg?.tag) : null), [ep, live, cells, sgRows, cfg]) // eslint-disable-line react-hooks/exhaustive-deps
  const liveStat = useMemo(() => {
    if (!live) return null
    const m = (live.res.metrics ?? live.res.episode.metrics ?? {}) as Metrics, tw = (live.res.twin?.metrics ?? {}) as Metrics
    const up = live.key.hcode === 'fake_unv_bull'
    const key = up ? 'overshoot_up' : 'overshoot'
    const g = (o: Metrics, k: string) => (fin(o[k]) ? (o[k] as number) : NaN)
    const ex = fin(g(m, `excess_${key}`)) ? g(m, `excess_${key}`) : fin(g(m, key)) && fin(g(tw, key)) ? g(m, key) - g(tw, key) : NaN
    return { v: ex, up, cell: cellFor(live.key) }
  }, [live, cells, sgRows]) // eslint-disable-line react-hooks/exhaustive-deps
  const maxLev = useMemo(() => {
    const reg = cfg?.regime === 'crypto' ? 10 : 2
    const obs = Math.max(0, ...frames.flatMap(f => f.agents.map(a => Math.abs(a.exposure))))
    return obs > reg * 1.05 ? Math.ceil(obs) : reg
  }, [frames, cfg])
  const headlineText = useMemo(() => frames.find(f => f.t === shockT)?.news?.[0], [frames, shockT])
  const symbol = cfg?.symbol ?? 'asset'
  const unit = cfg?.regime === 'crypto' ? 'coins' : 'shares'
  const bull = liveStat?.up ?? cellStat?.up ?? hcode === 'fake_unv_bull'
  const rise = useMemo(() => {
    if (shockT === null) return undefined
    const pre = vis.filter(f => f.t < shockT).slice(-5), post = vis.filter(f => f.t >= shockT)
    if (!pre.length || !post.length) return undefined
    const p0 = pre.reduce((a, f) => a + f.price, 0) / pre.length
    return Math.max(0, Math.max(...post.map(f => f.price)) / p0 - 1)
  }, [vis, shockT])

  if (idx.state === 'missing' && !epId || (idx.state === 'ok' && !list.length && !epId)) {
    return <div><Head /><Empty title="No recorded episodes on this server yet" what="episodes/index.json is missing or empty">The Arena replays recorded market episodes. Meanwhile the sweep results are on <a href="#/tipping">Tipping point</a>{src.state === 'ok' && src.data.source === 'live' ? <>, and you can trade the market yourself on <a href="#/connect">Connect an agent</a></> : null}.</Empty></div>
  }
  if (idx.state === 'error') return <div><Head /><div className="banner err">Could not load the episode index: {idx.error}</div></div>

  const priceTake = (() => {
    if (!ep || !frame) return 'Loading.'
    if (shockT === null || cursor < shockT) return `Before the headline the price tracks fundamental value (gap ${pct(Math.abs(frame.price - frame.fundamental) / frame.fundamental, 2)}). The headline reaches all ${ep.agents.length} agents at minute ${shockT}.`
    const m = running
    const tr = m.trough_t !== undefined ? ` at minute ${m.trough_t}` : ''
    const ov = fin(m.overshoot) && m.overshoot > 0.0005 ? `, ${pct(m.overshoot, 2)} below fundamental value` : ''
    const rec = m.recovery_ticks === null || m.recovery_ticks === undefined ? (cursor < lastT ? '; not yet back within 1% of fundamental' : '; it did not return within 1% of fundamental in this run') : `; back within 1% of fundamental ${m.recovery_ticks} min after the headline`
    const f0 = frames.find(f => f.t === (shockT as number) - 1)?.fundamental, f1 = frame.fundamental
    const fmove = truth === 'true' && f0 && Math.abs(f1 / f0 - 1) > 0.005 ? ` The fundamental value itself moved ${pct(f1 / f0 - 1, 1)}, because this headline is true.` : ''
    if (bull) return `Since the false good-news headline the price has risen at most ${pct(rise, 2)} above its pre-headline level${rec}.`
    return `Since the headline the price has fallen at most ${pct(m.crash_depth, 2)} from its pre-headline level${tr}${ov}${rec}.${fmove}`
  })()
  const priceShort = !ep || !frame ? 'Loading' : shockT === null || cursor < shockT ? `Headline lands at minute ${shockT}` : bull ? `Up ${pct(rise, 2)} since the headline` : `Down ${pct(running.crash_depth, 2)} since the headline`
  const after = shockT !== null && cursor >= shockT
  const ivl = (c: { lo?: number; hi?: number } | null | undefined) => (c && fin(c.lo) && fin(c.hi) ? `[${pct(c.lo, 2)}, ${pct(c.hi, 2)}]` : '')
  const exTile = live
    ? <Tile tone="accent" k={bull ? 'Excess rise' : 'Excess overshoot'} v={fin(liveStat?.v) ? pct(liveStat!.v, 2) : '–'} s="this run vs its no-headline twin"
        info={<>How far the price moved {bull ? 'above' : 'below'} fair value beyond what the same market, same seed, did without the headline.{liveStat?.cell ? <> Recorded cell mean for these settings: {pct(liveStat.cell.v, 2)} {ivl(liveStat.cell)}{liveStat.cell.n ? `, ${liveStat.cell.n} seeds` : ''}.</> : null}</>} />
    : <Tile tone="accent" k={bull ? 'Excess rise' : 'Excess overshoot'} v={cellStat ? pct(cellStat.v, 2) : '–'} s={cellStat ? `mean of ${cellStat.n ?? 'all'} seeds ${ivl(cellStat)}` : 'no sweep cell matches this run'}
        info={`How far the price moved ${bull ? 'above' : 'below'} fair value beyond the twin run with the same seed and no headline, averaged over the seeds of this cell, with a bootstrap interval.`} />
  const tiles = (
    <div className="tiles">
      {exTile}
      {bull
        ? <Tile k="Peak rise" v={after ? pct(rise, 2) : '–'} s="highest price after the headline vs before" />
        : <Tile k="Crash depth" v={after ? pct(fin_.crash_depth, 2) : '–'} s={fin(fin_.trough) && after ? `trough ${num(fin_.trough, fin_.trough! >= 1000 ? 1 : 2)} vs ${num(fin_.p0, fin_.p0! >= 1000 ? 1 : 2)} before` : 'fall from the pre-headline price'} />}
      <Tile k="Liquidations" v={fin(fin_.n_liquidations) ? String(fin_.n_liquidations) : '–'} s={fin_.liquidation_volume ? `${num(fin_.liquidation_volume, 1)} ${unit} force-sold` : 'forced sales by margin calls'} tone={fin_.n_liquidations ? 'bad' : undefined} />
      <Tile k="Recovery" v={!after ? '–' : fin_.recovery_ticks === null || fin_.recovery_ticks === undefined ? (done ? 'none' : 'not yet') : `${fin_.recovery_ticks} min`} s="until back within 1% of fair value"
        info="Minutes from the headline until the price stays within 1% of fundamental value for three minutes, counted from the trough." />
    </div>
  )
  const sk = live ? live.key : shown ? keyOf(shown) : null
  const cellLine = sk ? `${pinfo?.label ?? sk.pop}, ${regimeLabel(sk.regime).toLowerCase()} market, ${hcodeShort(sk.hcode).toLowerCase()} headline, capital ${capLabel(sk.capital)}, ${sgLabel(sk.safeguard).toLowerCase()}` : ''
  const agentsAre = cfg?.model === 'surrogate' || ep?.agents[0]?.kind === 'surrogate' ? 'Agents replay each model’s measured choices (response tables).' : 'Agents are model agents.'
  const diffs = !live && match ? match.diffs : []
  const status = (
    <div className="chips">
      {live
        ? <span className="chip live"><span className="dot" />Live run<Info align="left">Simulated {live.res.where === 'browser' ? 'in your browser' : 'on the server'}{fin(live.res.elapsed_ms) ? ` in ${num((live.res.elapsed_ms as number) / 1000, 1)} s` : ''}: {cellLine}{fin(live.res.episode.config.seed) ? `, seed ${live.res.episode.config.seed}` : ''}. {agentsAre}</Info></span>
        : <span className="chip rec"><span className="dot" />Recorded run<Info align="left">{cellLine}{shown ? `, seed ${shown.config.seed ?? '?'}` : ''}. {shown?.config.tag ? `Sweep ${String(shown.config.tag)}, ` : ''}episode {shown?.id}. {agentsAre}</Info></span>}
      {diffs.length > 0 && <span className="chip warn"><span className="dot" />Nearest recorded match<Info align="left">No recorded run has exactly these settings, so this one differs: {diffs.join('; ')}.{canLive ? ' Press Run market to simulate your exact settings.' : ''}</Info></span>}
      {simErr && <span className="chip warn"><span className="dot" />{simErr}</span>}
    </div>
  )

  if (shot) return (
    <div className="shot-arena">
      <div className="shot-head"><b>Recorded replay.</b> {cellLine}, seed {shown?.config.seed ?? '?'}. {symbol} is a fictional asset and every item is simulated.</div>
      <div className="shot-grid">
        <div className="shot-main">
          <Panel title={<>{symbol} price</>} sub={cfg ? `${regimeLabel(cfg.regime)}, ${unit}` : undefined} right={<span className="mono note">minute {frame?.t ?? 0} / {lastT}</span>}>
            <PriceChart frames={frames} cursor={cursor} nTicks={cfg?.n_ticks ?? Math.max(lastT + 1, 2)} shockT={shockT} corrT={corrT} symbol={symbol} troughT={(running as { trough_t?: number }).trough_t} p0={running.p0} height={340} peak={bull} />
          </Panel>
          {tiles}
        </div>
        <div className="shot-side">
          <Panel title="Agents" sub={ep ? `${ep.agents.length}, grouped by disposition` : undefined}>
            {ep ? <AgentGrid frames={frames} cursor={cursor} agents={ep.agents} shockT={shockT} maxLev={maxLev} compact /> : <p className="note">Loading.</p>}
          </Panel>
          <Panel title="News feed" sub="what every agent sees"><NewsFeed items={news} truth={truth} done={done} /></Panel>
        </div>
      </div>
    </div>
  )

  const capMarks = opts.capitals.filter(c => c >= CAP_MIN && c <= CAP_MAX)
  return (
    <div>
      <Head />
      <div className="arena">
        <section className="panel controls" aria-label="Build a market">
          <h2 className="ctl-h">Build a market</h2>
          <Seg label="Market" value={choice?.regime ?? ''} options={opts.regimes.map(r => ({ v: r, label: regimeLabel(r) }))} onChange={v => set('regime', v)} />
          {choice && <p className="note mt-4" style={{ marginBottom: 16 }}>{REGIME_NOTE[choice.regime] ?? ''}</p>}
          <div className="field">
            <label className="lab" htmlFor="pop">Who trades (24 agents)</label>
            <select id="pop" value={choice?.pop ?? ''} onChange={e => set('pop', e.target.value)}>
              {(['closed', 'open'] as const).map(fam => {
                const ps = opts.pops.filter(p => disposition(p).family === fam)
                return ps.length ? <optgroup key={fam} label={fam === 'closed' ? 'Commercial models' : 'Steered open model'}>{ps.map(p => <option key={p} value={p}>{popInfo(p, meta.populations).label}</option>)}</optgroup> : null
              })}
            </select>
          </div>
          <div className="field">
            <label className="lab" htmlFor="cap"><span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>Agent capital<Info>Total agent equity as a multiple of the resting order-book value within 1% of the mid price. Recorded levels: {capMarks.map(capLabel).join(', ')}.</Info></span><span className="val">{fin(choice?.capital) ? `${num(choice!.capital, 2)}x depth` : '–'}</span></label>
            <input id="cap" type="range" min={CAP_MIN} max={CAP_MAX} step={0.05} value={fin(choice?.capital) ? choice!.capital : 1} list="capmarks" onChange={e => set('capital', Math.round(+e.target.value * 100) / 100)} />
            <datalist id="capmarks">{capMarks.map(c => <option key={c} value={c} />)}</datalist>
          </div>
          <div className="field">
            <div className="lab">Headline</div>
            <div className="radios" role="radiogroup" aria-label="Headline">
              {opts.hcodes.map(h => <label key={h} className={choice?.hcode === h ? 'on' : ''}><input type="radio" name="hcode" checked={choice?.hcode === h} onChange={() => set('hcode', h)} /><span><b>{HCODE[h]?.short ?? h}</b><span className="note">{HCODE[h]?.long ?? ''}</span></span></label>)}
            </div>
          </div>
          <Sel label="Safeguard" value={choice?.safeguard ?? 'none'} options={opts.safeguards.map(s => ({ v: s, label: sgLabel(s) }))} onChange={v => set('safeguard', v)} />
          {headlineText && <div className="hl-preview"><div className="lab" style={{ justifyContent: 'flex-start' }}>Headline text <SimTag /></div><p>&ldquo;{headlineText.text}&rdquo;</p><span className="note">{headlineText.source}</span></div>}
          <button className="btn wide" onClick={run} disabled={!choice || busy}>{busy ? <><span className="spin" aria-hidden />{simSt === 'loading' ? 'Loading simulator' : 'Running market'}</> : canLive ? 'Run market' : 'Play from the headline'}</button>
          {busy && simSt === 'loading' && <p className="note sim-loading">Loading the simulator in your browser, about 10 MB, first run only.</p>}
        </section>

        <div className="arena-center">
          {epL.state === 'missing' && !live && <Empty title="This episode's trace is not on the server" what={epL.what} />}
          {epL.state === 'error' && !live && <div className="banner err">Could not load the episode: {epL.error}</div>}
          <Panel title={<>{symbol} price</>} sub={<span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>{priceShort}<Info align="left">{priceTake}</Info></span>}
            right={status}>
            <PriceChart frames={frames} cursor={cursor} nTicks={cfg?.n_ticks ?? Math.max(lastT + 1, 2)} shockT={shockT} corrT={corrT} symbol={symbol} troughT={(running as { trough_t?: number }).trough_t} p0={running.p0} peak={bull} height={isTour() ? 215 : undefined} />
            <div className="transport runbar">
              <button className={`btn sm ${preShock && !playing ? '' : 'ghost'}`} onClick={release} disabled={!ep}>{playing ? 'Pause' : done ? 'Replay' : preShock ? 'Release the headline' : 'Play'}</button>
              <input type="range" min={0} max={lastT} value={Math.min(cursor, lastT)} aria-label="Scrub through simulated minutes" onChange={e => { setPlaying(false); setCursor(+e.target.value) }} />
              <span className="clock">{String(frame?.t ?? 0).padStart(2, '0')} / {lastT} min</span>
              <Seg small value={speed} options={[0.5, 1, 2, 4].map(s => ({ v: s, label: `${s}x` }))} onChange={setSpeed} label="" />
            </div>
          </Panel>
          {tiles}
          <Panel title="Agents" sub={ep ? `${ep.agents.length} agents, grouped by disposition` : undefined}>
            {ep ? <AgentGrid frames={frames} cursor={cursor} agents={ep.agents} shockT={shockT} maxLev={maxLev} /> : <p className="note">Loading.</p>}
          </Panel>
        </div>

        <div className="arena-right">
          <Panel title="Order book" sub={`Top levels, ${unit}`}><DepthLadder frame={frame} unit={unit} /></Panel>
          <Panel title="News" sub="What every agent sees"><NewsFeed items={news} truth={truth} done={done} /></Panel>
          <Panel title="Return by group" sub="Mean return on starting equity">
            {ep && fin_.pnl_by_group ? <GroupPnl pnl={fin_.pnl_by_group} agents={ep.agents} /> : <p className="note">No data yet.</p>}
          </Panel>
        </div>
      </div>
    </div>
  )
}

function Head() {
  return <div className="pagehead arena-head"><h1>Arena</h1><p>Pick who trades, how much money they manage and which headline lands. Then run the market and watch the order book react.</p></div>
}
