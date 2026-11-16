import { useEffect, useMemo, useRef, useState } from 'react'
import * as D from '../data'
import type { Cell, Episode, EpisodeSummary, SgRow } from '../types'
import { HCODE, HCODE_ORDER, REGIME_NOTE, hcodeShort, regimeLabel, sgLabel } from '../config'
import { computeMetrics, correctionTick, epCapital, epHcode, epSafeguard, shockTick } from '../metrics'
import { useLoaded, useResult, useSource, Empty, Panel, Seg, Sel, SimTag, Tile, Takeaway } from '../components/ui'
import { AgentGrid, DepthLadder, GroupPnl, NewsFeed, PriceChart } from '../components/arena'
import { capLabel, disposition, fin, num, pct, popInfo, popOrder, uniq } from '../vocab'
import { api, isTour } from '../modes'

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

export default function Arena({ epId, speed: speed0, shot }: { epId?: string; speed?: number; shot?: boolean }) {
  const src = useSource()
  const idx = useLoaded(() => D.episodes(), [])
  const tip = useResult<Cell[]>('tipping_point')
  const sg = useResult<SgRow[]>('safeguards')
  const list = idx.state === 'ok' ? idx.data : []
  const meta = src.state === 'ok' ? src.data.meta : {}
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
  // Default: the paper's featured episode (unsteered Gemma, crypto, 3x, false post, no safeguard), whose crash starts
  // at the headline; otherwise the deepest false-headline episode without a safeguard.
  useEffect(() => {
    if (choice || !list.length) return
    if (epId) { const e = list.find(x => x.id === epId); if (e) { setChoice(keyOf(e)); return } }
    const feat = list.find(x => x.id === 'ep_07a9edc0bbaf')
    if (feat) { setChoice(keyOf(feat)); return }
    const fakes = list.filter(e => (epHcode(e.config) ?? '').startsWith('fake') && ((e.config as { safeguard?: string }).safeguard ?? 'none') === 'none')
    const pick = (fakes.length ? fakes : list).reduce((a, b) => ((b.metrics.overshoot ?? 0) > (a.metrics.overshoot ?? 0) ? b : a))
    setChoice(keyOf(pick))
  }, [list, epId, choice])

  const match = useMemo(() => {
    if (forced) { const e = list.find(x => x.id === forced); if (e) return { ep: e, diffs: [] as string[] } }
    return choice ? nearest(list, choice) : null
  }, [list, choice, forced])
  const set = (k: keyof Choice, v: string | number) => { setForced(undefined); setChoice(c => (c ? { ...c, [k]: v } : c)) }

  const directId = forced && idx.state === 'ok' && !list.some(x => x.id === forced) ? forced : undefined
  const loadId = directId ?? match?.ep.id
  const epL = useLoaded<Episode | null>(() => (loadId ? D.episode(loadId) : Promise.resolve(null)), [loadId])
  const ep = epL.state === 'ok' ? epL.data : null
  useEffect(() => { if (ep && !choice) setChoice(keyOf(ep)) }, [ep, choice])
  const shown: EpisodeSummary | null = match?.ep ?? (ep ? { id: ep.id, config: ep.config, metrics: ep.metrics } : null)

  const [cursor, setCursor] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(speed0 ?? 1)
  const loadedFor = useRef<string | null>(null)
  const frames = ep?.frames ?? []
  const lastT = frames.length ? frames[frames.length - 1].t : 0
  const shockT = ep ? shockTick(ep.config, frames) : null
  const corrT = ep ? correctionTick(ep.config) : null
  useEffect(() => {
    if (!ep || loadedFor.current === ep.id) return
    const auto = loadedFor.current === null && !!epId
    loadedFor.current = ep.id
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
  useEffect(() => { api.arena = { episodeId: ep?.id ?? '', t: cursor, lastT, shockT, ready: !!ep && statsReady } }, [ep, cursor, lastT, shockT, statsReady])
  useEffect(() => () => { api.arena = undefined }, [])
  useEffect(() => {
    if (!playing) return
    const id = setInterval(() => setCursor(c => { if (c >= lastT) { setPlaying(false); return c } return c + 1 }), 1000 / (RATE * speed))
    return () => clearInterval(id)
  }, [playing, speed, lastT])

  const release = () => {
    if (!ep) return
    if (playing) { setPlaying(false); return }
    if (cursor >= lastT) setCursor(Math.max(0, (shockT ?? 1) - 1))
    setPlaying(true)
  }

  const done = !!ep && cursor >= lastT
  const preShock = shockT !== null && cursor < shockT
  const vis = useMemo(() => frames.filter(f => f.t <= cursor), [frames, cursor])
  const running = useMemo(() => (ep ? computeMetrics(vis, ep.config, ep.agents) : {}), [ep, vis])
  const fin_ = done && ep ? { ...running, ...ep.metrics, recovery_ticks: running.recovery_ticks } : running
  const frame = vis[vis.length - 1]
  const news = useMemo(() => vis.flatMap(f => (f.news ?? []).map(n => ({ ...n, t: f.t }))).reverse(), [vis])
  const cfg = ep?.config
  const hcode = cfg ? epHcode(cfg) : undefined
  const truth = hcode ? HCODE[hcode]?.truth : cfg?.headline_kind === 'true' ? 'true' : cfg?.headline_kind ? 'false' : undefined
  const pinfo = cfg ? popInfo(String(cfg.pop ?? ''), meta.populations, cfg.groups) : null
  // The twin runs are not saved, so the excess measure comes from the sweep cell this episode belongs to.
  const cellStat = useMemo(() => {
    if (!ep) return null
    const k = keyOf(ep), tag = cfg?.tag
    const up = k.hcode === 'fake_unv_bull'
    if (tag === 'safeguards' || tag === 'safeguards_steer') {
      const r = sgRows.find(x => x.pop === k.pop && x.regime === k.regime && x.capital === k.capital && x.safeguard === k.safeguard)
      const h = k.hcode === 'fake_unv' || k.hcode === 'fake_ver' ? k.hcode : null
      if (!r || !h || !fin(r[`${h}_overshoot`])) return null
      return { v: r[`${h}_overshoot`] as number, lo: r[`${h}_lo`] as number | undefined, hi: r[`${h}_hi`] as number | undefined, n: fin(r.n) ? r.n : undefined, up: false, src: 'safeguard sweep cell' }
    }
    const c = cells.find(x => x.episode_ids?.includes(ep.id)) ?? cells.find(x => (!tag || x.tag === tag) && x.pop === k.pop && x.regime === k.regime && x.hcode === k.hcode && x.capital === k.capital && (x.safeguard_name ?? x.safeguard ?? 'none') === k.safeguard)
    if (!c) return null
    const key = up ? 'excess_overshoot_up' : 'excess_overshoot'
    if (!fin(c[key])) return null
    return { v: c[key] as number, lo: c[`${key}_lo`] as number | undefined, hi: c[`${key}_hi`] as number | undefined, n: c.n, up, src: 'this cell' }
  }, [ep, cells, sgRows, cfg])
  const maxLev = useMemo(() => {
    const reg = cfg?.regime === 'crypto' ? 10 : 2
    const obs = Math.max(0, ...frames.flatMap(f => f.agents.map(a => Math.abs(a.exposure))))
    return obs > reg * 1.05 ? Math.ceil(obs) : reg
  }, [frames, cfg])
  const headlineText = useMemo(() => frames.find(f => f.t === shockT)?.news?.[0], [frames, shockT])
  const symbol = cfg?.symbol ?? 'asset'
  const unit = cfg?.regime === 'crypto' ? 'coins' : 'shares'
  const bull = cellStat?.up ?? hcode === 'fake_unv_bull'
  const rise = useMemo(() => {
    if (shockT === null) return undefined
    const pre = vis.filter(f => f.t < shockT).slice(-5), post = vis.filter(f => f.t >= shockT)
    if (!pre.length || !post.length) return undefined
    const p0 = pre.reduce((a, f) => a + f.price, 0) / pre.length
    return Math.max(0, Math.max(...post.map(f => f.price)) / p0 - 1)
  }, [vis, shockT])

  if (idx.state === 'missing' && !epId || (idx.state === 'ok' && !list.length && !epId)) {
    return <div><Head /><Empty title="No recorded episodes on this server yet" what="episodes/index.json is missing or empty">The Arena replays recorded market episodes. When the sweeps are exported, every population, capital level and headline condition that was run becomes playable here. Meanwhile the sweep results are on <a href="#/tipping">Tipping point</a>{src.state === 'ok' && src.data.source === 'live' ? <>, and you can trade the live arena yourself on <a href="#/connect">Connect your agent</a></> : null}.</Empty></div>
  }
  if (idx.state === 'error') return <div><Head /><div className="banner err">Could not load the episode index: {idx.error}</div></div>

  const priceTake = (() => {
    if (!ep || !frame) return 'Loading.'
    if (shockT === null || cursor < shockT) return `Before the headline the price tracks fundamental value (gap ${pct(Math.abs(frame.price - frame.fundamental) / frame.fundamental, 2)}). Press Release to send the headline to all ${ep.agents.length} agents at minute ${shockT}.`
    const m = running
    const tr = m.trough_t !== undefined ? ` at minute ${m.trough_t}` : ''
    const ov = fin(m.overshoot) && m.overshoot > 0.0005 ? `, ${pct(m.overshoot, 2)} below fundamental value` : ''
    const rec = m.recovery_ticks === null || m.recovery_ticks === undefined ? (cursor < lastT ? '; not yet back within 1% of fundamental' : '; it did not return within 1% of fundamental in this run') : `; back within 1% of fundamental ${m.recovery_ticks} min after the headline`
    const f0 = frames.find(f => f.t === (shockT as number) - 1)?.fundamental, f1 = frame.fundamental
    const fmove = truth === 'true' && f0 && Math.abs(f1 / f0 - 1) > 0.005 ? ` The fundamental value itself moved ${pct(f1 / f0 - 1, 1)}, because this headline is true.` : ''
    if (bull) return `Since the false good-news headline the price has risen at most ${pct(rise, 2)} above its pre-headline level${rec}.`
    return `Since the headline the price has fallen at most ${pct(m.crash_depth, 2)} from its pre-headline level${tr}${ov}${rec}.${fmove}`
  })()
  const after = shockT !== null && cursor >= shockT
  const statSub = cellStat ? `${fin(cellStat.lo) && fin(cellStat.hi) ? `[${pct(cellStat.lo, 2)}, ${pct(cellStat.hi, 2)}] ` : ''}${cellStat.src}${cellStat.n ? `, ${cellStat.n} seeds` : ''}; ${cellStat.up ? 'above' : 'below'} fair value beyond the no-headline twin` : 'no sweep cell matches this episode'
  const tiles = (
    <div className="tiles">
      <Tile k={bull ? 'Excess overshoot above fair value' : 'Excess overshoot'} v={cellStat ? pct(cellStat.v, 2) : '\u2013'} s={statSub} />
      {bull
        ? <Tile k="Peak rise, this replay" v={after ? pct(rise, 2) : '\u2013'} s="highest price after the headline vs before" />
        : <Tile k="Crash depth, this replay" v={after ? pct(fin_.crash_depth, 2) : '\u2013'} s={fin(fin_.trough) && after ? `trough ${num(fin_.trough, fin_.trough! >= 1000 ? 1 : 2)} vs ${num(fin_.p0, fin_.p0! >= 1000 ? 1 : 2)} before` : 'fall from the pre-headline price'} />}
      <Tile k="Liquidations" v={fin(fin_.n_liquidations) ? String(fin_.n_liquidations) : '\u2013'} s={fin_.liquidation_volume ? `${num(fin_.liquidation_volume, 1)} ${unit} force-sold` : 'forced sales by margin calls'} tone={fin_.n_liquidations ? 'bad' : undefined} />
      <Tile k="Recovery" v={!after ? '\u2013' : fin_.recovery_ticks === null || fin_.recovery_ticks === undefined ? (done ? 'none' : 'not yet') : `${fin_.recovery_ticks} min`} s="after the headline; back within 1% of fundamental for 3 min, counted from the trough" />
    </div>
  )
  const sk = shown ? keyOf(shown) : null
  const cellLine = sk && shown ? <>{pinfo?.label ?? sk.pop} · {regimeLabel(sk.regime)} · {hcodeShort(sk.hcode)} · capital {capLabel(sk.capital)} · {sgLabel(sk.safeguard).toLowerCase()} · seed {shown.config.seed ?? '?'}</> : null

  if (shot) return (
    <div className="shot-arena">
      <div className="shot-head"><b>Recorded replay</b> · {cellLine} · <span className="mono">{shown?.id}</span> · {symbol} is a fictional asset; every item is simulated</div>
      <div className="shot-grid">
        <div className="shot-main">
          <Panel title={<>{symbol} price and fundamental value</>} sub={cfg ? `${regimeLabel(cfg.regime)}, ${unit}` : undefined} right={<span className="mono note">minute {frame?.t ?? 0} / {lastT}</span>}>
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

  return (
    <div>
      <Head />
      <div className="arena">
        <section className="panel controls" aria-label="Scenario">
          <Seg label="Market regime" value={choice?.regime ?? ''} options={opts.regimes.map(r => ({ v: r, label: regimeLabel(r) }))} onChange={v => set('regime', v)} />
          {choice && <p className="note mt-4">{REGIME_NOTE[choice.regime] ?? ''}</p>}
          <div className="field">
            <label className="lab" htmlFor="pop">Population of 24 agents</label>
            <select id="pop" value={choice?.pop ?? ''} onChange={e => set('pop', e.target.value)}>
              {(['closed', 'open'] as const).map(fam => {
                const ps = opts.pops.filter(p => disposition(p).family === fam)
                return ps.length ? <optgroup key={fam} label={fam === 'closed' ? 'Closed models (measured policies)' : 'Steered open model'}>{ps.map(p => <option key={p} value={p}>{popInfo(p, meta.populations).label}</option>)}</optgroup> : null
              })}
            </select>
            {pinfo?.detail && <span className="note">{pinfo.detail}</span>}
          </div>
          <Seg small label="Agent capital, as a multiple of book depth" value={choice?.capital ?? NaN} options={opts.capitals.map(c => ({ v: c, label: capLabel(c) }))} onChange={v => set('capital', v)} />
          <p className="note mt-4">Total agent equity divided by the resting order-book value within 1% of the mid price.</p>
          <div className="field">
            <div className="lab">Headline condition</div>
            <div className="radios" role="radiogroup" aria-label="Headline condition">
              {opts.hcodes.map(h => <label key={h} className={choice?.hcode === h ? 'on' : ''}><input type="radio" name="hcode" checked={choice?.hcode === h} onChange={() => set('hcode', h)} /><span><b>{HCODE[h]?.short ?? h}</b><span className="note">{HCODE[h]?.long ?? ''}</span></span></label>)}
            </div>
          </div>
          <Sel label="Safeguard" value={choice?.safeguard ?? 'none'} options={opts.safeguards.map(s => ({ v: s, label: sgLabel(s) }))} onChange={v => set('safeguard', v)} />
          {opts.safeguards.length <= 1 && <p className="note mt-4">Only unprotected runs are recorded so far; safeguard runs appear here when exported.</p>}
          {opts.safeguards.length > 1 && <p className="note mt-4">Safeguard runs exist only for some populations and capital levels; the nearest recorded one is used.</p>}
          {headlineText && <div className="hl-preview"><div className="lab">Headline to release <SimTag /></div><p>&ldquo;{headlineText.text}&rdquo;</p><span className="note">{headlineText.source}</span></div>}
          <button className="btn wide" onClick={release} disabled={!ep}>{!ep ? 'Loading episode' : playing ? 'Pause' : done ? 'Replay from the headline' : shockT !== null && cursor < shockT ? 'Release the headline' : 'Resume'}</button>
        </section>

        <div className="arena-center">
          <div className="replay-banner" role="note">
            <b>Recorded replay.</b>{' '}
            {shown ? <>Matched cell: {cellLine}{shown.config.tag ? <> · sweep <span className="mono">{String(shown.config.tag)}</span></> : null} · <span className="mono">{shown.id}</span>.</> : 'Loading the episode index.'}
            {match && match.diffs.length > 0 && <span className="diff"> No episode is recorded for exactly your settings; nearest has {match.diffs.join('; ')}.</span>}
            {' '}Agents are {cfg?.model === 'surrogate' || ep?.agents[0]?.kind === 'surrogate' ? 'response-table surrogates of each model\'s measured choices' : 'model agents'}; nothing is computed live.
          </div>
          {epL.state === 'missing' && <Empty title="This episode's trace is not on the server" what={epL.what} />}
          {epL.state === 'error' && <div className="banner err">Could not load the episode: {epL.error}</div>}
          <Panel title={<>{symbol} price and fundamental value</>} sub={cfg ? `${regimeLabel(cfg.regime)}, ${unit}` : undefined}
            right={<span className="mono note">minute {frame?.t ?? 0} / {lastT}</span>}>
            <Takeaway>{priceTake}</Takeaway>
            <PriceChart frames={frames} cursor={cursor} nTicks={cfg?.n_ticks ?? Math.max(lastT + 1, 2)} shockT={shockT} corrT={corrT} symbol={symbol} troughT={(running as { trough_t?: number }).trough_t} p0={running.p0} peak={bull} height={isTour() ? 215 : undefined} />
            <div className="transport">
              <button className={`btn ${preShock && !playing ? '' : 'ghost'}`} onClick={release} disabled={!ep}>{playing ? 'Pause' : done ? 'Replay' : preShock ? 'Release the headline' : 'Play'}</button>
              <input type="range" min={0} max={lastT} value={Math.min(cursor, lastT)} aria-label="Scrub through simulated minutes" onChange={e => { setPlaying(false); setCursor(+e.target.value) }} />
              <Seg small value={speed} options={[0.5, 1, 2, 4].map(s => ({ v: s, label: `${s}x` }))} onChange={setSpeed} label="" />
            </div>
          </Panel>
          {tiles}
          <Panel title="Agents" sub={ep ? `${ep.agents.length} agents, grouped by disposition` : undefined}>
            {ep ? <AgentGrid frames={frames} cursor={cursor} agents={ep.agents} shockT={shockT} maxLev={maxLev} /> : <p className="note">Loading.</p>}
          </Panel>
        </div>

        <div className="arena-right">
          <Panel title="Order book" sub={`top levels, ${unit}`}><DepthLadder frame={frame} unit={unit} /></Panel>
          <Panel title="News feed" sub="what every agent sees"><NewsFeed items={news} truth={truth} done={done} /></Panel>
          <Panel title="Profit and loss by group" sub="mean return on starting equity">
            {ep && fin_.pnl_by_group ? <GroupPnl pnl={fin_.pnl_by_group} agents={ep.agents} /> : <p className="note">No data yet.</p>}
          </Panel>
        </div>
      </div>
    </div>
  )
}

function Head() {
  return <div className="pagehead arena-head"><h1>Arena</h1><p>Pick a population of AI trading agents, how much capital they manage and what kind of headline arrives, then release it and watch the order book. Every run here is a recorded replay of a simulated market with a fictional asset.</p></div>
}
