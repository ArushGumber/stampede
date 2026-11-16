import { displayOrigin } from './data'
// Automatic walkthrough for the demo video (?tour=1), following the paper's evaluation: one caption at a time in a strip
// docked under the page, about two and a half minutes. Every number in a caption is computed from the loaded results;
// a scene whose data is missing falls back to a caption without numbers and is listed in window.__stampede.tourMissing.
// When it finishes, window.__stampede.tourDone is set to true so a recorder knows when to stop.
import { useEffect, useState } from 'react'
import * as D from './data'
import type { ArenaRun, Cell, DoseRecord, EpisodeSummary, SgRow, YardRow } from './types'
import { CLOSED, PRODUCT_NAME } from './config'
import { api, emit } from './modes'
import { epCapital, epHcode, epSafeguard } from './metrics'
import { fin, num, pct } from './vocab'

interface Step { id: string; ms: number; caption: string; go?: () => void | Promise<void> }
const sleep = (ms: number) => new Promise(r => setTimeout(r, ms))
async function until(f: () => boolean, ms = 5000) { const t = Date.now(); while (!f() && Date.now() - t < ms) await sleep(80) }
async function scrollTo(id: string, block: ScrollLogicalPosition = 'center') {
  await until(() => !!document.getElementById(id), 6000)
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block })
}
async function openArena(id: string | undefined, speed: number) {
  if (!id) return
  location.hash = `#/arena/${encodeURIComponent(id)}?speed=${speed}`
  await until(() => api.arena?.episodeId === id && !!api.arena?.ready, 8000)
  // Keep the price chart and the metric tiles above the caption.
  document.querySelector('.arena-center')?.scrollIntoView({ block: 'start' })
}
const mean = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN)
const short = (a: string) => CLOSED[a]?.label.replace(/^Claude /, '') ?? a

async function buildSteps(missing: string[]): Promise<Step[]> {
  const safe = <T,>(p: Promise<T>, d: T) => p.catch(() => d)
  const [idx, tip, dose, yard, sgr, runs] = await Promise.all([
    safe(D.episodes(), [] as EpisodeSummary[]), safe(D.result<Cell[]>('tipping_point'), [] as Cell[]),
    safe(D.result<DoseRecord[]>('dose_response'), [] as DoseRecord[]), safe(D.result<YardRow[]>('yardstick'), [] as YardRow[]),
    safe(D.result<SgRow[]>('safeguards'), [] as SgRow[]), safe(D.result<ArenaRun[]>('arena_runs'), [] as ArenaRun[]),
  ])
  const have = new Set(idx.map(e => e.id))
  // The cell of the main sweep for a scenario, and its first recorded episode on this server.
  const cell = (pop: string, regime: string, hcode: string, capital: number) => tip.find(c => (c.tag === 'models' || !c.tag) && c.pop === pop && c.regime === regime && c.hcode === hcode && c.capital === capital && (c.safeguard_name ?? c.safeguard ?? 'none') === 'none')
    ?? tip.find(c => c.pop === pop && c.regime === regime && c.hcode === hcode && c.capital === capital)
  const epOf = (c: Cell | undefined): EpisodeSummary | undefined => {
    const id = c?.episode_ids?.find(i => have.has(i))
    if (id) return idx.find(e => e.id === id)
    return c ? idx.find(e => e.config.pop === c.pop && e.config.regime === c.regime && epHcode(e.config) === c.hcode && epCapital(e.config) === c.capital && epSafeguard(e.config) === 'none') : undefined
  }
  const ex = (c: Cell | undefined) => (c && fin(c.excess_overshoot) ? pct(c.excess_overshoot, 2) : null)
  const need = <T,>(v: T | null | undefined, scene: string): v is T => { if (v === null || v === undefined) { missing.push(scene); return false } return true }

  // Scene 1: the same false post at the same capital, equity market, 2x book depth.
  const cOpusU = cell('claude-opus', 'equity', 'fake_unv', 2), cGemU = cell('gemma-base', 'equity', 'fake_unv', 2), cOpusV = cell('claude-opus', 'equity', 'fake_ver', 2)
  const credOf = (a: string) => yard.find(r => r.agent === a && r.metric === 'credulity')?.value
  const s1a = need(ex(cOpusU), 'opus-rumour') ? `Everyone runs Claude Opus. A false rumour from an unverified post, agents manage 2x the book's depth: the price falls only ${ex(cOpusU)} below fair value beyond normal noise.` : 'Everyone runs Claude Opus, and a false rumour arrives from an unverified post.'
  const s1b = need(ex(cGemU), 'gemma-rumour') ? `Same post, same capital, 24 unsteered Gemma agents: ${ex(cGemU)} below fair value.` : 'Same post, same capital, unsteered Gemma agents.'
  const co = credOf('claude-opus'), cg = credOf('gemma-4-12b')
  const s1c = fin(co) && fin(cg) ? `Opus discounts unverified posts (credulity ${num(co, 2)}); Gemma takes them at face value (${num(cg, 2)}).` : (missing.push('credulity-pair'), 'Opus discounts unverified posts; Gemma takes them at face value.')
  const s1d = need(ex(cOpusV), 'opus-verified') ? `The same false story from a verified-looking source: now Opus falls ${ex(cOpusV)}.` : 'The same false story from a verified-looking source.'

  // Scene 2: the crypto crash.
  const cCry = cell('gemma-base', 'crypto', 'fake_unv', 3), eCry = epOf(cCry)
  const liqFree = eCry && (eCry.metrics.n_liquidations ?? 0) === 0
  const s2a = need(ex(cCry), 'crypto-crash') && eCry ? `Crypto market, unsteered Gemma managing 3x the book's depth: this run falls ${pct(eCry.metrics.crash_depth, 1)}${liqFree ? ' without a single forced liquidation' : ''}; across ${cCry!.n} seeds, ${ex(cCry)} below fair value beyond the twin.` : 'Crypto market, agents managing 3x the book’s depth.'
  const capSeries = [0.25, 1, 3].map(c => cell('gemma-base', 'equity', 'fake_unv', c))
  const s2b = capSeries.every(c => ex(c)) ? `Capital decides how far it goes: in equities the same rumour moves the price ${capSeries.map((c, i) => `${ex(c)} at ${[0.25, 1, 3][i]}x`).join(', ')}.` : (missing.push('capital-series'), 'Capital decides how far it goes.')

  // Scene 3: the risk dial.
  const risk = (Array.isArray(dose) ? dose : [dose]).find(d => d && d.trait === 'risk')
  const steer = (m: string) => (risk?.series ?? []).filter(s => s.kind === 'steer' && s.metric === m && fin(s.value)).sort((a, b) => a.alpha - b.alpha)
  const end = (m: string) => { const s = steer(m); const z = s.find(x => x.alpha === 0); return z && s.length ? [z, s[s.length - 1]] : null }
  const g = end('crra_gamma'), k = end('kelly_fraction')
  const sa = (a: number) => (a > 0 ? `+${a}` : `${a}`)
  const s3b = g && k ? `In economic units: from alpha 0 to ${sa(k[1].alpha)} the stake rises from ${num(k[0].value, 2)} to ${num(k[1].value, 2)} times the Kelly bet, and CRRA risk aversion falls from ${num(g[0].value, 2)} to ${num(g[1].value, 2)}.` : (missing.push('dial-units'), 'Calibrated in economic units: CRRA risk aversion and the Kelly stake.')
  const capS = steer('capability').map(s => s.value), capR = (risk?.series ?? []).filter(s => s.kind === 'random' && s.metric === 'capability').map(s => s.value)
  const s3c = capS.length && capR.length ? `Random directions of the same size break the model (quiz score ${pct(Math.min(...capR), 0)} to ${pct(Math.max(...capR), 0)}); the risk direction keeps it at ${pct(Math.min(...capS), 0)} to ${pct(Math.max(...capS), 0)}.` : (missing.push('dial-random'), 'Random directions of the same size break the model; the risk direction does not.')
  const pk = (risk?.series ?? []).filter(s => s.kind === 'prompt' && s.metric === 'kelly_fraction').sort((a, b) => a.alpha - b.alpha)
  const s3d = pk.length >= 2 ? `Written personas only reach the extremes: ${num(pk[0].value, 2)} and ${num(pk[pk.length - 1].value, 2)} times the Kelly stake.` : null

  // Scene 4: credulity yardstick.
  const cred = yard.filter(r => r.metric === 'credulity' && fin(r.value))
  const closed = cred.filter(r => CLOSED[r.agent]).sort((a, b) => a.value - b.value)
  const open = cred.filter(r => !CLOSED[r.agent]).map(r => r.value)
  const s4b = closed.length ? `How much an unverified post moves an agent relative to a verified wire: ${closed.map(r => `${short(r.agent)} ${num(r.value, 2)}`).join(', ')}${open.length ? `; every open model about ${num(mean(open), 2)}` : ''}.` : (missing.push('yardstick'), 'How much an unverified post moves an agent relative to a verified wire.')

  // Scene 5: safeguards at capital 2.
  const sg = (pop: string, regime: string, s: string) => sgr.find(r => r.pop === pop && r.regime === regime && r.capital === 2 && r.safeguard === s)
  const gN = sg('gemma-base', 'equity', 'none'), gH = sg('gemma-base', 'equity', 'hide_unverified')
  const s5a = gN && gH && fin(gN.fake_unv_overshoot) && fin(gH.fake_unv_overshoot) ? `Hiding unverified posts removes the rumour crash: unsteered Gemma, equities, 2x, from ${pct(gN.fake_unv_overshoot, 2)} to ${pct(gH.fake_unv_overshoot, 2)}.` : (missing.push('safeguard-filter'), 'Hiding unverified posts removes the rumour crash.')
  const s5b = gN && gH && fin(gN.fake_ver_overshoot) && fin(gH.fake_ver_overshoot) ? `It does nothing against the verified-looking fake: ${pct(gN.fake_ver_overshoot, 2)} without the filter, ${pct(gH.fake_ver_overshoot, 2)} with it.` : (missing.push('safeguard-fake-wire'), 'Nothing tested stops the verified-looking fake.')
  const rN = sg('r50', 'crypto', 'none'), rL = sg('r50', 'crypto', 'leverage_cap'), rS = sg('r50', 'crypto', 'steer_cautious')
  const s5c = rN && rL && rS && fin(rN.fake_unv_overshoot) ? `Removing the risk-takers backfires: in a half-reckless crypto market a leverage cap raises the crash from ${pct(rN.fake_unv_overshoot, 2)} to ${pct(rL.fake_unv_overshoot, 2)}, and steering everyone cautious to ${pct(rS.fake_unv_overshoot, 2)}.` : (missing.push('safeguard-backfire'), 'Removing the risk-takers who absorb the selling makes the crash deeper.')

  // Scene 6: agents in the loop over MCP.
  const fe = runs.filter(r => r.scenario === 'fake-equity')
  const byAgent = Object.keys(CLOSED).map(a => { const rs = fe.filter(r => r.agent === a); return { a, n: rs.length, k: rs.filter(r => r.sold_into_the_rumour).length } }).filter(z => z.n)
  const all = byAgent.filter(z => z.k === z.n), none = byAgent.filter(z => z.k === 0)
  const names = (zs: typeof byAgent) => zs.map(z => short(z.a)).join(zs.length === 2 ? ' and ' : ', ')
  const tot = (zs: typeof byAgent) => [zs.reduce((s, z) => s + z.k, 0), zs.reduce((s, z) => s + z.n, 0)]
  const s6b = runs.length && (all.length || none.length) ? `${runs.length} recorded MCP sessions. On a false equity rumour${all.length ? `, ${names(all)} sold every time (${tot(all)[0]} of ${tot(all)[1]})` : ''}${none.length ? `${all.length ? ';' : ','} ${names(none)} never did (${tot(none)[0]} of ${tot(none)[1]})` : ''}.` : (missing.push('arena-runs'), 'Commercial agents traded the arena themselves over MCP.')
  const te = runs.filter(r => r.scenario === 'true-equity')
  const s6c = te.length ? `On true news, ${te.filter(r => r.sold_into_the_rumour).length} of ${te.length} sessions sold, as they should.` : null

  const sweep = (from: number, to: number, ms: number) => {
    const t0 = performance.now()
    const id = setInterval(() => { const f = Math.min(1, (performance.now() - t0) / ms); emit('dialAlpha', Math.round((from + (to - from) * f) * 100) / 100); if (f >= 1) clearInterval(id) }, 80)
  }
  const steps: (Step | null)[] = [
    { id: 'hook', ms: 7000, go: () => openArena(epOf(cOpusU)?.id, 2), caption: 'Brokers are letting AI agents trade client accounts. What happens when they all read the same false headline?' },
    { id: 'opus-rumour', ms: 7500, caption: s1a },
    { id: 'gemma-rumour', ms: 7000, go: () => openArena(epOf(cGemU)?.id, 2), caption: s1b },
    { id: 'credulity-pair', ms: 5500, caption: s1c },
    { id: 'opus-verified', ms: 6000, go: () => openArena(epOf(cOpusV)?.id, 2), caption: s1d },
    { id: 'opus-verified-2', ms: 5000, caption: 'Discounting rumours does not protect against a compromised official account.' },
    { id: 'crypto-crash', ms: 9000, go: () => openArena(eCry?.id, 2), caption: s2a },
    { id: 'capital', ms: 7500, go: async () => { location.hash = '#/tipping?tag=models&regime=equity&hcode=fake_unv'; await until(() => !!document.querySelector('[data-cell]'), 6000) }, caption: s2b },
    { id: 'dials-1', ms: 5500, go: async () => { location.hash = '#/dials'; await until(() => !!document.getElementById('dose-kelly_fraction'), 6000); emit('dialAlpha', 0); sweep(0, 2, 10000) }, caption: 'Each agent’s disposition is a dial inside the model: one activation direction in Gemma-4-12B.' },
    { id: 'dials-2', ms: 8000, caption: s3b },
    s3d ? { id: 'dials-3', ms: 5500, caption: s3d } : null,
    { id: 'dials-4', ms: 6500, go: () => scrollTo('dose-capability'), caption: s3c },
    { id: 'yard-1', ms: 5000, go: async () => { emit('dialAlpha', null); location.hash = '#/yardstick'; await scrollTo('yard-credulity') }, caption: 'The same tests place commercial agents on the same scales.' },
    { id: 'yard-2', ms: 9000, caption: s4b },
    { id: 'sg-1', ms: 6500, go: async () => { location.hash = '#/safeguards'; await scrollTo('sg-gemma-base-equity', 'start') }, caption: s5a },
    { id: 'sg-2', ms: 5500, caption: s5b },
    { id: 'sg-3', ms: 9000, go: () => scrollTo('sg-r50-crypto', 'start'), caption: s5c },
    { id: 'connect-1', ms: 5500, go: async () => { location.hash = '#/connect'; await scrollTo('connect-add', 'start') }, caption: 'Any MCP client can join the market: one command adds it to Claude Code.' },
    { id: 'connect-2', ms: 9000, go: () => scrollTo('connect-runs', 'start'), caption: s6b },
    s6c ? { id: 'connect-3', ms: 5000, caption: s6c } : null,
    { id: 'close', ms: 7000, go: () => { location.hash = '#/arena'; window.scrollTo(0, 0); document.getElementById('root')?.scrollTo?.(0, 0) }, caption: `${PRODUCT_NAME}: paper, code and live demo at ${displayOrigin()}` },
  ]
  return steps.filter((s): s is Step => !!s)
}

export default function Tour() {
  const [cap, setCap] = useState('')
  const [shown, setShown] = useState(false)
  useEffect(() => {
    let dead = false
    // Keep the flag in the query so route changes inside the tour do not drop it.
    if (new URLSearchParams(location.search).get('tour') !== '1') history.replaceState(null, '', `?tour=1${location.hash.replace(/\?.*$/, '')}`)
    api.tourDone = false
    ;(async () => {
      const missing: string[] = []
      const steps = await buildSteps(missing)
      api.tourMissing = missing
      const t0 = performance.now()
      for (const s of steps) {
        if (dead) return
        const ts = performance.now()
        api.tourStep = s.id
        await s.go?.()
        if (dead) return
        setCap(s.caption); setShown(true)
        await sleep(Math.max(0, s.ms - (performance.now() - ts)))
      }
      setShown(false)
      api.tourSeconds = Math.round((performance.now() - t0) / 100) / 10
      api.tourDone = true
    })()
    return () => { dead = true }
  }, [])
  return <div className={`tour-cap ${shown && cap ? '' : 'hidden'}`} role="status" aria-live="polite"><span>{cap}</span></div>
}
