import type { Frame, Metrics, AgentInfo, EpConfig } from './types'

export function shockTick(cfg: EpConfig, frames: Frame[]): number {
  if (typeof cfg.shock_t === 'number') return cfg.shock_t
  const s = cfg.shocks?.find(x => x.kind !== 'correction')
  if (s) return s.t
  const f = frames.find(fr => fr.news?.length)
  return f ? f.t : 0
}
export function correctionTick(cfg: EpConfig): number | null {
  if (typeof cfg.correction_t === 'number') return cfg.correction_t
  const c = cfg.shocks?.find(x => x.kind === 'correction')
  return c ? c.t : null
}
export const epCapital = (c: EpConfig): number | undefined => (c.capital ?? c.capital_ratio ?? c.adoption) as number | undefined
export const epSafeguard = (c: EpConfig): string => (c.safeguard as string) || (c.safeguards && Object.keys(c.safeguards).length ? Object.keys(c.safeguards).join('+') : 'none')
export const epHcode = (c: EpConfig): string | undefined => c.hcode ?? (c.headline_kind ? `${c.headline_kind}_${c.source_verified ? 'ver' : 'unv'}` : undefined)

const mean = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN)

/** Contract section 4 metrics, computed from the frames seen so far (used while a replay is in progress). */
export function computeMetrics(frames: Frame[], cfg: EpConfig, agents: AgentInfo[]): Metrics & { trough_t?: number } {
  if (!frames.length) return {}
  const ts = shockTick(cfg, frames)
  const pre = frames.filter(f => f.t < ts).slice(-5)
  const p0 = pre.length ? mean(pre.map(f => f.price)) : frames[0].price
  const post = frames.filter(f => f.t >= ts)
  if (!post.length) return { p0, pnl_by_group: groupPnl(frames[frames.length - 1], agents) }
  let tr = post[0]
  for (const f of post) if (f.price < tr.price) tr = f
  const crash = Math.max(0, (p0 - tr.price) / p0)
  const overshoot = Math.max(0, (tr.fundamental - tr.price) / p0)
  // Recovery: minutes from the headline until the price is back within 1% of fundamental value for 3 straight
  // minutes, counted from the trough on (0 if it never left the band; null if it has not returned yet).
  const dis = (f: Frame) => Math.abs(f.price - f.fundamental) / f.fundamental
  let rec: number | null = null
  if (!post.some(f => dis(f) >= 0.01)) rec = post.length >= 3 ? 0 : null
  else {
    const i0 = post.indexOf(tr)
    for (let i = i0; i + 2 < post.length; i++) if (dis(post[i]) < 0.01 && dis(post[i + 1]) < 0.01 && dis(post[i + 2]) < 0.01) { rec = post[i].t - ts; break }
  }
  return {
    p0, trough: tr.price, trough_t: tr.t, crash_depth: crash, overshoot, recovery_ticks: rec,
    n_liquidations: frames.reduce((s, f) => s + (f.liquidations || 0), 0),
    liquidation_volume: frames.reduce((s, f) => s + (f.liquidation_volume || 0), 0),
    volume_total: frames.reduce((s, f) => s + (f.volume || 0), 0),
    pnl_by_group: groupPnl(frames[frames.length - 1], agents),
  }
}
function groupPnl(last: Frame, agents: AgentInfo[]): Record<string, number> {
  const info = new Map(agents.map(a => [a.id, a]))
  const g: Record<string, number[]> = {}
  for (const a of last.agents) { const i = info.get(a.id); if (i && i.cash0) (g[i.group] ||= []).push(a.equity / i.cash0 - 1) }
  return Object.fromEntries(Object.entries(g).map(([k, v]) => [k, mean(v)]))
}
