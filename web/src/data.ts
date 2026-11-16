// Data layer. Source order: live API on the same origin, then static files under data/ (results/site copied into
// dist/data), then nothing. The in-browser mock is used only when forced with ?mock=1. There is no silent fallback
// to mock data: a missing file is reported as missing and the page shows an empty state.
import type { Meta, Episode, EpisodeSummary, SessionRow, ArenaState, Metrics } from './types'
import { HOSTED_MCP, LIVE_ORIGIN } from './config'
import { BrowserSimError, arenaActInBrowser, arenaStartInBrowser, simulateInBrowser } from './sim/client'

export type Source = 'live' | 'static' | 'mock' | 'none'
export interface SourceInfo { source: Source; meta: Meta }
export type Loaded<T> = { state: 'loading' } | { state: 'ok'; data: T } | { state: 'missing'; what: string } | { state: 'error'; error: string }

type MockModule = typeof import('./mock')
let mockMod: Promise<MockModule> | null = null
const mock = () => (mockMod ??= import('./mock'))

class Missing extends Error { constructor(public what: string) { super(`missing: ${what}`) } }

async function getJSON<T>(url: string, ms = 15000): Promise<T> {
  const ctl = new AbortController(); const to = setTimeout(() => ctl.abort(), ms)
  try {
    const r = await fetch(url, { signal: ctl.signal, headers: { Accept: 'application/json' } })
    if (r.status === 404) throw new Missing(url)
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`)
    const ct = r.headers.get('content-type') ?? ''
    const txt = await r.text()
    // Static hosts often answer unknown paths with the SPA's index.html; treat that as missing.
    if (!ct.includes('json') && /^\s*</.test(txt)) throw new Missing(url)
    try { return JSON.parse(txt) as T } catch (e) {
      // Python's json writes NaN and Infinity, which are not JSON; read them as null rather than failing the page.
      const fixed = txt.replace(/(?<=[:,\[]\s*)(?:-?Infinity|NaN)(?=\s*[,\]}])/g, 'null')
      if (fixed === txt) throw e
      return JSON.parse(fixed) as T
    }
  } finally { clearTimeout(to) }
}

let infoP: Promise<SourceInfo> | null = null
export function init(): Promise<SourceInfo> {
  return (infoP ??= (async () => {
    const q = new URLSearchParams(location.search)
    if (q.get('mock') === '1') { const m = await mock(); return { source: 'mock' as Source, meta: m.MOCK_META } }
    try { const meta = await getJSON<Meta>('/api/meta', 4000); if (meta && typeof meta === 'object') return { source: 'live' as Source, meta } } catch { /* try static */ }
    try { const meta = await getJSON<Meta>('data/meta.json', 6000); return { source: 'static' as Source, meta } } catch { /* nothing */ }
    return { source: 'none' as Source, meta: {} }
  })())
}

const cache = new Map<string, Promise<unknown>>()
function once<T>(key: string, f: () => Promise<T>): Promise<T> {
  let p = cache.get(key) as Promise<T> | undefined
  if (!p) { p = f(); cache.set(key, p); p.catch(() => cache.delete(key)) }
  return p
}

/** A precomputed result set by name (dose_response, tipping_point, yardstick, safeguards, early_warning, decide, arena_runs, engine_validation). */
export function result<T>(name: string): Promise<T> {
  return once(`r:${name}`, async () => {
    const i = await init()
    if (i.source === 'mock') { const m = await mock(); const d = m.mockResult(name); if (d === undefined) throw new Missing(`results/${name}.json`); return d as T }
    if (i.source === 'none') throw new Missing(`results/${name}.json`)
    try {
      return await getJSON<T>(i.source === 'live' ? `/api/results/${name}` : `data/results/${name}.json`)
    } catch (e) {
      if (e instanceof Missing) throw new Missing(`results/${name}.json`)
      // A server error on the API (for example a file it cannot serialise) falls back to the same file served statically.
      if (i.source === 'live') { try { return await getJSON<T>(`data/results/${name}.json`) } catch { /* report the first error */ } }
      throw e
    }
  })
}

export function episodes(): Promise<EpisodeSummary[]> {
  return once('episodes', async () => {
    const i = await init()
    if (i.source === 'mock') return (await mock()).mockIndex()
    if (i.source === 'none') throw new Missing('episodes/index.json')
    try {
      const d = await getJSON<EpisodeSummary[]>(i.source === 'live' ? '/api/episodes?limit=1000' : 'data/episodes/index.json')
      return Array.isArray(d) ? d : []
    } catch (e) { if (e instanceof Missing) throw new Missing('episodes/index.json'); throw e }
  })
}

export function episode(id: string): Promise<Episode> {
  return once(`ep:${id}`, async () => {
    const i = await init()
    if (i.source === 'mock') return (await mock()).mockEpisode(id)
    if (i.source === 'none') throw new Missing(`episodes/${id}.json`)
    try {
      return await getJSON<Episode>(i.source === 'live' ? `/api/episodes/${encodeURIComponent(id)}` : `data/episodes/${encodeURIComponent(id)}.json`, 30000)
    } catch (e) { if (e instanceof Missing) throw new Missing(`episodes/${id}.json`); throw e }
  })
}

export function describeError(e: unknown): Loaded<never> {
  if (e instanceof Missing) return { state: 'missing', what: e.what }
  return { state: 'error', error: e instanceof Error ? e.message : String(e) }
}

// ------------------------------------------------------------------ live endpoints
// Markets, the browser arena and the session board need a running simulator. In order of preference: the server
// this page came from, the hosted server in LIVE_ORIGIN (static copies of the site), and, for markets and the
// browser arena, the same Python running inside the visitor's browser.

let apiP: Promise<string | null> | null = null
/** Base URL of a reachable live server ('' for the same origin), or null when there is none. */
export function liveApi(): Promise<string | null> {
  return (apiP ??= (async () => {
    const i = await init()
    if (i.source === 'live') return ''
    if (i.source !== 'static' || !LIVE_ORIGIN) return null
    const base = LIVE_ORIGIN.replace(/\/+$/, '')
    try { const h = await getJSON<{ ok?: boolean }>(`${base}/api/health`, 3500); return h?.ok ? base : null } catch { return null }
  })())
}

export async function sessions(): Promise<SessionRow[] | null> {
  const i = await init()
  if (i.source === 'mock') return (await mock()).mockSessions()
  const base = await liveApi()
  if (base === null) return null
  try { const d = await getJSON<SessionRow[]>(`${base}/api/sessions`, 4000); return Array.isArray(d) ? d : [] } catch { return null }
}

/** The server could not be reached or failed; the caller may use the in-browser simulator instead. */
class ServerDown extends Error {}
async function post<T>(url: string, body: unknown, ms = 30000): Promise<T> {
  const ctl = new AbortController(); const to = setTimeout(() => ctl.abort(), ms)
  let r: Response
  try {
    r = await fetch(url, { method: 'POST', signal: ctl.signal, headers: { 'Content-Type': 'application/json', Accept: 'application/json' }, body: JSON.stringify(body) })
  } catch (e) { clearTimeout(to); throw new ServerDown(e instanceof Error ? e.message : String(e)) }
  try {
    if (r.status === 404 || r.status === 405 || r.status >= 500) throw new ServerDown(`HTTP ${r.status}`)
    const txt = await r.text()
    const parse = () => { try { return JSON.parse(txt) } catch { return JSON.parse(txt.replace(/(?<=[:,\[]\s*)(?:-?Infinity|NaN)(?=\s*[,\]}])/g, 'null')) } }
    if (!r.ok) {
      let msg = r.status === 429 ? 'Too many runs at once. Try again in a few seconds.' : `The request was rejected (HTTP ${r.status}).`
      try { const j = parse(); if (j?.detail && r.status !== 429) msg = String(j.detail) } catch { /* keep the generic message */ }
      throw new Error(msg)
    }
    try { return parse() as T } catch { throw new ServerDown('bad JSON') }
  } finally { clearTimeout(to) }
}

const browserSessions = new Set<string>()
const browserFailed = (e: unknown) => new Error(e instanceof BrowserSimError && e.status ? e.message : 'The market could not start in this browser. Try reloading the page.')

export async function arenaStart(scenario: string, agent_name: string): Promise<ArenaState> {
  const i = await init()
  if (i.source === 'mock') return (await mock()).mockArenaStart(scenario)
  const base = await liveApi()
  if (base !== null) {
    try { return await post<ArenaState>(`${base}/api/arena/start`, { scenario, agent_name }) } catch (e) { if (!(e instanceof ServerDown)) throw e }
  }
  try {
    const s = await arenaStartInBrowser<ArenaState>({ scenario, agent_name })
    if (s.session_id) browserSessions.add(s.session_id)
    return s
  } catch (e) { throw browserFailed(e) }
}
export async function arenaAct(id: string, target: number | null): Promise<ArenaState> {
  const i = await init()
  if (i.source === 'mock') return (await mock()).mockArenaAct(id, target)
  const body = target === null ? { advance: true } : { target_exposure_x: target, advance: true }
  if (browserSessions.has(id)) {
    try { return await arenaActInBrowser<ArenaState>(id, body) } catch (e) { throw browserFailed(e) }
  }
  const base = await liveApi()
  try { return await post<ArenaState>(`${base ?? ''}/api/arena/${encodeURIComponent(id)}/act`, body) } catch (e) {
    if (e instanceof ServerDown) throw new Error('Lost the connection to the server. Start a new session.')
    throw e
  }
}

/** Run one market live (a headline episode and its no-headline twin, table-driven agents). */
export interface SimParams { regime: string; pop: string; capital: number; hcode: string; safeguard: string; seed?: number; resilience?: string }
export interface SimResult { episode: Episode; twin?: { metrics?: Metrics }; metrics?: Metrics; elapsed_ms?: number; where?: 'server' | 'browser' }
/** No simulator is available; callers fall back to recorded runs without telling the visitor. */
export class SimUnavailable extends Error {}
const okEpisode = (d: SimResult | null | undefined) => !!d && !!d.episode && Array.isArray(d.episode.frames) && d.episode.frames.length > 0

/** True when this copy of the site can run markets (a server, or the in-browser simulator). */
export async function canSimulate(): Promise<boolean> {
  const i = await init()
  return i.source === 'live' || i.source === 'static'
}

export async function simulate(p: SimParams, ms = 45000): Promise<SimResult> {
  const i = await init()
  if (i.source !== 'live' && i.source !== 'static') throw new SimUnavailable(i.source)
  const base = await liveApi()
  if (base !== null) {
    try {
      const d = await post<SimResult>(`${base}/api/simulate`, p, ms)
      if (okEpisode(d)) return { ...d, where: 'server' }
    } catch (e) { if (!(e instanceof ServerDown)) throw e }
  }
  let d: SimResult
  try { d = await simulateInBrowser<SimResult>(p as unknown as Record<string, unknown>) } catch (e) {
    if (e instanceof BrowserSimError && e.status === 400) throw new Error(e.message)
    throw new SimUnavailable(e instanceof Error ? e.message : String(e))
  }
  if (!okEpisode(d)) throw new SimUnavailable('empty episode')
  return { ...d, where: 'browser' }
}

/** Public origin shown to people (copy-paste commands, tour captions). `?origin=https://...` overrides it for
 *  recordings made on a local server; it never changes where requests go. */
export const displayOrigin = (): string => {
  try {
    const o = new URLSearchParams(location.search).get('origin')
    if (o && /^https?:\/\/[\w.-]+(:\d+)?$/.test(o)) return o
  } catch { /* ignore */ }
  return location.origin
}
/** The MCP endpoint people should add: this server, or the hosted one named in LIVE_ORIGIN; null when neither exists. */
export const mcpUrl = (source: Source = 'live'): string | null =>
  source === 'live' || source === 'mock' ? `${displayOrigin()}/mcp/` : LIVE_ORIGIN ? `${LIVE_ORIGIN.replace(/\/+$/, '')}/mcp/` : HOSTED_MCP || null
