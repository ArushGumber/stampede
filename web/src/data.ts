// Data layer. Source order: live API on the same origin, then static files under data/ (results/site copied into
// dist/data), then nothing. The in-browser mock is used only when forced with ?mock=1. There is no silent fallback
// to mock data: a missing file is reported as missing and the page shows an empty state.
import type { Meta, Episode, EpisodeSummary, SessionRow, ArenaState } from './types'

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

// ------------------------------------------------------------------ live-only endpoints
export async function sessions(): Promise<SessionRow[] | null> {
  const i = await init()
  if (i.source === 'mock') return (await mock()).mockSessions()
  if (i.source !== 'live') return null
  try { const d = await getJSON<SessionRow[]>('/api/sessions', 4000); return Array.isArray(d) ? d : [] } catch { return null }
}

async function post<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  if (!r.ok) {
    let msg = `HTTP ${r.status}`
    try { const j = await r.json(); if (j?.detail) msg = String(j.detail) } catch { /* ignore */ }
    throw new Error(msg)
  }
  return r.json() as Promise<T>
}
export async function arenaStart(scenario: string, agent_name: string): Promise<ArenaState> {
  const i = await init()
  if (i.source === 'mock') return (await mock()).mockArenaStart(scenario)
  return post<ArenaState>('/api/arena/start', { scenario, agent_name })
}
export async function arenaAct(id: string, target: number | null): Promise<ArenaState> {
  const i = await init()
  if (i.source === 'mock') return (await mock()).mockArenaAct(id, target)
  return post<ArenaState>(`/api/arena/${encodeURIComponent(id)}/act`, target === null ? { advance: true } : { target_exposure_x: target, advance: true })
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
export const mcpUrl = () => `${displayOrigin()}/mcp/`
