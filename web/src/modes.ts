// URL modes and the small global hook used by capture scripts.
//   ?shot=1   screenshot mode (no chrome, replay jumps to its last frame, compact Arena at 1200x720)
//   ?tour=1   automatic walkthrough with captions for the demo video
// Parameters may sit in the query before the hash (canonical: /?shot=1#/arena/<id>) or after the route in the hash
// (#/arena/<id>?shot=1); both are read.

export function params(): URLSearchParams {
  const p = new URLSearchParams(location.search)
  const h = location.hash.indexOf('?')
  if (h >= 0) new URLSearchParams(location.hash.slice(h + 1)).forEach((v, k) => p.set(k, v))
  return p
}
export const isShot = () => params().get('shot') === '1'
export const isTour = () => params().get('tour') === '1'

export interface StampedeApi {
  seek?: (t: number) => void
  play?: (speed?: number) => void
  arena?: { episodeId: string; t: number; lastT: number; shockT: number | null; ready: boolean; live?: boolean }
  tourDone?: boolean
  tourStep?: string
  [k: string]: unknown
}
declare global { interface Window { __stampede: StampedeApi } }
export const api: StampedeApi = (window.__stampede ??= {})

// Minimal event bus so the tour can drive pages without coupling to their internals.
type Fn = (v: unknown) => void
const subs = new Map<string, Set<Fn>>()
export function on(type: string, f: Fn): () => void { if (!subs.has(type)) subs.set(type, new Set()); subs.get(type)!.add(f); return () => { subs.get(type)?.delete(f) } }
export function emit(type: string, v?: unknown) { subs.get(type)?.forEach(f => f(v)) }
