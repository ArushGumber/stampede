// Main-thread side of the in-browser simulator. The worker starts on first use; the first run downloads Pyodide,
// numpy and the simulator bundle (about 10 MB, cached by the browser afterwards).
import type { Req, Res } from './worker'

export type SimStatus = 'idle' | 'loading' | 'ready' | 'failed'
let status: SimStatus = 'idle'
const listeners = new Set<(s: SimStatus) => void>()
const setStatus = (s: SimStatus) => { status = s; listeners.forEach(f => f(s)) }
export const simStatus = () => status
export function onSimStatus(f: (s: SimStatus) => void): () => void { listeners.add(f); return () => { listeners.delete(f) } }

export class BrowserSimError extends Error { constructor(msg: string, public status?: number) { super(msg) } }

let worker: Worker | null = null
let nextId = 1
const pending = new Map<number, { ok: (s: string) => void; fail: (e: BrowserSimError) => void }>()
let booted: Promise<void> | null = null

type WithoutId<T> = T extends unknown ? Omit<T, 'id'> : never
function send(msg: WithoutId<Req>, ms: number): Promise<string> {
  if (!worker) {
    worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
    worker.onmessage = (ev: MessageEvent<Res>) => {
      const r = ev.data, h = pending.get(r.id)
      if (!h) return
      pending.delete(r.id)
      if (r.ok) h.ok(r.json); else h.fail(new BrowserSimError(r.error, r.status))
    }
    worker.onerror = ev => {
      const e = new BrowserSimError(ev.message || 'simulator worker failed')
      pending.forEach(h => h.fail(e)); pending.clear()
    }
  }
  const id = nextId++
  return new Promise<string>((ok, fail) => {
    const to = setTimeout(() => { if (pending.delete(id)) fail(new BrowserSimError('the in-browser simulator timed out')) }, ms)
    pending.set(id, { ok: s => { clearTimeout(to); ok(s) }, fail: e => { clearTimeout(to); fail(e) } })
    worker!.postMessage({ ...msg, id } as Req)
  })
}

/** Load Pyodide and the simulator once. Safe to call repeatedly. */
export function bootSim(): Promise<void> {
  if (!booted) {
    setStatus('loading')
    const bundle = new URL('py/stampede-sim.zip', document.baseURI).href
    booted = send({ op: 'boot', bundle }, 180000).then(() => setStatus('ready'), e => { booted = null; setStatus('failed'); throw e })
  }
  return booted
}

const parse = <T>(s: string): T => JSON.parse(s) as T

export async function simulateInBrowser<T>(params: Record<string, unknown>): Promise<T> {
  await bootSim()
  return parse<T>(await send({ op: 'simulate', params }, 60000))
}
export async function arenaStartInBrowser<T>(body: Record<string, unknown>): Promise<T> {
  await bootSim()
  return parse<T>(await send({ op: 'arenaStart', body }, 60000))
}
export async function arenaActInBrowser<T>(sid: string, body: Record<string, unknown>): Promise<T> {
  await bootSim()
  return parse<T>(await send({ op: 'arenaAct', sid, body }, 60000))
}
