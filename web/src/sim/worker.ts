/// <reference lib="webworker" />
// Runs the market simulator in the browser: Pyodide (CPython compiled to WebAssembly) plus numpy, with the same
// Python modules the server uses (stampede.sim.live for markets, stampede.mcp_server.arena for the browser arena).

export const PYODIDE_VERSION = '0.27.7'
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`

export type Req =
  | { id: number; op: 'boot'; bundle: string }
  | { id: number; op: 'simulate'; params: Record<string, unknown> }
  | { id: number; op: 'arenaStart'; body: Record<string, unknown> }
  | { id: number; op: 'arenaAct'; sid: string; body: Record<string, unknown> }
export type Res = { id: number; ok: true; json: string } | { id: number; ok: false; error: string; status?: number }

const BRIDGE = `
import json, sys, time, zipfile
zipfile.ZipFile('/tmp/stampede-sim.zip').extractall('/home/pyodide/app')
sys.path.insert(0, '/home/pyodide/app')
from stampede.sim import live
from stampede.mcp_server.arena import REGISTRY

def _err(msg, status):
    return json.dumps({"__error__": msg, "status": status})

def run_simulate(params_json):
    t0 = time.perf_counter()
    try:
        p = live.normalise(json.loads(params_json))
    except live.BadRequest as e:
        return _err(str(e), 400)
    out = live.simulate(p)
    out["params"] = p
    out["cached"] = False
    out["elapsed_ms"] = int(round((time.perf_counter() - t0) * 1000))
    return json.dumps(out)

def _view(s, step=None):
    d = {"market": s.market(), "portfolio": s.portfolio(), "news": s.news()}
    if step is not None:
        d["step"] = step
    return json.dumps(live.clean(d))

def arena_start(body_json):
    b = json.loads(body_json)
    s = REGISTRY.create(b.get("scenario", "fake-equity"), b.get("agent_name", "human"), b.get("seed"))
    d = json.loads(_view(s))
    d["session_id"] = s.id
    return json.dumps(d)

def arena_act(sid, body_json):
    b = json.loads(body_json)
    try:
        s = REGISTRY.get(sid)
    except KeyError as e:
        return _err(str(e), 404)
    if b.get("target_exposure_x") is not None:
        s.set_target_exposure(float(b["target_exposure_x"]))
    step = s.advance() if b.get("advance", True) else {}
    return _view(s, step)
`

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Py = any
let py: Promise<Py> | null = null

async function boot(bundle: string): Promise<Py> {
  const mod = await import(/* @vite-ignore */ `${PYODIDE_URL}pyodide.mjs`)
  const p = await mod.loadPyodide({ indexURL: PYODIDE_URL })
  const [, buf] = await Promise.all([
    p.loadPackage('numpy', { messageCallback: () => {} }),
    fetch(bundle).then(r => { if (!r.ok) throw new Error(`simulator bundle: HTTP ${r.status}`); return r.arrayBuffer() }),
  ])
  p.FS.writeFile('/tmp/stampede-sim.zip', new Uint8Array(buf))
  p.runPython(BRIDGE)
  return p
}

function call(p: Py, fn: string, ...args: string[]): string {
  const f = p.globals.get(fn)
  try { return f(...args) as string } finally { f.destroy?.() }
}

self.onmessage = async (ev: MessageEvent<Req>) => {
  const m = ev.data
  const reply = (r: Res) => (self as unknown as Worker).postMessage(r)
  try {
    if (m.op === 'boot') { py ??= boot(m.bundle); py.catch(() => { py = null }); await py; reply({ id: m.id, ok: true, json: 'true' }); return }
    if (!py) throw new Error('simulator not loaded')
    const p = await py
    let out: string
    if (m.op === 'simulate') out = call(p, 'run_simulate', JSON.stringify(m.params))
    else if (m.op === 'arenaStart') out = call(p, 'arena_start', JSON.stringify(m.body))
    else out = call(p, 'arena_act', m.sid, JSON.stringify(m.body))
    const err = out.startsWith('{"__error__"') ? JSON.parse(out) : null
    reply(err ? { id: m.id, ok: false, error: String(err.__error__), status: err.status } : { id: m.id, ok: true, json: out })
  } catch (e) {
    reply({ id: m.id, ok: false, error: e instanceof Error ? e.message : String(e) })
  }
}
