"""Stampede web server: precomputed results, recorded episodes, the arena over MCP, and the static site.

    STAMPEDE_SITE_DATA=results/site uvicorn stampede.server.app:app --port 8765

Endpoints follow docs/CONTRACTS.md section 5. No GPU and no free-text model calls are exposed:
/api/simulate runs a headline episode and its no-headline twin from response tables on CPU, /api/run returns
the nearest recorded episode, and /api/decide serves precomputed decisions. The arena (/mcp and /api/arena/*)
also runs on CPU with response-table agents.
"""
from __future__ import annotations

import asyncio
import contextlib
import gzip
import json
import math
import os
import time
from collections import defaultdict, deque
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SITE_DATA = os.path.abspath(os.environ.get("STAMPEDE_SITE_DATA", os.path.join(ROOT, "results", "site")))
WEB_DIST = os.path.abspath(os.environ.get("STAMPEDE_WEB_DIST", os.path.join(ROOT, "web", "dist")))
EPISODE_DIRS = [os.path.join(SITE_DATA, "episodes")]


def _load_json(path: str, default=None):
    try:
        if path.endswith(".gz"):
            with gzip.open(path, "rt") as f:
                return json.load(f)
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


class Store:
    """Reads the exported site data; reloads when files change so new results appear without a restart."""

    def __init__(self):
        self._mtime, self._index, self._meta = 0.0, [], {}

    def _refresh(self):
        idx_path = os.path.join(SITE_DATA, "episodes", "index.json")
        m = max([os.path.getmtime(p) for p in (idx_path, os.path.join(SITE_DATA, "meta.json")) if os.path.exists(p)] or [0])
        if m > self._mtime:
            self._index = _load_json(idx_path, []) or []
            self._meta = _load_json(os.path.join(SITE_DATA, "meta.json"), {}) or {}
            self._mtime = m

    def meta(self) -> dict:
        self._refresh()
        return dict(self._meta)

    def index(self) -> list:
        self._refresh()
        return self._index

    def episode(self, eid: str) -> Optional[dict]:
        safe = "".join(c for c in eid if c.isalnum() or c in "_-")
        if safe.startswith("live_"):
            return SIM_CACHE.episode(safe)
        for d in EPISODE_DIRS:
            for name in (safe + ".json", safe + ".json.gz"):
                p = os.path.join(d, name)
                if os.path.exists(p):
                    return _load_json(p)
        return None

    def result(self, name: str) -> Any:
        safe = "".join(c for c in name if c.isalnum() or c in "_-")
        return _load_json(os.path.join(SITE_DATA, "results", safe + ".json"))


STORE = Store()


# ------------------------------------------------------------------------------------------ rate limit
class RateLimiter:
    def __init__(self, per_minute: int = 120):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str):
        now = time.time()
        q = self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            raise HTTPException(429, "rate limit; try again in a minute")
        q.append(now)


LIMIT = RateLimiter(int(os.environ.get("STAMPEDE_RATE_PER_MIN", "240")))
SIM_LIMIT = RateLimiter(int(os.environ.get("STAMPEDE_SIM_PER_MIN", "40")))


# ------------------------------------------------------------------------------------------ live simulation
from collections import OrderedDict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import multiprocessing  # noqa: E402

from stampede.sim import live as LIVE  # noqa: E402


class SimCache:
    """Small LRU of finished runs, keyed by normalised parameters; runs are also reachable by episode id."""

    def __init__(self, size: int = 256):
        self.size, self.by_key, self.by_id = size, OrderedDict(), {}

    def get(self, key: str) -> Optional[dict]:
        r = self.by_key.get(key)
        if r is not None:
            self.by_key.move_to_end(key)
        return r

    def put(self, key: str, r: dict):
        self.by_key[key] = r
        self.by_id[r["episode"]["id"]] = key
        self.by_key.move_to_end(key)
        while len(self.by_key) > self.size:
            _, old = self.by_key.popitem(last=False)
            self.by_id.pop(old["episode"]["id"], None)

    def episode(self, eid: str) -> Optional[dict]:
        key = self.by_id.get(eid)
        r = self.by_key.get(key) if key else None
        return r["episode"] if r else None


SIM_CACHE = SimCache(int(os.environ.get("STAMPEDE_SIM_CACHE", "256")))
SIM_WORKERS = max(1, int(os.environ.get("STAMPEDE_SIM_WORKERS", "2")))
SIM_MAX_WAITING = int(os.environ.get("STAMPEDE_SIM_MAX_WAITING", "8"))
_sim_pool: Optional[ProcessPoolExecutor] = None
_sim_sem: Optional[asyncio.Semaphore] = None
_sim_waiting = 0


def _pool() -> ProcessPoolExecutor:
    global _sim_pool
    if _sim_pool is None:
        _sim_pool = ProcessPoolExecutor(max_workers=SIM_WORKERS, mp_context=multiprocessing.get_context("spawn"))
    return _sim_pool


async def _run_sim(params: dict) -> dict:
    """Run in a worker process; if the pool cannot start or has died, reset it and run in a thread."""
    global _sim_pool
    from concurrent.futures.process import BrokenProcessPool

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(_pool(), LIVE.simulate, params)
    except (BrokenProcessPool, OSError):
        with contextlib.suppress(Exception):
            _sim_pool.shutdown(wait=False, cancel_futures=True)
        _sim_pool = None
        return await asyncio.to_thread(LIVE.simulate, params)


def _client(req: Request) -> str:
    return req.headers.get("cf-connecting-ip") or req.headers.get("x-forwarded-for", "").split(",")[0] or (req.client.host if req.client else "?")


# ------------------------------------------------------------------------------------------ MCP arena
from stampede.mcp_server.arena import REGISTRY, SCENARIOS  # noqa: E402
from stampede.mcp_server.server import build_mcp  # noqa: E402

_mcp = build_mcp()
_mcp_app = _mcp.streamable_http_app(streamable_http_path="/", stateless_http=False, host="0.0.0.0")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        sm = getattr(_mcp, "session_manager", None)
        if sm is not None and hasattr(sm, "run"):
            await stack.enter_async_context(sm.run())
        yield
    if _sim_pool is not None:
        _sim_pool.shutdown(wait=True, cancel_futures=True)


app = FastAPI(title="Stampede", lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["mcp-session-id"])
app.mount("/mcp", _mcp_app)


# ------------------------------------------------------------------------------------------ API
@app.get("/api/meta")
def meta():
    m = STORE.meta()
    m.setdefault("models", [])
    m.setdefault("traits", ["risk", "integrity", "credulity"])
    m.setdefault("regimes", ["equity", "crypto"])
    m.setdefault("safeguards", [])
    m.setdefault("headlines", [])
    m["live"] = False
    m["simulate"] = {"path": "/api/simulate", "populations": list(LIVE.POP_SPECS), "safeguards": list(LIVE.SAFEGUARDS),
                     "hcodes": list(LIVE.HKINDS), "regimes": list(LIVE.SYM), "capital_range": list(LIVE.CAPITAL_RANGE),
                     "resilience": list(LIVE.RESILIENCE), "n_agents": LIVE.N_AGENTS}
    m["arena_scenarios"] = list(SCENARIOS)
    m["mcp_path"] = "/mcp"
    return m


@app.get("/api/results/{name}")
def results(name: str):
    r = STORE.result(name)
    if r is None:
        raise HTTPException(404, f"no result set {name}")
    return r


def _num(x, d=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


@app.get("/api/episodes")
def episodes(regime: Optional[str] = None, headline: Optional[str] = None, share_reckless: Optional[float] = None,
             alpha: Optional[float] = None, safeguard: Optional[str] = None, limit: int = 200):
    rows = STORE.index()
    out = []
    for r in rows:
        c = r.get("config", {})
        if regime and c.get("regime") != regime:
            continue
        if headline and c.get("headline_kind", c.get("headline")) != headline:
            continue
        if safeguard and (c.get("safeguard") or "none") != safeguard:
            continue
        out.append(r)
    if share_reckless is not None or alpha is not None:
        def dist(r):
            c = r.get("config", {})
            d = 0.0
            if share_reckless is not None:
                d += abs((_num(c.get("share_reckless"), 0) or 0) - share_reckless)
            if alpha is not None:
                d += abs((_num(c.get("alpha"), 0) or 0) - alpha) / 4
            return d
        out.sort(key=dist)
    return out[: max(1, min(limit, 1000))]


@app.get("/api/episodes/{eid}")
def episode(eid: str):
    ep = STORE.episode(eid)
    if ep is None:
        raise HTTPException(404, "episode not found")
    return ep


def _nearest(body: dict) -> Optional[dict]:
    rows = STORE.index()
    if not rows:
        return None
    sg = body.get("safeguards") or []
    sg_key = sg[0] if isinstance(sg, list) and sg else (sg if isinstance(sg, str) else "none")
    hk = (body.get("headline") or {}).get("kind")

    def score(r):
        c = r.get("config", {})
        s = 0.0
        s += 0 if c.get("regime") == body.get("regime") else 10
        s += 0 if (hk is None or c.get("headline_kind") == hk) else 5
        s += 0 if (c.get("safeguard") or "none") == sg_key else 3
        for k, w in (("adoption", 4.0), ("share_reckless", 4.0), ("share_credulous", 3.0), ("share_manipulative", 2.0), ("alpha", 1.0)):
            a, b = _num(c.get(k)), _num(body.get(k))
            if a is not None and b is not None:
                s += w * abs(a - b)
        if body.get("model") and c.get("model") and c.get("model") != body.get("model"):
            s += 1
        return s

    return min(rows, key=score)


@app.post("/api/run")
async def run(req: Request):
    LIMIT.check(_client(req))
    body = await req.json()
    best = _nearest(body)
    if best is None:
        raise HTTPException(503, "no recorded episodes yet")
    return {"episode_id": best["id"], "mode": "replay", "matched_config": best.get("config")}


@app.post("/api/simulate")
async def simulate(req: Request):
    """Run a headline episode and its same-seed no-headline twin from response tables, on CPU."""
    global _sim_sem, _sim_waiting
    LIMIT.check(_client(req))
    try:
        body = await req.json()
    except ValueError:
        raise HTTPException(400, "body must be JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be a JSON object")
    try:
        params = LIVE.normalise(body)
    except LIVE.BadRequest as e:
        raise HTTPException(400, str(e))
    key = json.dumps(params, sort_keys=True)
    t0 = time.perf_counter()
    hit = SIM_CACHE.get(key)
    if hit is None:
        SIM_LIMIT.check(_client(req))
        if _sim_sem is None:
            _sim_sem = asyncio.Semaphore(SIM_WORKERS)
        if _sim_waiting >= SIM_MAX_WAITING:
            raise HTTPException(429, "the simulator is busy; try again in a few seconds")
        _sim_waiting += 1
        try:
            async with _sim_sem:
                hit = SIM_CACHE.get(key)
                if hit is None:
                    hit = await _run_sim(params)
                    SIM_CACHE.put(key, hit)
        finally:
            _sim_waiting -= 1
        cached = False
    else:
        cached = True
    out = dict(hit)
    out["params"] = params
    out["cached"] = cached
    out["elapsed_ms"] = int(round((time.perf_counter() - t0) * 1000))
    return JSONResponse(out)


@app.websocket("/ws/episodes/{eid}")
async def ws_episode(ws: WebSocket, eid: str):
    await ws.accept()
    ep = STORE.episode(eid)
    if ep is None:
        await ws.send_json({"error": "episode not found"})
        await ws.close()
        return
    try:
        await ws.send_json({"header": True, "id": ep["id"], "config": ep.get("config"), "agents": ep.get("agents")})
        delay = float(ws.query_params.get("delay", "0.12"))
        for fr in ep.get("frames", []):
            await ws.send_json(fr)
            await asyncio.sleep(max(0.0, min(delay, 2.0)))
        await ws.send_json({"done": True, "metrics": ep.get("metrics", {})})
    except WebSocketDisconnect:
        return
    with contextlib.suppress(Exception):
        await ws.close()


@app.post("/api/decide")
async def decide(req: Request):
    """Serve a precomputed decision for a fixed vignette and dial setting (nearest grid point)."""
    LIMIT.check(_client(req))
    body = await req.json()
    table = STORE.result("decide") or {}
    vid = body.get("vignette_id") or ""
    model = body.get("model") or table.get("default_model")
    alphas = body.get("alphas") or {}
    if body.get("text") and not vid:
        raise HTTPException(400, "free-text decisions are disabled on the public demo; choose a vignette")
    vig = (table.get("vignettes") or {}).get(vid)
    if vig is None:
        raise HTTPException(404, f"unknown vignette {vid}")
    recs = [r for r in table.get("records", []) if r["vignette_id"] == vid and (not model or r["model"] == model)]
    if not recs:
        raise HTTPException(404, "no precomputed decisions for that vignette and model")

    def d(r):
        return sum((float(r["alphas"].get(t, 0.0)) - float(alphas.get(t, 0.0))) ** 2 for t in set(r["alphas"]) | set(alphas))

    r = min(recs, key=d)
    return {"options": vig["options"], "probs": r["probs"], "reason": r.get("reason", ""), "matched_alphas": r["alphas"],
            "model": r["model"], "precomputed": True}


@app.get("/api/sessions")
def sessions():
    return REGISTRY.board()


@app.post("/api/arena/start")
async def arena_start(req: Request):
    """Browser version of the arena for attendees without an MCP client."""
    LIMIT.check(_client(req))
    body = await req.json()
    s = REGISTRY.create(body.get("scenario", "fake-equity"), body.get("agent_name", "human"), body.get("seed"))
    return {"session_id": s.id, "market": s.market(), "portfolio": s.portfolio(), "news": s.news()}


@app.post("/api/arena/{sid}/act")
async def arena_act(sid: str, req: Request):
    LIMIT.check(_client(req))
    body = await req.json()
    try:
        s = REGISTRY.get(sid)
    except KeyError as e:
        raise HTTPException(404, str(e))
    if body.get("target_exposure_x") is not None:
        s.set_target_exposure(float(body["target_exposure_x"]))
    step = s.advance() if body.get("advance", True) else {}
    return {"step": step, "market": s.market(), "portfolio": s.portfolio(), "news": s.news()}


@app.get("/api/health")
def health():
    return {"ok": True, "episodes": len(STORE.index()), "sessions": len(REGISTRY.sessions), "site_data": SITE_DATA}


# ------------------------------------------------------------------------------------------ static
if os.path.isdir(SITE_DATA):
    app.mount("/data", StaticFiles(directory=SITE_DATA), name="data")
if os.path.isdir(WEB_DIST):
    @app.get("/")
    def index_html():
        return FileResponse(os.path.join(WEB_DIST, "index.html"))

    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
