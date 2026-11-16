"""Minimal MCP client over streamable HTTP (JSON responses) for scripted tests."""
from __future__ import annotations

import itertools
import json
import time
import urllib.request

_ids = itertools.count(1)
HDRS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": "2025-06-18", "User-Agent": "stampede-test"}


def rpc(url: str, method: str, params: dict | None = None, timeout: float = 60) -> tuple[dict, float]:
    body = {"jsonrpc": "2.0", "id": next(_ids), "method": method}
    if params is not None:
        body["params"] = params
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=HDRS, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        ctype = r.headers.get("content-type", "")
    dt = time.perf_counter() - t0
    if "text/event-stream" in ctype:
        data = [ln[5:].strip() for ln in raw.decode().splitlines() if ln.startswith("data:")]
        return json.loads(data[-1]), dt
    return json.loads(raw), dt


def initialize(url: str) -> dict:
    r, _ = rpc(url, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                    "clientInfo": {"name": "stampede-test", "version": "0"}})
    return r


def call(url: str, tool: str, **args) -> dict:
    r, dt = rpc(url, "tools/call", {"name": tool, "arguments": args})
    if "error" in r:
        raise RuntimeError(f"{tool}: {r['error']}")
    res = r["result"]
    if res.get("isError"):
        raise RuntimeError(f"{tool}: {res['content'][0].get('text')}")
    out = res.get("structuredContent")
    if out is None:
        out = json.loads(res["content"][0]["text"])
    if isinstance(out, dict) and set(out) == {"result"}:
        out = out["result"]
    call.last_dt = dt
    return out


call.last_dt = 0.0
