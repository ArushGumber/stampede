"""Stampede arena MCP endpoint for Vercel (Python runtime, Fluid compute).

    POST /mcp/      MCP over streamable HTTP, stateless, JSON responses
    GET  /          what this is and how to connect
    GET  /api/health

Every MCP request is served by its own short-lived session manager, so nothing depends on ASGI lifespan
events or on the event loop surviving between requests. Arena sessions are rebuilt from the session_id
the agent passes (see hosted.py), so they survive cold starts and moves between instances.
"""
from __future__ import annotations

import time

T_START = time.time()

import os  # noqa: E402
import secrets  # noqa: E402
import sys  # noqa: E402

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hosted  # noqa: E402
from stampede.mcp_server.arena import PUBLIC_SCENARIOS as SCENARIOS  # noqa: E402

T_IMPORT = time.time()
INSTANCE = secrets.token_hex(3)
COUNTS = {"mcp": 0, "lifespan": 0}

_mcp = hosted.build_mcp()
_server = _mcp._lowlevel_server
hosted.surrogates()  # load the response tables while the instance starts
INIT_S = round(time.time() - T_START, 3)


def _process_age() -> float | None:
    """Seconds since this process started (Linux only), to tell cold instances from warm ones."""
    try:
        with open("/proc/self/stat") as f:
            start = int(f.read().rsplit(")", 1)[1].split()[19]) / os.sysconf("SC_CLK_TCK")
        with open("/proc/uptime") as f:
            return round(float(f.read().split()[0]) - start, 2)
    except (OSError, ValueError, IndexError):
        return None


async def handle_mcp(scope, receive, send):
    COUNTS["mcp"] += 1
    manager = StreamableHTTPSessionManager(app=_server, stateless=True, json_response=True)
    async with manager.run():
        await manager.handle_request(scope, receive, send)


class MCPRoute:
    """Sends /mcp and /mcp/ to the MCP handler without a redirect; everything else goes to the API."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path", "").rstrip("/") == "/mcp":
            if scope.get("method") != "POST":
                # No server-to-client stream and no session to delete: per the spec, answer 405 so clients
                # do not hold a long-lived GET open against a serverless function.
                await send({"type": "http.response.start", "status": 405,
                            "headers": [(b"allow", b"POST"), (b"content-type", b"application/json")]})
                await send({"type": "http.response.body",
                            "body": b'{"error":"this endpoint is stateless; use POST"}'})
                return
            await handle_mcp(scope, receive, send)
            return
        await self.app(scope, receive, send)


def lifespan(app):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def run(app):
        COUNTS["lifespan"] += 1
        yield

    return run(app)


app = FastAPI(title="Stampede arena MCP", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.add_middleware(MCPRoute)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["mcp-session-id"])


def _base(req: Request) -> str:
    host = req.headers.get("x-forwarded-host") or req.headers.get("host") or "localhost"
    return f"https://{host}"


@app.get("/api/health")
def health():
    return {"ok": True, "instance": INSTANCE, "uptime_s": round(time.time() - T_IMPORT, 1), "process_age_s": _process_age(),
            "app_import_s": INIT_S, "mcp_requests": COUNTS["mcp"],
            "lifespan_started": COUNTS["lifespan"] > 0, "cached_sessions": len(hosted.STORE.items),
            "replays": hosted.STORE.replays, **hosted.STORE.counts, "python": sys.version.split()[0]}


@app.get("/api/sessions")
def sessions():
    """Sessions this instance has served recently (each instance keeps its own list)."""
    return {"instance": INSTANCE, "sessions": hosted.STORE.board()[:50]}


@app.get("/", response_class=HTMLResponse)
def index(req: Request):
    url = _base(req) + "/mcp/"
    scen = ", ".join(SCENARIOS)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Stampede arena MCP</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:46rem;margin:2rem auto;padding:0 1rem;background:#fff;color:#222}}
code,pre{{background:#f3f1ec;border-radius:4px;padding:.1rem .3rem}}pre{{padding:.8rem;overflow-x:auto}}
@media (prefers-color-scheme: dark){{body{{background:#1b1a18;color:#e8e6e1}}code,pre{{background:#2a2825}}}}</style></head>
<body><h1>Stampede arena</h1>
<p>A simulated market for testing trading agents. A headline arrives partway through a 40-minute session; it may be
false. Everything is fictional and no real order is ever placed.</p>
<p>Connect an MCP client over streamable HTTP:</p>
<pre>claude mcp add --transport http stampede {url}</pre>
<p>Then ask your agent to start a session and trade every simulated minute until the report card. Scenarios: {scen}.</p>
</body></html>"""


@app.get("/api/info")
def info(req: Request):
    return JSONResponse({"name": "stampede-arena", "mcp": _base(req) + "/mcp/", "transport": "streamable-http",
                         "scenarios": list(SCENARIOS), "tools": ["start_session", "get_market", "get_portfolio", "get_news",
                         "set_target_exposure", "place_order", "cancel_order", "wait_for_next_tick", "get_report"]})
