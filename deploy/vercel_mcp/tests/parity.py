"""Scripted sessions against the hosted MCP endpoint, checked against the lab arena for the same actions.

    python tests/parity.py --url https://host/mcp/ [--url http://127.0.0.1:8799/mcp/] [--lab-http http://127.0.0.1:8765]
        [--runs fake-equity:3:mixed fake-crypto:11:targets ...] [--stale] [--threads 2]

With several --url values the calls rotate between them, so each call lands on a server whose memory is
behind the session and has to rebuild it from the session_id. --stale sends the previous session_id on
every other call, which a warm server should map to its newer state. For each run the report card is
compared with (a) the lab arena code run in-process on the same actions and (b) for target-only scripts,
the running lab server's /api/arena endpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import call, initialize  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PY = os.path.join(REPO, ".venv", "bin", "python")


def plan(kind: str, rng: random.Random, minute: int, price: float) -> list:
    acts = []
    if kind == "targets":
        if minute == 0 or rng.random() < 0.6:
            acts.append(["x", rng.choice([0.0, 0.25, 0.5, 1.0, 1.5, 2.0])])
        return acts
    if kind == "hold":
        return acts
    r = rng.random()
    if r < 0.35:
        acts.append(["x", rng.choice([0.0, 0.5, 1.0, 1.25, 2.0])])
    elif r < 0.55:
        acts.append(["o", rng.choice(["buy", "sell"]), "n", float(rng.choice([2000, 5000, 15000])), "market", None])
    elif r < 0.7:
        acts.append(["o", rng.choice(["buy", "sell"]), "q", round(rng.uniform(5, 80), 3), "market", None])
    elif r < 0.85:
        side = rng.choice(["buy", "sell"])
        lp = round(price * (0.995 if side == "buy" else 1.005), 2)
        acts.append(["o", side, "q", 25.0, "limit", lp])
    elif r < 0.92:
        acts.append(["c", str(rng.randint(1, 4000))])
    if rng.random() < 0.15:
        acts.append(["x", 1.0])
    return acts


def run_hosted(urls: list, scenario: str, seed: int, label: str, kind: str, stale: bool) -> dict:
    rng = random.Random(f"{scenario}:{seed}:{kind}")
    k = [0]

    def url():
        k[0] += 1
        return urls[k[0] % len(urls)]

    st = call(url(), "start_session", agent_name=label, scenario=scenario, seed=seed)
    sid, prev, script, lat = st["session_id"], None, [], []
    calls = 0
    while True:
        use = prev if (stale and prev and calls % 2 == 1) else sid
        m = call(url(), "get_market", session_id=use)
        calls += 1
        lat.append(call.last_dt)
        sid = m["session_id"]
        acts = plan(kind, rng, m["minute"], m["price"])
        script.append(acts)
        for a in acts:
            if a[0] == "x":
                r = call(url(), "set_target_exposure", session_id=sid, target_exposure_x=a[1])
            elif a[0] == "o":
                args = dict(session_id=sid, side=a[1], order_type=a[4])
                args["quantity" if a[2] == "q" else "notional"] = a[3]
                if a[5] is not None:
                    args["limit_price"] = a[5]
                r = call(url(), "place_order", **args)
            else:
                r = call(url(), "cancel_order", session_id=sid, order_id=a[1])
            lat.append(call.last_dt)
            prev, sid = sid, r["session_id"]
        use = prev if (stale and prev and calls % 2 == 1) else sid
        w = call(url(), "wait_for_next_tick", session_id=use)
        calls += 1
        lat.append(call.last_dt)
        prev, sid = sid, w["session_id"]
        if w.get("done"):
            break
    rep = call(url(), "get_report", session_id=sid)
    port = call(url(), "get_portfolio", session_id=sid)
    rep.pop("session_id", None)
    port.pop("session_id", None)
    return {"report": rep, "portfolio": port, "script": script, "sid": sid, "latency": lat}


def run_lab_inprocess(scenario: str, seed: int, label: str, script: list) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(script, f)
    out = subprocess.run([PY, "deploy/vercel_mcp/tests/lab_reference.py", scenario, str(seed), label, f.name],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    os.unlink(f.name)
    return json.loads(out.strip().splitlines()[-1])


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def run_lab_http(base: str, scenario: str, seed: int, label: str, script: list) -> dict:
    s = _post(base + "/api/arena/start", {"scenario": scenario, "agent_name": label, "seed": seed})
    sid = s["session_id"]
    last = None
    for acts in script:
        xs = [a[1] for a in acts if a[0] == "x"]
        assert len(acts) == len(xs) and len(xs) <= 1, "lab http supports one target per minute only"
        body = {"advance": True}
        if xs:
            body["target_exposure_x"] = xs[0]
        last = _post(base + f"/api/arena/{sid}/act", body)
    return {"report": last["step"].get("report"), "portfolio": last["portfolio"]}


def compare(a: dict, b: dict) -> list:
    """Fields that differ. A float that differs only below 1e-12 relative is tagged "float-ulp"."""
    diffs = []
    for key in sorted(set(a) | set(b)):
        x, y = a.get(key), b.get(key)
        if x == y:
            continue
        tiny = isinstance(x, float) and isinstance(y, float) and abs(x - y) <= 1e-12 * max(abs(x), abs(y), 1e-300)
        diffs.append((key, x, y, "float-ulp" if tiny else "DIFFERENT"))
    return diffs


def real(diffs: list) -> list:
    return [d for d in diffs if d[3] != "float-ulp"]


def one(args, spec, results):
    scenario, seed, kind = spec.split(":")
    seed = int(seed)
    label = f"parity-{kind}-{seed}"
    t0 = time.perf_counter()
    h = run_hosted(args.url, scenario, seed, label, kind, args.stale)
    wall = time.perf_counter() - t0
    ref = run_lab_inprocess(scenario, seed, label, h["script"])
    row = {"run": spec, "wall_s": round(wall, 1), "calls": len(h["latency"]) + 3,
           "median_call_ms": round(sorted(h["latency"])[len(h["latency"]) // 2] * 1000),
           "max_call_ms": round(max(h["latency"]) * 1000),
           "actions": sum(len(m) for m in h["script"]), "sid_len": len(h["sid"]),
           "report_vs_lab_code": compare(h["report"], ref["report"]),
           "portfolio_vs_lab_code": compare(h["portfolio"], ref["portfolio"]),
           "your_pnl_pct": h["report"]["your_pnl_pct"], "orders": h["report"]["orders"]}
    if args.lab_http and kind in ("targets", "hold"):
        lab = run_lab_http(args.lab_http, scenario, seed, label, h["script"])
        row["report_vs_lab_server"] = compare(h["report"], lab["report"])
    results.append(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", action="append", required=True)
    ap.add_argument("--lab-http", default="")
    ap.add_argument("--runs", nargs="+", default=["fake-equity:3:mixed", "fake-crypto:11:targets"])
    ap.add_argument("--stale", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    initialize(args.url[0])
    results, threads = [], []
    for spec in args.runs:
        th = threading.Thread(target=one, args=(args, spec, results))
        threads.append(th)
        th.start()
        if len(threads) >= args.threads:
            for t in threads:
                t.join()
            threads = []
    for t in threads:
        t.join()
    exact = close = True
    for r in results:
        print(json.dumps(r, default=str))
        d = r["report_vs_lab_code"] + r["portfolio_vs_lab_code"] + r.get("report_vs_lab_server", [])
        exact &= not d
        close &= not real(d)
    if len(results) != len(args.runs):
        print("FAILURE: some runs did not finish")
    elif exact:
        print("ALL MATCH EXACTLY")
    elif close:
        print("ALL MATCH (some unrounded floats differ in the last bits only)")
    else:
        print("MISMATCH")


if __name__ == "__main__":
    main()
