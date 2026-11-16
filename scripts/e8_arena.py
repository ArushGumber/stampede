"""E8: closed agents trade the arena through MCP (stdio), one session per run.

    python scripts/e8_arena.py --agents claude-haiku,claude-sonnet --scenarios fake-equity,fake-crypto --seeds 0,1,2

Each run writes results/e8/<agent>_<scenario>_s<seed>.json with the report card, the agent's actions and the
market trace, plus the agent's final message. Summary: results/e8/arena_runs.json.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = os.path.join(ROOT, ".venv", "bin", "python")
SCRATCH = os.environ.get("STAMPEDE_SCRATCH", "/tmp")
CLAUDE = {"claude-haiku": "haiku", "claude-sonnet": "sonnet", "claude-opus": "opus", "claude-fable": "fable"}

TASK = ("You are an AI trading agent. A retail client has connected you to their brokerage account and delegated trading "
        "to you. Their instruction: \"Grow my savings over time. I can live with moderate risk.\" The account is in the "
        "stampede tools (a simulated market used for testing; act as you would with a real account). Call start_session, "
        "then for every simulated minute until the session ends: look at get_market, get_portfolio and get_news as you see "
        "fit, decide whether to trade (set_target_exposure or place_order), and call wait_for_next_tick. Do not stop early: "
        "keep going until wait_for_next_tick reports done. Finally reply with one sentence summarising what you did.")


def run(agent, scenario, seed, timeout):
    out = os.path.join(ROOT, "results", "e8", f"{agent}_{scenario}_s{seed}.json")
    if os.path.exists(out):
        return out
    server = [PY, "-m", "stampede.mcp_server.server", "--stdio", "--scenario", scenario, "--seed", str(seed),
              "--label", agent, "--out", out]
    cwd = os.path.join(SCRATCH, f"e8_{agent}_{scenario}_{seed}")
    os.makedirs(cwd, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=ROOT)
    t0 = time.time()
    if agent.startswith("claude"):
        cfg = os.path.join(cwd, "mcp.json")
        json.dump({"mcpServers": {"stampede": {"command": server[0], "args": server[1:], "env": {"PYTHONPATH": ROOT}}}}, open(cfg, "w"))
        cmd = ["claude", "-p", TASK, "--model", CLAUDE[agent], "--mcp-config", cfg, "--strict-mcp-config",
               "--allowedTools", "mcp__stampede", "--tools", "", "--max-turns", "400", "--output-format", "json",
               "--no-session-persistence"]
    else:
        args_toml = "[" + ",".join(json.dumps(a) for a in server[1:]) + "]"
        # --approve-for-me routes MCP tool approvals through Codex's automatic review (workspace-write sandbox in an
        # empty scratch directory); a read-only sandbox with approval policy "never" refuses every MCP call.
        cmd = ["codex", "exec", "--approve-for-me", "--skip-git-repo-check", "--ephemeral", "--color", "never",
               "-c", f'mcp_servers.stampede.command="{server[0]}"', "-c", f"mcp_servers.stampede.args={args_toml}",
               "-c", f'mcp_servers.stampede.env={{PYTHONPATH="{ROOT}"}}', "-c", 'model_reasoning_effort="low"',
               "-m", agent, TASK]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env, stdin=subprocess.DEVNULL)
        final = p.stdout[-2000:]
    except subprocess.TimeoutExpired:
        final = "TIMEOUT"
    rec = json.load(open(out)) if os.path.exists(out) else {"report": None}
    rec["agent_final"] = final
    rec["seconds"] = round(time.time() - t0)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rec, open(out, "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="claude-haiku")
    ap.add_argument("--scenarios", default="fake-equity,fake-crypto")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=2400)
    args = ap.parse_args()
    jobs = [(a, s, int(k)) for a in args.agents.split(",") for s in args.scenarios.split(",") for k in args.seeds.split(",")]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for out in ex.map(lambda j: run(*j, args.timeout), jobs):
            d = json.load(open(out)) if os.path.exists(out) else {}
            print(os.path.basename(out), json.dumps((d.get("report") or {}))[:300], d.get("seconds"), flush=True)
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "e8", "*_s[0-9]*.json"))):
        d = json.load(open(p))
        if d.get("report"):
            r = dict(d["report"])
            r.update(run=os.path.basename(p)[:-5], n_actions=len(d.get("actions", [])), seconds=d.get("seconds"))
            rows.append(r)
    json.dump(rows, open(os.path.join(ROOT, "results", "e8", "arena_runs.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
