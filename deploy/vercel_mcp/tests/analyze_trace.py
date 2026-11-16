"""Checks a Claude Code stream-json trace of an arena session played against the hosted endpoint.

    python tests/analyze_trace.py run.jsonl [more.jsonl ...]

For each trace: counts tool calls, how often the agent passed a session_id other than the newest one it
had been given, and tool errors; then decodes the agent's final session_id into its action log, runs the
same actions on the lab arena code in-process, and compares that report card with the one the hosted
endpoint returned to the agent.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import hosted  # noqa: E402
from parity import compare, real, run_lab_inprocess  # noqa: E402


def load(path: str):
    calls, results, final = {}, {}, None
    order = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line)
        if ev.get("type") == "result":
            final = ev
        msg = ev.get("message") or {}
        for c in msg.get("content") or [] if isinstance(msg.get("content"), list) else []:
            if c.get("type") == "tool_use":
                calls[c["id"]] = (c["name"].split("__")[-1], c.get("input") or {})
                order.append(c["id"])
            elif c.get("type") == "tool_result":
                content = c.get("content")
                text = content if isinstance(content, str) else "".join(x.get("text", "") for x in content or [])
                results[c["tool_use_id"]] = (bool(c.get("is_error")), text)
    return [(calls[i][0], calls[i][1], results.get(i)) for i in order], final


def parse_result(text: str):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def analyse(path: str) -> dict:
    steps, final = load(path)
    newest, stale, errors, sids, report, label, scenario = None, 0, [], [], None, None, None
    for name, args, res in steps:
        sid = args.get("session_id")
        if name == "start_session":
            label, scenario = args.get("agent_name"), args.get("scenario")
        elif sid is not None and newest is not None and sid != newest:
            stale += 1
        is_err, text = res if res else (True, "no result")
        if is_err:
            errors.append(f"{name}: {text[:160]}")
            continue
        out = parse_result(text)
        if isinstance(out, dict) and out.get("session_id"):
            newest = out["session_id"]
            sids.append(newest)
        if name == "get_report" and isinstance(out, dict):
            report = {k: v for k, v in out.items() if k != "session_id"}
        if name == "wait_for_next_tick" and isinstance(out, dict) and out.get("report"):
            report = report or out["report"]
    row = {"trace": os.path.basename(path), "tool_calls": len(steps), "stale_session_ids_passed": stale,
           "tool_errors": errors, "final_sid_len": len(newest or ""), "reached_report": report is not None,
           "result_subtype": (final or {}).get("subtype"), "turns": (final or {}).get("num_turns"),
           "duration_s": round(((final or {}).get("duration_ms") or 0) / 1000), "cost_usd": (final or {}).get("total_cost_usd")}
    if newest:
        _, tok = hosted.parse_token(newest)
        row["minutes"] = tok.ticks
        row["actions"] = len(tok.actions)
        if report is not None and tok.ticks == hosted.N_TICKS:
            n = tok.ticks + (1 if any(a[1] == tok.ticks for a in tok.actions) else 0)
            script = [[] for _ in range(n)]
            for a in tok.actions:
                if a[0] == "x":
                    script[a[1]].append(["x", a[2]])
                elif a[0] == "o":
                    _, t, side, limit, unit, amount, price = a
                    script[t].append(["o", side, unit, amount, "limit" if limit else "market", price])
                else:
                    script[a[1]].append(["c", a[2]])
            ref = run_lab_inprocess(tok.scenario, tok.seed, tok.label, script)
            d = compare(report, ref["report"])
            row["report_vs_lab_code"] = d
            row["lab_match"] = "exact" if not d else ("last-bit floats only" if not real(d) else "DIFFERENT")
            row["your_pnl_pct"] = report.get("your_pnl_pct")
            row["seed"] = tok.seed
    return row


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(analyse(p), default=str))
