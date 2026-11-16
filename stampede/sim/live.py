"""Run one headline episode and its no-headline twin from response tables, on demand (CPU only).

The settings match the market sweeps (scripts/e4_sweep.py and scripts/run_market_sweeps.sh): 24 agents,
50 ticks, headline at tick 15, correction at tick 30, act probability 0.5, the same headline bank and the
same safeguard settings. With the same seed, a run reproduces the recorded episode for that cell.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SURROGATE_DIR = os.environ.get("STAMPEDE_SURROGATES", os.path.join(ROOT, "results", "surrogates"))

N_AGENTS, N_TICKS, SHOCK_T, CORRECTION_T, ACT_PROB = 24, 50, 15, 30, 0.5
SEVERITY, N_HEADLINES = ("severe", "moderate"), 3
CAPITAL_RANGE = (0.25, 3.0)

G, L = "gemma-4-12b", "llama-3.1-8b"
POP_SPECS = {
    "base": f"{G}_base", "gemma-base": f"{G}_base", "r0": f"{G}_base",
    "risk-1": f"{G}_risk-1", "risk+0.5": f"{G}_risk+0.5", "risk+0.75": f"{G}_risk+0.75",
    "risk+1": f"{G}_risk+1", "risk+1.5": f"{G}_risk+1.5",
    "r25": f"{G}_base:0.75,{G}_risk+1.5:0.25", "r50": f"{G}_base:0.5,{G}_risk+1.5:0.5",
    "r75": f"{G}_base:0.25,{G}_risk+1.5:0.75", "r100": f"{G}_risk+1.5",
    "base_shift": f"{G}_risk-1", "r50_shift": f"{G}_risk-1:0.5,{G}_risk+0.5:0.5",
    "sceptic": f"{G}_sceptic-prompt",
    "claude-haiku": "claude-haiku", "claude-sonnet": "claude-sonnet", "claude-opus": "claude-opus",
    "claude-fable": "claude-fable", "gpt-6-luna": "gpt-6-luna",
    "llama-base": f"{L}_base", "llama-risk-2": f"{L}_risk-2", "llama-risk+1": f"{L}_risk+1",
    "llama-risk+2": f"{L}_risk+2", "llama-half-reckless": f"{L}_base:0.5,{L}_risk+2:0.5",
    "llama-lora": f"{L}+risky-lora_base",
}

SAFEGUARDS = {
    "none": {},
    "order_cap": {"order_cap_frac": 0.25},
    "leverage_cap": {"leverage_cap": 1.0},
    "approval_delay": {"approval_threshold_frac": 0.2, "approval_delay_ticks": 5},
    "cooldown": {"cooldown_ticks": 5},
    "stop_loss": {"account_stop_loss": 0.05},
    "circuit_breaker": {"circuit_breaker_move": 0.02, "circuit_breaker_window": 5, "circuit_breaker_halt_ticks": 5},
    "hide_unverified": {"provenance_filter": "hide_unverified"},
    "delay_unverified": {"provenance_filter": "delay_unverified:15"},
}
SAFEGUARD_ALIASES = {"account_stop_loss": "stop_loss", "provenance_filter": "hide_unverified"}

SYM = {"equity": "HLVN", "crypto": "MRD"}
HKINDS = {
    "fake_unv": ("fake", False, "unverified social post", "bearish"),
    "true_ver": ("true", True, "verified wire", "bearish"),
    "fake_ver": ("fake", True, "verified wire", "bearish"),
    "true_unv": ("true", False, "unverified social post", "bearish"),
    "fake_unv_bull": ("fake", False, "unverified social post", "bullish"),
}
RESILIENCE = ("high", "medium", "low")
EXCESS_KEYS = ("crash_depth", "overshoot", "overshoot_up", "dislocation_area", "n_liquidations", "liquidation_volume",
               "action_correlation", "volume_total", "mean_spread_bps_post", "min_depth_post")

_HEADLINES = None
_SURR = None


class BadRequest(ValueError):
    pass


def _headlines():
    global _HEADLINES
    if _HEADLINES is None:
        with open(os.path.join(ROOT, "data", "headlines.json")) as f:
            h = json.load(f)
        _HEADLINES = h["items"] if isinstance(h, dict) else h
    return _HEADLINES


def headlines_for(regime: str, direction: str) -> list:
    c = [h for h in _headlines() if h["asset"] == SYM[regime] and h["direction"] == direction and h["severity"] in SEVERITY]
    return c[:N_HEADLINES]


def _surrogates():
    global _SURR
    if _SURR is None:
        from stampede.sim.surrogate import load_surrogates

        _SURR = load_surrogates(SURROGATE_DIR)
    return _SURR


def _surrogate_fn(name, obs, vals, spec):
    s = _surrogates()[f"{name}_{spec.regime}"]
    bull = (spec.headline or {}).get("direction") == "bullish" if spec.headline else None
    return s.probs_for(obs, vals, bullish=bull)


def pop_groups(pop: str, n_agents: int = N_AGENTS) -> list:
    """Split a population spec ("table:fraction,...") into groups, rounding as the sweeps did."""
    fr = []
    for x in [x for x in POP_SPECS[pop].split(",") if x]:
        t, f = (x.split(":") + ["1"])[:2]
        fr.append((t, float(f)))
    tot = sum(f for _, f in fr)
    left, groups = n_agents, []
    for i, (t, f) in enumerate(fr):
        n = left if i == len(fr) - 1 else int(round(n_agents * f / tot))
        left -= n
        if n > 0:
            groups.append({"table": t, "n": n})
    return groups


def normalise(params: dict) -> dict:
    """Validate request parameters; raise BadRequest with a readable message."""
    regime = params.get("regime", "equity")
    if regime not in SYM:
        raise BadRequest(f"unknown regime {regime!r}; use equity or crypto")
    pop = params.get("pop")
    if pop not in POP_SPECS:
        raise BadRequest(f"unknown population {pop!r}")
    hcode = params.get("hcode", "fake_unv")
    if hcode not in HKINDS:
        raise BadRequest(f"unknown headline condition {hcode!r}; use one of {', '.join(HKINDS)}")
    sg = params.get("safeguard") or "none"
    sg = SAFEGUARD_ALIASES.get(sg, sg)
    if sg not in SAFEGUARDS:
        raise BadRequest(f"unknown safeguard {sg!r}; use one of {', '.join(SAFEGUARDS)}")
    try:
        capital = float(params.get("capital", 1.0))
    except (TypeError, ValueError):
        raise BadRequest("capital must be a number")
    if not (CAPITAL_RANGE[0] <= capital <= CAPITAL_RANGE[1]):
        raise BadRequest(f"capital must be between {CAPITAL_RANGE[0]} and {CAPITAL_RANGE[1]}")
    seed = params.get("seed")
    if seed is None:
        seed = int.from_bytes(os.urandom(2), "big") % 10000
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        raise BadRequest("seed must be an integer")
    if not (0 <= seed <= 9999):
        raise BadRequest("seed must be between 0 and 9999")
    res = params.get("resilience") or None
    if res is not None and res not in RESILIENCE:
        raise BadRequest(f"unknown resilience {res!r}; use high, medium or low")
    for g in pop_groups(pop):
        if f"{g['table']}_{regime}" not in _surrogates():
            raise BadRequest(f"population {pop!r} has no response table for the {regime} regime")
    return {"regime": regime, "pop": pop, "capital": round(capital, 4), "hcode": hcode, "safeguard": sg, "seed": seed,
            "resilience": None if res == "medium" else res}


def _r(x):
    return None if x is None else float(f"{x:.6g}")


def _compact(tr: dict) -> dict:
    """Same shape as the site's recorded episodes (scripts/export_site.py)."""
    out = {k: v for k, v in tr.items() if k != "frames"}
    frames = []
    for f in tr["frames"]:
        g = {k: f[k] for k in ("t", "halted", "liquidations", "news") if k in f}
        for k in ("price", "fundamental", "best_bid", "best_ask", "volume", "liquidation_volume"):
            if k in f:
                g[k] = _r(f[k])
        g["bids"] = [[_r(p), round(q, 2)] for p, q in (f.get("bids") or [])[:5]]
        g["asks"] = [[_r(p), round(q, 2)] for p, q in (f.get("asks") or [])[:5]]
        g["agents"] = [{"id": a["id"], "exposure": round(a["exposure"], 3), "equity": round(a["equity"]),
                        "action": None if a.get("action") is None else round(a["action"], 3),
                        "liquidated": a.get("liquidated", False)} for a in f.get("agents", [])]
        frames.append(g)
    out["frames"] = frames
    return out


def clean(o):
    """NaN and infinities become null; numpy scalars become Python numbers."""
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, "item"):
        return clean(o.item())
    return o


def _num(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def simulate(params: dict) -> dict:
    """Run the headline episode and its twin. params must already be normalised."""
    from stampede.sim.episode import EpisodeSpec, Group, run_episodes

    p = params
    kind, verified, source, direction = HKINDS[p["hcode"]]
    hs = headlines_for(p["regime"], direction)
    h = hs[p["seed"] % len(hs)]
    groups = pop_groups(p["pop"])
    specs = []
    for k in (kind, "none"):
        specs.append(EpisodeSpec(
            regime=p["regime"], seed=p["seed"], n_ticks=N_TICKS, shock_t=SHOCK_T,
            correction_t=CORRECTION_T if k == "fake" else None, headline_kind=k, headline=h if k != "none" else None,
            source_verified=verified, source_label=source, capital_ratio=p["capital"],
            safeguards=dict(SAFEGUARDS[p["safeguard"]]), act_prob=ACT_PROB, resilience=p["resilience"],
            depth_scale=1.0, model="surrogate", tag="live",
            groups=[Group(name=g["table"], n=g["n"], surrogate=g["table"], init_exposure=1.0) for g in groups]))
    head, twin = run_episodes(specs, decider=None, surrogate_fn=_surrogate_fn)
    key = hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()[:12]
    head["id"] = f"live_{key}"
    head["config"].update(pop=p["pop"], hcode=p["hcode"], safeguard=p["safeguard"], adoption=p["capital"], live=True,
                          resilience_label=p["resilience"] or "medium")
    hm, tm = head["metrics"], twin["metrics"]
    metrics = dict(hm)
    for m in EXCESS_KEYS:
        a, b = _num(hm.get(m)), _num(tm.get(m))
        metrics[f"excess_{m}"] = None if a is None or b is None else a - b
    return clean({"episode": _compact(head), "twin": {"metrics": tm}, "metrics": metrics})
