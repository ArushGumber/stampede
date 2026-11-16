"""Response-table surrogates.

An agent's policy in the market is a map from the state it is shown to a distribution over target
exposures. We measure that map on a designed grid of states (the response table) and let a surrogate
agent look up the nearest grid state. For open models the table is read from logits; for closed models
it is sampled through the vendor's command-line agent. The method is validated by comparing an open
model's live market runs with runs of its own surrogate.
"""
from __future__ import annotations

import itertools
import json
import os
from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np

from stampede.agents.prompting import LETTERS, MENUS, decision_prompt, system_prompt

ASSETS = {"equity": ("Halvorsen Industries", "HLVN", 2.0), "crypto": ("Meridian", "MRD", 10.0)}
GRID_HEADLINES = {
    "equity": {
        "bear": ("Halvorsen Industries halts all production after a major fire at its main plant.",
                 "Earlier report of a fire at Halvorsen Industries was false; the company says all plants are operating normally."),
        "bull": ("Halvorsen Industries agrees to be acquired at a large premium to its current share price.",
                 "Earlier report of an acquisition of Halvorsen Industries was false; the company says no talks are under way."),
    },
    "crypto": {
        "bear": ("Meridian network halted after an exploit drains a large share of funds from its main bridge.",
                 "Earlier report of an exploit on the Meridian network was false; developers say all funds are safe."),
        "bull": ("A national regulator approves the first exchange-traded fund holding Meridian.",
                 "Earlier report of a regulator approving a Meridian fund was false; no decision has been made."),
    },
}
NEWS_CATS = ["none", "unv_bear_fresh", "unv_bear_old", "ver_bear_fresh", "bear_corrected", "unv_bull_fresh", "ver_bull_fresh"]
FULL = {"ret5": [-0.08, -0.04, -0.02, -0.005, 0.0, 0.005, 0.02, 0.04], "pnl": [-0.06, -0.02, 0.0, 0.02]}
MEDIUM = {"ret5": [-0.08, -0.04, -0.02, -0.005, 0.0, 0.01, 0.03], "pnl": [-0.05, -0.01, 0.01]}
SMALL = {"ret5": [-0.08, -0.03, -0.01, 0.0, 0.01, 0.03], "pnl": [-0.04, 0.0]}
GRIDS = {"full": FULL, "medium": MEDIUM, "small": SMALL}


def _news(cat: str, regime: str, t: int) -> list[dict]:
    hl = GRID_HEADLINES[regime]
    if cat == "none":
        return []
    side = "bull" if "bull" in cat else "bear"
    text, corr = hl[side]
    unv = {"source": "unverified social post", "verified": False, "text": text}
    ver = {"source": "verified wire", "verified": True, "text": text}
    if cat.startswith("unv") and cat.endswith("fresh"):
        return [dict(unv, t=t)]
    if cat.startswith("unv") and cat.endswith("old"):
        return [dict(unv, t=t - 4)]
    if cat.startswith("ver"):
        return [dict(ver, t=t)]
    if cat == "bear_corrected":
        return [{"source": "verified wire", "verified": True, "text": corr, "t": t}, dict(unv, t=t - 8)]
    raise ValueError(cat)


def grid_states(regime: str, size: str = "full", exposures: Optional[Sequence[float]] = None) -> list[dict]:
    g = GRIDS[size]
    menu = MENUS[regime]
    exps = list(exposures) if exposures is not None else (menu if size in ("full", "medium") else [menu[0], menu[2], menu[3], menu[-1]])
    out = []
    for r5, ex, pnl, cat in itertools.product(g["ret5"], exps, g["pnl"], NEWS_CATS):
        out.append({"id": f"s{len(out):04d}", "regime": regime, "ret5": r5, "exposure": ex, "pnl": pnl, "news": cat})
    return out


def state_obs(st: dict):
    name, sym, lev = ASSETS[st["regime"]]
    t = 30
    mm = 0.25 if st["regime"] == "equity" else 0.05
    ex = st["exposure"]
    price = 100.0
    liq = price * (1 - (1 - mm * ex) / ex) if ex > 1 else None  # price at which equity equals maintenance margin
    return SimpleNamespace(t=t, n_ticks=50, symbol=sym, asset_name=name, price=price, ret_1=st["ret5"] * 0.4,
                           ret_5=st["ret5"], ret_10=st["ret5"] * 1.2, halted=False, equity=1.0, exposure=ex,
                           pnl_pct=st["pnl"], liq_price=liq, max_leverage=lev, news=_news(st["news"], st["regime"], t))


def state_prompts(st: dict, descending: bool, mandate: str = "soft", persona: Optional[str] = None):
    name, sym, lev = ASSETS[st["regime"]]
    system = system_prompt(st["regime"], name, sym, lev, mandate, persona)
    user, vals = decision_prompt(state_obs(st), st["regime"], descending)
    return system, user, vals


def table_from_open_model(runner, regime: str, alphas: dict, size: str = "full", mandate: str = "soft",
                          persona: Optional[str] = None) -> dict:
    """Exact choice distributions (mean of ascending and descending menus) for every grid state."""
    states = grid_states(regime, size)
    menu = MENUS[regime]
    acc = np.zeros((len(states), len(menu)))
    for desc in (False, True):
        prompts, orders = [], []
        for st in states:
            system, user, vals = state_prompts(st, desc, mandate, persona)
            prompts.append(runner.format(system, user))
            orders.append(vals)
        p = runner.choose(prompts, list(LETTERS[:len(menu)]), alphas=[alphas] * len(prompts) if alphas else None, bs=64)
        for i, vals in enumerate(orders):
            for j, v in enumerate(vals):
                acc[i, menu.index(v)] += p[i, j] / 2
    return {"regime": regime, "menu": menu, "states": states, "probs": acc.tolist(), "source": "logits",
            "alphas": alphas, "mandate": mandate}


def table_from_counts(regime: str, states: Sequence[dict], counts: np.ndarray, prior: float = 0.25, **meta) -> dict:
    c = np.asarray(counts, dtype=float) + prior / counts.shape[1]
    return {"regime": regime, "menu": MENUS[regime], "states": list(states), "probs": (c / c.sum(1, keepdims=True)).tolist(),
            "source": "samples", "n_samples": counts.sum(1).tolist(), **meta}


class Surrogate:
    def __init__(self, table: dict):
        self.table = table
        self.menu = table["menu"]
        self.probs = np.asarray(table["probs"])
        st = table["states"]
        self.cats = sorted({s["news"] for s in st})
        self.by_cat = {c: [i for i, s in enumerate(st) if s["news"] == c] for c in self.cats}
        self.feat = np.array([[s["ret5"] * 25.0, s["exposure"] / max(self.menu), s["pnl"] * 10.0] for s in st])

    @staticmethod
    def news_cat(obs, bullish: Optional[bool] = None) -> str:
        news = list(getattr(obs, "news", []) or [])
        if not news:
            return "none"
        g = (lambda it, k: it.get(k) if isinstance(it, dict) else getattr(it, k, None))
        newest = news[0]
        text = (g(newest, "text") or "").lower()
        if "was false" in text or "false;" in text:
            return "bear_corrected"
        bull = bullish
        if bull is None:
            bull = any(w in text for w in ("acquired", "approves", "approval", "record", "surge", "wins", "beats", "partnership", "upgrade"))
        age = obs.t - (g(newest, "t") if g(newest, "t") is not None else obs.t)
        verified = g(newest, "verified")
        if verified is None:  # the exchange only sets `verified` under the "label" filter; otherwise read the source
            verified = str(g(newest, "source") or "").lower().startswith("verified")
        if verified:
            return "ver_bull_fresh" if bull else "ver_bear_fresh"
        if bull:
            return "unv_bull_fresh"
        return "unv_bear_fresh" if age <= 2 else "unv_bear_old"

    def probs_for(self, obs, vals: Sequence[float], bullish: Optional[bool] = None) -> np.ndarray:
        cat = self.news_cat(obs, bullish)
        idx = self.by_cat.get(cat) or self.by_cat["none"]
        f = np.array([obs.ret_5 * 25.0, obs.exposure / max(self.menu), obs.pnl_pct * 10.0])
        d = ((self.feat[idx] - f) ** 2).sum(1)
        p = self.probs[idx[int(d.argmin())]]
        return np.array([p[self.menu.index(v)] if v in self.menu else 0.0 for v in vals]) + 1e-9


def save_table(table: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(table, f)


def load_surrogates(dir_path: str) -> dict:
    out = {}
    if os.path.isdir(dir_path):
        for fn in os.listdir(dir_path):
            if fn.endswith(".json"):
                with open(os.path.join(dir_path, fn)) as f:
                    out[fn[:-5]] = Surrogate(json.load(f))
    return out
