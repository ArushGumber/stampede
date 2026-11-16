"""Run market episodes populated by language-model agents, many episodes in lockstep.

Each tick, every agent that checks in gets a prompt built from its own observation and picks a target
exposure. Identical (prompt, steering) pairs are evaluated once and memoised; each agent then samples
its own action from the resulting distribution, so agents stay heterogeneous while the GPU work is shared.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Callable, Optional, Sequence

import numpy as np

from stampede.agents.prompting import LETTERS, MENUS, decision_prompt, system_prompt

SOFT = ["soft", "soft1", "soft2"]


@dataclass
class Group:
    name: str
    n: int
    alphas: dict = field(default_factory=dict)        # trait -> steering strength
    persona: Optional[str] = None                     # prompt persona text (alternative to steering)
    model: Optional[str] = None                       # defaults to the episode model
    surrogate: Optional[str] = None                   # name of a response table to use instead of a live model
    init_exposure: Optional[float] = None             # None: sample from the group's own calm-market choice


@dataclass
class EpisodeSpec:
    regime: str = "equity"
    seed: int = 0
    n_ticks: int = 50
    shock_t: int = 20
    correction_t: Optional[int] = 32
    headline_kind: str = "fake"                       # "fake" | "true" | "none"
    headline: Optional[dict] = None                   # {id, text, correction_text, implied_jump, direction}
    source_verified: bool = False
    source_label: str = "unverified social post"
    capital_ratio: float = 0.25                       # total agent equity / resting notional within 1% of mid
    groups: list = field(default_factory=list)
    safeguards: dict = field(default_factory=dict)
    steer_shift: dict = field(default_factory=dict)   # added to every agent's alphas (steering as a safeguard)
    act_prob: float = 0.5
    temperature: float = 1.0
    mandate: str = "soft"
    model: str = "llama-3.1-8b"
    depth_scale: float = 1.0
    resilience: Optional[str] = None                  # None: the regime's default ("medium"); or "high" | "low"
    tag: str = ""

    def key(self) -> str:
        d = asdict(self)
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


class ChoiceCache:
    """Exact memo of choice distributions keyed by model, system text, user text and steering."""

    def __init__(self, path: Optional[str] = None):
        self.path, self.d, self.hits, self.misses = path, {}, 0, 0
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                self.d = pickle.load(f)

    @staticmethod
    def k(model: str, system: str, user: str, alphas: dict) -> tuple:
        a = tuple(sorted((t, round(float(v), 4)) for t, v in (alphas or {}).items() if v))
        return (model, hashlib.sha1((system + "\x00" + user).encode()).hexdigest(), a)

    def save(self):
        if self.path:
            tmp = self.path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(self.d, f)
            os.replace(tmp, self.path)


class Decider:
    """Turns (model, system, user, alphas) requests into choice distributions, with dedup and memo."""

    def __init__(self, runners: dict, cache: Optional[ChoiceCache] = None, surrogates: Optional[dict] = None,
                 log_proj: bool = True, bs: int = 64):
        self.runners, self.cache, self.surrogates = runners, cache or ChoiceCache(), surrogates or {}
        self.log_proj, self.bs = log_proj, bs

    def decide(self, reqs: Sequence[tuple]) -> list[tuple]:
        """reqs: (model, system, user, alphas, n_options). Returns [(probs, proj_or_None)] aligned with reqs."""
        keys = [ChoiceCache.k(m, s, u, a) for (m, s, u, a, _n) in reqs]
        todo: dict[tuple, int] = {}
        for i, k in enumerate(keys):
            if k in self.cache.d:
                self.cache.hits += 1
            elif k not in todo:
                todo[k] = i
                self.cache.misses += 1
            else:
                self.cache.hits += 1
        by_model: dict[str, list[tuple]] = {}
        for k, i in todo.items():
            by_model.setdefault(reqs[i][0], []).append((k, i))
        for model, items in by_model.items():
            r = self.runners[model]
            prompts = [r.format(reqs[i][1], reqs[i][2]) for _, i in items]
            alphas = [reqs[i][3] for _, i in items]
            opts = [list(LETTERS[:reqs[i][4]]) for _, i in items]
            out = r.choose(prompts, opts, alphas=alphas, bs=self.bs, return_proj=self.log_proj)
            probs, proj = (out if self.log_proj else (out, [None] * len(items)))
            for j, (k, _i) in enumerate(items):
                self.cache.d[k] = (probs[j].astype(np.float32), proj[j] if proj else None)
        return [self.cache.d[k] for k in keys]


def _sample(p: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    q = np.asarray(p, dtype=np.float64)
    if temperature != 1.0:
        q = np.power(np.clip(q, 1e-12, 1), 1.0 / max(temperature, 1e-3))
    q = q / q.sum()
    return int(rng.choice(len(q), p=q))


def _calm_obs(cfg, t=0, exposure=1.0):
    return SimpleNamespace(t=t, n_ticks=cfg.n_ticks, symbol=cfg.symbol, price=cfg.initial_price, ret_1=0.0, ret_5=0.0,
                           ret_10=0.0, halted=False, equity=1.0, exposure=exposure, pnl_pct=0.0, liq_price=None,
                           max_leverage=cfg.max_leverage, news=[])


class _Ep:
    """One episode's state: exchange plus per-agent bookkeeping."""

    def __init__(self, spec: EpisodeSpec, decider: Decider):
        from stampede.exchange import Exchange, NewsItem, SafeguardConfig, Shock, regime

        self.spec = spec
        cfg = regime(spec.regime, seed=spec.seed, n_ticks=spec.n_ticks)
        if spec.depth_scale != 1.0:
            cfg.background.depth_notional *= spec.depth_scale
        if spec.resilience:
            cfg.background.resilience = spec.resilience
        sg = dict(spec.safeguards or {})
        if sg:
            cfg.safeguards = SafeguardConfig(**sg)
        shocks = []
        h = spec.headline or {}
        if spec.headline_kind in ("fake", "true") and h:
            jump = float(h.get("implied_jump", 0.0)) if spec.headline_kind == "true" else 0.0
            shocks.append(Shock(t=spec.shock_t, kind=spec.headline_kind, jump=jump,
                                news=NewsItem(source=spec.source_label, verified=spec.source_verified, text=h["text"])))
            if spec.headline_kind == "fake" and spec.correction_t is not None and h.get("correction_text"):
                shocks.append(Shock(t=spec.correction_t, kind="correction", jump=0.0,
                                    news=NewsItem(source="verified wire", verified=True, text=h["correction_text"])))
        cfg.shocks = shocks
        self.cfg = cfg
        self.ex = Exchange(cfg)
        self.menu = [m for m in MENUS[spec.regime] if m <= cfg.max_leverage + 1e-9]
        self.agents = []
        n_total = sum(g.n for g in spec.groups)
        cash = spec.capital_ratio * cfg.background.depth_notional / max(n_total, 1)
        idx = 0
        for g in spec.groups:
            alphas = {t: float(v) + float(spec.steer_shift.get(t, 0.0)) for t, v in g.alphas.items()}
            for t, v in spec.steer_shift.items():
                alphas.setdefault(t, float(v))
            for _ in range(g.n):
                aid = f"a{idx:03d}"
                mandate = SOFT[idx % 3] if spec.mandate == "soft" else spec.mandate
                system = system_prompt(spec.regime, cfg.asset_name, cfg.symbol, cfg.max_leverage, mandate, g.persona)
                self.agents.append(SimpleNamespace(
                    id=aid, group=g.name, alphas=alphas, system=system, model=g.model or spec.model,
                    surrogate=g.surrogate, rng=np.random.default_rng([spec.seed, 7919, idx]),
                    init=g.init_exposure, cash=cash, kind="surrogate" if g.surrogate else "llm"))
                idx += 1
        self._init_accounts(decider)

    def _init_accounts(self, decider: Decider):
        reqs, order = [], []
        for a in self.agents:
            if a.init is None and not a.surrogate:
                desc = bool(a.rng.integers(2))
                user, vals = decision_prompt(_calm_obs(self.cfg), self.spec.regime, desc, self.menu)
                reqs.append((a.model, a.system, user, a.alphas, len(vals)))
                order.append((a, vals))
        res = decider.decide(reqs) if reqs else []
        for (a, vals), (p, _pj) in zip(order, res):
            a.init = float(vals[_sample(p, self.spec.temperature, a.rng)])
        for a in self.agents:
            cap = 1.0 if self.spec.mandate == "cap1" else None
            self.ex.add_account(a.id, cash=a.cash, group=a.group, mandate_max_exposure=cap,
                                initial_exposure=float(a.init if a.init is not None else 1.0))

    def requests(self):
        out = []
        for a in self.agents:
            if a.rng.random() >= self.spec.act_prob:
                continue
            obs = self.ex.observe(a.id)
            desc = bool(a.rng.integers(2))
            user, vals = decision_prompt(obs, self.spec.regime, desc, self.menu)
            out.append((a, obs, user, vals))
        return out


def run_episodes(specs: Sequence[EpisodeSpec], decider: Decider, surrogate_fn: Optional[Callable] = None,
                 on_frame: Optional[Callable] = None) -> list[dict]:
    """Run episodes in lockstep and return their traces (contract section 3) with config and agent details."""
    from stampede.exchange import TargetExposure

    eps = [_Ep(s, decider) for s in specs]
    live = list(range(len(eps)))
    while live:
        batch, reqs = [], []
        for ei in live:
            ep = eps[ei]
            for (a, obs, user, vals) in ep.requests():
                if a.surrogate:
                    p = surrogate_fn(a.surrogate, obs, vals, ep.spec)
                    x = float(vals[_sample(p, ep.spec.temperature, a.rng)])
                    ep.ex.submit(a.id, TargetExposure(x=x, style="market"))
                    continue
                batch.append((ep, a, vals))
                reqs.append((a.model, a.system, user, a.alphas, len(vals)))
        res = decider.decide(reqs) if reqs else []
        for (ep, a, vals), (p, pj) in zip(batch, res):
            x = float(vals[_sample(p, ep.spec.temperature, a.rng)])
            ep.ex.submit(a.id, TargetExposure(x=x, style="market"))
            if pj:
                ep.ex.annotate(a.id, proj={k: round(float(v), 4) for k, v in pj.items()})
        nxt = []
        for ei in live:
            frame = eps[ei].ex.step()
            if on_frame:
                on_frame(ei, frame)
            if not eps[ei].ex.done:
                nxt.append(ei)
        live = nxt
    traces = []
    for ep in eps:
        tr = ep.ex.trace()
        meta = {a.id: a for a in ep.agents}
        for rec in tr.get("agents", []):
            a = meta.get(rec["id"])
            if a is not None:
                rec.update(kind=a.kind, model=a.surrogate or a.model, alphas=a.alphas, init_exposure=a.init)
        cfg = dict(tr.get("config", {}))
        sp = asdict(ep.spec)
        cfg.update({k: sp[k] for k in ("regime", "seed", "n_ticks", "shock_t", "correction_t", "headline_kind",
                                       "source_verified", "capital_ratio", "safeguards", "steer_shift", "act_prob",
                                       "mandate", "model", "depth_scale", "resilience", "tag")})
        cfg["headline_id"] = (ep.spec.headline or {}).get("id")
        cfg["groups"] = [{"name": g.name, "n": g.n, "alphas": g.alphas, "persona": bool(g.persona),
                          "model": g.model or ep.spec.model, "surrogate": g.surrogate} for g in ep.spec.groups]
        tr["config"] = cfg
        tr["id"] = "ep_" + ep.spec.key()
        # a no-headline twin is scored over the same window as its headline episode
        tr["metrics"] = ep.ex.metrics(ref_t=ep.spec.shock_t if ep.spec.headline_kind == "none" else None)
        traces.append(tr)
    return traces


def save_trace(tr: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, tr["id"] + ".json.gz")
    with gzip.open(path, "wt") as f:
        json.dump(tr, f, separators=(",", ":"))
    return path


def summary_row(tr: dict) -> dict:
    c, m = tr["config"], tr["metrics"]
    row = {"id": tr["id"], **{k: c.get(k) for k in ("regime", "seed", "headline_kind", "headline_id", "source_verified",
                                                    "capital_ratio", "model", "mandate", "depth_scale", "resilience", "tag")}}
    row["safeguards"] = json.dumps(c.get("safeguards") or {}, sort_keys=True)
    row["steer_shift"] = json.dumps(c.get("steer_shift") or {}, sort_keys=True)
    row["groups"] = json.dumps(c.get("groups"), sort_keys=True)
    for k, v in m.items():
        row[k] = json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
    return row
