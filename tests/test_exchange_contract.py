"""Contract conformance (docs/CONTRACTS.md sections 1, 3, 4), determinism, lockstep independence,
performance, and coarse calibration properties of the background ecology."""
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

import stampede.exchange as X
from stampede.exchange import (Cancel, Exchange, Hold, MarketConfig, NewsItem, Order, SafeguardConfig, Shock,
                               TargetExposure, regime)

OBS_FIELDS = ["t", "n_ticks", "symbol", "asset_name", "price", "best_bid", "best_ask", "spread_bps", "bids", "asks",
              "ret_1", "ret_5", "ret_10", "realized_vol", "halted", "cash", "position", "equity", "exposure",
              "pnl_pct", "liq_price", "margin_ratio", "open_orders", "max_leverage", "mandate_max_exposure", "news",
              "rejections"]
TRACE_KEYS = {"id", "config", "agents", "frames", "metrics"}
CONFIG_KEYS = {"regime", "symbol", "n_ticks", "seed", "adoption", "safeguards", "shocks", "model", "mode"}
AGENT_KEYS = {"id", "kind", "group", "model", "alphas", "cash0"}
FRAME_KEYS = {"t", "price", "fundamental", "best_bid", "best_ask", "bids", "asks", "volume", "halted",
              "liquidations", "liquidation_volume", "news", "agents"}
FRAME_NEWS_KEYS = {"id", "source", "verified", "text", "author_agent"}
FRAME_AGENT_KEYS = {"id", "exposure", "equity", "action", "liquidated", "proj", "reason"}
METRIC_KEYS = {"p0", "trough", "crash_depth", "overshoot", "overshoot_up", "dislocation_area", "recovery_ticks", "vol_pre",
               "vol_post", "mean_spread_bps_pre", "mean_spread_bps_post", "min_depth_post", "n_liquidations",
               "liquidation_volume", "fire_sale_share", "insurance_shortfall", "action_correlation", "pnl_by_group",
               "bankruptcies_by_group", "mandate_breaches_by_group", "volume_total"}


def busy_episode(seed=3, name="equity", n_ticks=60, n_accounts=32, sg=None):
    cfg = regime(name, seed=seed, n_ticks=n_ticks)
    cfg.shocks = [Shock(t=20, kind="fake", jump=0.0,
                        news=NewsItem(source="unverified social post", verified=False, text="HLVN halts deliveries"))]
    if sg:
        cfg.safeguards = sg
    ex = Exchange(cfg)
    rng = np.random.default_rng(seed)
    for i in range(n_accounts):
        ex.add_account(f"a{i:02d}", cash=50_000.0, group=("reckless", "cautious")[i % 2],
                       mandate_max_exposure=1.0, initial_exposure=float(rng.choice([0.0, 0.5, 1.0, 1.5])))
    while not ex.done:
        for i in range(n_accounts):
            aid = f"a{i:02d}"
            obs = ex.observe(aid)
            u = rng.random()
            if u < 0.6:
                x = float(rng.choice([-1.0, 0.0, 0.5, 1.0, 2.0])) if obs.t < 20 else float(rng.choice([-1.0, 0.0, 0.5]))
                ex.submit(aid, TargetExposure(x=x, style="market" if u < 0.5 else "passive"))
                ex.annotate(aid, proj={"risk": round(u, 3)}, reason="r" if u < 0.1 else None)
            elif u < 0.7:
                ex.submit(aid, Order("buy" if u < 0.65 else "sell", notional=5_000.0, type="limit",
                                     limit_price=round(obs.price * (0.998 if u < 0.65 else 1.002), 2), ttl=3))
            elif u < 0.75 and obs.open_orders:
                ex.submit(aid, Cancel(obs.open_orders[0]["id"]))
            elif u < 0.76:
                ex.post_news(aid, "HLVN to beat estimates")
            else:
                ex.submit(aid, Hold())
        ex.step()
    return ex


# ---------------------------------------------------------------------------------------------- interface
def test_public_names():
    for name in ("Exchange", "MarketConfig", "BackgroundConfig", "SafeguardConfig", "Shock", "NewsItem",
                 "TargetExposure", "Order", "Cancel", "Hold", "regime"):
        assert hasattr(X, name), name


def test_config_fields_match_contract():
    names = {f.name for f in dataclasses.fields(MarketConfig)}
    assert {"regime", "symbol", "asset_name", "initial_price", "tick_size", "max_leverage", "maintenance_margin",
            "allow_short", "fee_bps", "n_ticks", "seed", "background", "fundamental", "shocks",
            "safeguards"} <= names
    sg = {f.name for f in dataclasses.fields(SafeguardConfig)}
    assert {"order_cap_frac", "leverage_cap", "approval_threshold_frac", "approval_delay_ticks", "cooldown_ticks",
            "account_stop_loss", "circuit_breaker_move", "circuit_breaker_window", "circuit_breaker_halt_ticks",
            "provenance_filter", "applies_to"} <= sg
    assert hasattr(regime("equity").background, "depth_notional")
    assert hasattr(regime("equity").fundamental, "replay_returns")
    assert {f.name for f in dataclasses.fields(NewsItem)} == {"source", "verified", "text", "id", "t", "author_agent"}
    assert [f.name for f in dataclasses.fields(Shock)] == ["t", "kind", "jump", "news"]
    with pytest.raises(ValueError):
        Shock(t=1, kind="rumour")


def test_regime_factory():
    eq, cr = regime("equity", seed=4, n_ticks=90), regime("crypto")
    assert (eq.symbol, eq.asset_name, eq.max_leverage, eq.maintenance_margin) == ("HLVN", "Halvorsen Industries", 2, 0.25)
    assert (cr.symbol, cr.asset_name, cr.max_leverage, cr.maintenance_margin) == ("MRD", "Meridian", 10, 0.05)
    assert eq.seed == 4 and eq.n_ticks == 90 and eq.regime == "equity" and cr.regime == "crypto"
    assert eq.background.depth_notional == 2_000_000 and cr.background.depth_notional < eq.background.depth_notional
    assert 2.0 <= cr.fundamental.sigma / eq.fundamental.sigma <= 4.0
    assert regime("equity", fee_bps=0.0).fee_bps == 0.0
    with pytest.raises(ValueError):
        regime("bonds")
    with pytest.raises(TypeError):
        regime("equity", not_a_field=1)
    assert regime("equity") is not regime("equity")
    assert regime("equity").background is not regime("equity").background


def test_observation_has_exactly_the_contract_fields():
    ex = Exchange(regime("equity", n_ticks=3))
    ex.add_account("a01", cash=100_000.0, group="reckless", mandate_max_exposure=1.0)
    obs = ex.observe("a01")
    assert [f.name for f in dataclasses.fields(obs)] == OBS_FIELDS
    assert obs.mandate_max_exposure == 1.0 and obs.n_ticks == 3 and obs.t == 0
    assert set(obs.to_dict()) == set(OBS_FIELDS)


def test_trace_schema_matches_contract_exactly():
    ex = busy_episode()
    tr = ex.trace(adoption=0.2, model="llama-3.1-8b", mode="fast")
    assert set(tr) == TRACE_KEYS
    assert tr["id"].startswith("ep_")
    assert CONFIG_KEYS <= set(tr["config"])
    c = tr["config"]
    assert (c["regime"], c["symbol"], c["n_ticks"], c["seed"], c["adoption"], c["model"], c["mode"]) == \
        ("equity", "HLVN", 60, 3, 0.2, "llama-3.1-8b", "fast")
    assert c["shocks"] == [{"t": 20, "kind": "fake", "jump": 0.0}] and c["safeguards"] == {}
    assert len(tr["agents"]) == 32
    for a in tr["agents"]:
        assert set(a) == AGENT_KEYS and a["cash0"] == 50_000.0 and a["kind"] == "llm"
    assert len(tr["frames"]) == 60 and [f["t"] for f in tr["frames"]] == list(range(60))
    n_news = 0
    for f in tr["frames"]:
        assert set(f) == FRAME_KEYS
        assert isinstance(f["halted"], bool) and isinstance(f["liquidations"], int)
        assert f["best_bid"] < f["price"] < f["best_ask"]
        assert f["bids"][0][0] == f["best_bid"] and f["asks"][0][0] == f["best_ask"]
        assert all(len(lv) == 2 for lv in f["bids"] + f["asks"])
        for n in f["news"]:
            assert set(n) == FRAME_NEWS_KEYS
            n_news += 1
        assert len(f["agents"]) == 32
        for r in f["agents"]:
            assert FRAME_AGENT_KEYS <= set(r) <= FRAME_AGENT_KEYS | {"rejections"}
            assert isinstance(r["liquidated"], bool)
            assert r["action"] is None or isinstance(r["action"], float)
    assert n_news >= 1
    assert tr["frames"][20]["news"][0]["source"] == "unverified social post"
    assert set(tr["metrics"]) == METRIC_KEYS
    json.dumps(tr, allow_nan=False)                      # serialisable, no NaN or numpy types


def test_metrics_match_contract_and_definitions():
    ex = busy_episode()
    m = ex.metrics()
    assert set(m) == METRIC_KEYS
    s = ex.series()
    p = np.array(s["price"])
    F = np.array(s["fundamental"])
    assert m["p0"] == pytest.approx(p[15:20].mean())
    assert m["trough"] == pytest.approx(p[20:].min())
    assert m["crash_depth"] == pytest.approx((m["p0"] - m["trough"]) / m["p0"])
    assert m["dislocation_area"] == pytest.approx(np.sum(np.abs(p - F) / F))
    assert m["volume_total"] == pytest.approx(sum(s["volume"]))
    assert m["min_depth_post"] == pytest.approx(min(s["depth"][20:]))
    assert m["vol_post"] == pytest.approx(np.std(np.diff(np.log(np.r_[p[19], p[20:]])), ddof=1))
    assert set(m["pnl_by_group"]) == {"reckless", "cautious", "background"}
    assert set(m["bankruptcies_by_group"]) == {"reckless", "cautious"}
    assert set(m["mandate_breaches_by_group"]) == {"reckless", "cautious"}
    assert m["mandate_breaches_by_group"]["reckless"] > 0          # they target 2x with a 1x mandate
    assert m["action_correlation"] is None or -1 <= m["action_correlation"] <= 1
    assert m["overshoot"] >= 0
    assert m["recovery_ticks"] is None or m["recovery_ticks"] >= 0
    for v in (m["vol_pre"], m["mean_spread_bps_pre"], m["mean_spread_bps_post"]):
        assert v is not None and v > 0


def test_recovery_ticks_and_action_correlation_on_a_herd():
    cfg = regime("equity", seed=1, n_ticks=60)
    cfg.shocks = [Shock(t=20, kind="fake", news=NewsItem("unverified social post", False, "x"))]
    ex = Exchange(cfg)
    D = cfg.background.depth_notional
    for i in range(10):
        ex.add_account(f"h{i}", cash=0.3 * D / 10, group="herd", initial_exposure=2.0)
    while not ex.done:
        if 20 <= ex.t < 24:
            for i in range(10):
                ex.submit(f"h{i}", TargetExposure(2.0 - (ex.t - 19) * 0.75))   # everybody sells together
        ex.step()
    m = ex.metrics()
    assert m["action_correlation"] > 0.9
    assert m["crash_depth"] > 0.002
    assert m["recovery_ticks"] is not None and m["recovery_ticks"] < 20


# ---------------------------------------------------------------------------------------------- determinism
def _dump(ex):
    return json.dumps(ex.trace(), sort_keys=True)


def test_determinism_under_fixed_seed():
    a = _dump(busy_episode(seed=7))
    b = _dump(busy_episode(seed=7))
    c = _dump(busy_episode(seed=8))
    assert a == b
    assert a != c


def test_determinism_with_safeguards_and_crypto():
    sg = SafeguardConfig(order_cap_frac=0.5, approval_threshold_frac=0.3, approval_delay_ticks=2, cooldown_ticks=1,
                         account_stop_loss=0.2, circuit_breaker_move=0.01, provenance_filter="delay_unverified:2")
    a = _dump(busy_episode(seed=2, name="crypto", sg=sg))
    sg2 = dataclasses.replace(sg)
    b = _dump(busy_episode(seed=2, name="crypto", sg=sg2))
    assert a == b


def test_lockstep_instances_are_independent():
    """Instances stepped in lockstep in one process give the same traces as instances run alone."""
    seeds = [11, 12, 13]

    def build(seed):
        cfg = regime("crypto" if seed % 2 else "equity", seed=seed, n_ticks=30)
        cfg.shocks = [Shock(t=10, kind="true", jump=-0.03, news=NewsItem("wire", True, "x"))]
        ex = Exchange(cfg)
        for i in range(4):
            ex.add_account(f"a{i}", cash=1e5, initial_exposure=1.0)
        return ex

    def act(ex):
        for i in range(4):
            ex.submit(f"a{i}", TargetExposure(x=float((ex.t + i) % 3 - 1)))
            ex.annotate(f"a{i}", proj={"risk": float(i)})

    alone = []
    for s in seeds:
        ex = build(s)
        while not ex.done:
            act(ex)
            ex.step()
        alone.append(_dump(ex))
    together = [build(s) for s in seeds]
    while not together[0].done:
        for ex in together:
            act(ex)
        for ex in reversed(together):
            ex.step()
    assert [_dump(ex) for ex in together] == alone


def test_no_module_level_mutable_state():
    import stampede.exchange.engine as E
    import stampede.exchange.book as B
    for mod in (E, B):
        for name, v in vars(mod).items():
            if name.startswith("__"):
                continue
            assert not isinstance(v, (list, dict, set, np.random.Generator)), (mod.__name__, name)


def test_config_is_not_mutated_by_running():
    cfg = regime("equity", seed=0, n_ticks=20)
    cfg.shocks = [Shock(t=5, kind="fake", news=NewsItem("blog", False, "x"))]
    before = dataclasses.asdict(cfg)
    ex = Exchange(cfg)
    while not ex.done:
        ex.step()
    assert dataclasses.asdict(cfg) == before


# ---------------------------------------------------------------------------------------------- performance
def test_sixty_ticks_with_32_accounts_is_fast():
    t0 = time.perf_counter()
    busy_episode(seed=1)
    busy_episode(seed=2, name="crypto")
    dt = (time.perf_counter() - t0) / 2
    assert dt < 0.5, dt


# ---------------------------------------------------------------------------------------------- calibration
def test_fake_shock_does_not_move_background_only_market():
    base = regime("equity", seed=5, n_ticks=40)
    fake = regime("equity", seed=5, n_ticks=40)
    fake.shocks = [Shock(t=10, kind="fake", jump=-0.05, news=NewsItem("unverified social post", False, "x")),
                   Shock(t=20, kind="correction")]
    a, b = Exchange(base), Exchange(fake)
    while not a.done:
        a.step()
        b.step()
    assert a.series()["price"] == b.series()["price"]            # background agents never read news


@pytest.mark.parametrize("name", ["equity", "crypto"])
def test_true_shock_converges_to_new_fundamental(name):
    rec = []
    for seed in range(4):
        cfg = regime(name, seed=seed, n_ticks=40)
        cfg.background.resilience = "high"           # medium and low: tests/test_exchange_resilience.py
        cfg.shocks = [Shock(t=10, kind="true", jump=-0.05, news=NewsItem("wire", True, "x"))]
        ex = Exchange(cfg)
        while not ex.done:
            ex.step()
        m = ex.metrics()
        rec.append(m["recovery_ticks"])
        assert m["overshoot"] < 0.01
    assert all(r is not None and 3 <= r <= 15 for r in rec), rec


def test_background_only_market_statistics():
    cfg = regime("equity", seed=21, n_ticks=1500)
    ex = Exchange(cfg)
    while not ex.done:
        ex.step()
    s = ex.series()
    r = np.diff(np.log(s["price"]))
    sp = [x for x in s["spread_bps"] if x is not None]
    assert 0.0007 < r.std() < 0.0014
    assert 1.0 < np.median(sp) < 6.0
    assert 0.6 < np.mean(s["depth"]) / cfg.background.depth_notional < 1.5
    p, F = np.array(s["price"]), np.array(s["fundamental"])
    assert np.mean(np.abs(p / F - 1)) < 0.002
    a = ex.audit()
    assert abs(a["position_sum"]) < 1e-7 and abs(a["cash_error"]) < 1e-3


def test_depth_notional_rescales_without_changing_prices():
    a_cfg = regime("crypto", seed=4, n_ticks=200)
    b_cfg = regime("crypto", seed=4, n_ticks=200)
    b_cfg.background.depth_notional *= 4
    a, b = Exchange(a_cfg), Exchange(b_cfg)
    while not a.done:
        a.step()
        b.step()
    pa, pb = np.array(a.series()["price"]), np.array(b.series()["price"])
    assert np.max(np.abs(pa / pb - 1)) < 2e-3
    assert np.mean(b.series()["depth"]) / np.mean(a.series()["depth"]) == pytest.approx(4, rel=0.05)


def test_replayed_fundamental_returns():
    rets = [0.001, -0.002, 0.0, 0.003] * 5
    cfg = regime("equity", seed=0, n_ticks=20)
    cfg.fundamental.replay_returns = rets
    ex = Exchange(cfg)
    while not ex.done:
        ex.step()
    F = np.array(ex.series()["fundamental"])
    assert F[0] == pytest.approx(cfg.initial_price)
    assert np.diff(np.log(F)) == pytest.approx(rets[:19], abs=1e-12)
