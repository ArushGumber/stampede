"""BackgroundConfig.resilience: presets, units, the high setting left as calibrated, ordering of
dislocation half-lives, calibration of the medium default, and stability under heavy two-sided flow."""
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest

from stampede.exchange import (BackgroundConfig, Exchange, NewsItem, Shock, TargetExposure, regime)
from stampede.exchange.config import RESILIENCE, effective_background

SETTINGS = ("high", "medium", "low")


def cfg_for(name, res, seed=0, n_ticks=60):
    cfg = regime(name, seed=seed, n_ticks=n_ticks)
    cfg.background.resilience = res
    return cfg


def dump_gap(name, res, seeds, frac=1.0, t_dump=20, n_ticks=50):
    """Mean over seeds of price / same-seed baseline price - 1 after 8 accounts holding frac * depth go flat."""
    gaps = []
    for seed in seeds:
        base = Exchange(cfg_for(name, res, seed, n_ticks))
        ex = Exchange(cfg_for(name, res, seed, n_ticks))
        D = ex.cfg.background.depth_notional
        for i in range(8):
            ex.add_account(f"d{i}", cash=frac * D / 8, initial_exposure=1.0)
        while not ex.done:
            if ex.t == t_dump:
                for i in range(8):
                    ex.submit(f"d{i}", TargetExposure(0.0))
            ex.step()
            base.step()
        gaps.append(np.array(ex.series()["price"]) / np.array(base.series()["price"]) - 1)
    return np.mean(gaps, axis=0)


def half_life(g, t0=20):
    i = t0 + int(np.argmin(g[t0:t0 + 4]))
    for j in range(i, len(g)):
        if g[j] >= g[i] / 2:
            return j - i
    return None


def test_defaults_and_validation():
    assert BackgroundConfig().resilience == "high"
    assert regime("equity").background.resilience == "medium"
    assert regime("crypto").background.resilience == "medium"
    assert set(RESILIENCE) == set(SETTINGS) and RESILIENCE["high"] == {}
    cfg = regime("equity")
    cfg.background.resilience = "extreme"
    with pytest.raises(ValueError):
        Exchange(cfg)


def test_preset_units_and_config_not_mutated():
    for name, sigma_bps in (("equity", 9.2), ("crypto", 22.0)):
        cfg = regime(name)
        before = dataclasses.asdict(cfg)
        med = effective_background(cfg.background, cfg.fundamental.sigma)
        low = effective_background(dataclasses.replace(cfg.background, resilience="low"), cfg.fundamental.sigma)
        high = effective_background(dataclasses.replace(cfg.background, resilience="high"), cfg.fundamental.sigma)
        assert dataclasses.asdict(cfg) == before
        assert high == dataclasses.replace(cfg.background, resilience="high")
        assert med.fund_noise_bps == pytest.approx(RESILIENCE["medium"]["fund_noise_bps"] * sigma_bps)
        assert 15 < med.fund_noise_bps < 100                          # tens of basis points
        assert med.fund_order_frac == pytest.approx(4.0 * cfg.background.noise_order_frac)
        assert low.mm_flow_impact == pytest.approx(1.5 * cfg.background.mm_flow_impact)
        Exchange(cfg)
        assert dataclasses.asdict(cfg) == before


@pytest.mark.parametrize("res", SETTINGS)
def test_fake_shock_moves_nothing_in_any_setting(res):
    a = Exchange(cfg_for("crypto", res, seed=3, n_ticks=40))
    cfg = cfg_for("crypto", res, seed=3, n_ticks=40)
    cfg.shocks = [Shock(t=10, kind="fake", news=NewsItem("unverified social post", False, "x")),
                  Shock(t=20, kind="correction")]
    b = Exchange(cfg)
    while not a.done:
        a.step()
        b.step()
    assert a.series()["price"] == b.series()["price"]


def test_fundamental_path_is_shared_across_settings():
    fs = []
    for res in SETTINGS:
        ex = Exchange(cfg_for("equity", res, seed=4, n_ticks=30))
        while not ex.done:
            ex.step()
        fs.append(ex.series()["fundamental"])
    assert fs[0] == fs[1] == fs[2]


def test_dislocation_half_life_ordering():
    seeds = range(6)
    hl = {res: half_life(dump_gap("equity", res, seeds)) for res in SETTINGS}
    peak = {res: float(dump_gap("equity", res, seeds)[20:24].min()) for res in ("high", "medium")}
    assert hl["high"] <= 2
    assert 3 <= hl["medium"] <= 12
    assert hl["low"] is None or hl["low"] > hl["medium"]
    assert peak["medium"] < -0.012 and peak["medium"] < peak["high"]


@pytest.mark.parametrize("name,target", [("equity", 0.0010), ("crypto", 0.0022)])
def test_medium_per_tick_volatility_close_to_original(name, target):
    sds = []
    for seed in (1, 2):
        ex = Exchange(regime(name, seed=seed, n_ticks=1500))
        while not ex.done:
            ex.step()
        sds.append(np.diff(np.log(ex.series()["price"])).std())
    assert abs(np.mean(sds) / target - 1) < 0.15
    a = ex.audit()
    assert abs(a["position_sum"]) < 1e-7 and abs(a["cash_error"]) < 1e-3


@pytest.mark.parametrize("name", ["equity", "crypto"])
def test_medium_true_shock_converges(name):
    rec = []
    for seed in range(6):
        cfg = regime(name, seed=seed, n_ticks=60)
        cfg.shocks = [Shock(t=10, kind="true", jump=-0.05, news=NewsItem("wire", True, "x"))]
        ex = Exchange(cfg)
        while not ex.done:
            ex.step()
        rec.append(ex.metrics()["recovery_ticks"])
    assert all(r is not None for r in rec), rec
    assert 6 <= np.mean(rec) <= 25, rec


@pytest.mark.parametrize("res", ["medium", "low"])
def test_heavy_two_sided_flow_stays_orderly(res):
    """32 accounts holding 0.8x depth churning between -1x and 2x every tick: the book stays two-sided
    and the price stays within a few percent of the fundamental."""
    import test_exchange_contract as T
    orig = T.regime

    def patched(*a, **k):
        c = orig(*a, **k)
        c.background.resilience = res
        return c

    T.regime = patched
    try:
        for name in ("equity", "crypto"):
            ex = T.busy_episode(seed=1, name=name)
            frames = ex.trace()["frames"]
            assert all(f["best_bid"] is not None and f["best_ask"] is not None for f in frames)
            dev = max(abs(f["price"] / f["fundamental"] - 1) for f in frames)
            assert dev < 0.06, (name, dev)
    finally:
        T.regime = orig
