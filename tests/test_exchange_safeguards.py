"""Every SafeguardConfig field: order cap, leverage cap, approval delay, cooldown, account stop-loss,
circuit breaker, provenance filter, and applies_to."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from stampede.exchange import (Exchange, NewsItem, Order, SafeguardConfig, Shock, TargetExposure, regime)


def make(name="equity", n_ticks=30, seed=0, **sg):
    cfg = regime(name, seed=seed, n_ticks=n_ticks)
    cfg.safeguards = SafeguardConfig(**sg)
    return cfg


def reasons(ex, aid):
    return [r["reason"] for r in ex.last_rejections(aid)]


def expo(frame, aid):
    return {r["id"]: r for r in frame["agents"]}[aid]["exposure"]


def test_all_safeguards_off_by_default():
    sg = SafeguardConfig()
    assert sg.active() == {}
    assert regime("equity").safeguards.active() == {}
    assert regime("crypto").safeguards.active() == {}


def test_order_cap_rejects_raw_orders_and_clips_rebalances():
    ex = Exchange(make(order_cap_frac=0.25))
    ex.add_account("a", cash=100_000)
    ex.submit("a", Order("buy", notional=50_000))
    ex.step()
    assert reasons(ex, "a") == ["order_cap"] and ex.accounts["a"].position == 0
    ex.submit("a", Order("buy", notional=20_000))
    f = ex.step()
    assert reasons(ex, "a") == [] and expo(f, "a") == pytest.approx(0.2, abs=0.01)
    xs = []
    for _ in range(4):                                  # each rebalance moves at most 25% of equity
        ex.submit("a", TargetExposure(1.0))
        f = ex.step()
        xs.append(expo(f, "a"))
    assert xs[0] == pytest.approx(0.45, abs=0.02)
    assert "order_cap_clipped" in [r["reason"] for r in f["agents"][0].get("rejections", [])] or xs[-1] > 0.95
    assert xs[-1] == pytest.approx(1.0, abs=0.02)


def test_leverage_cap_below_regime_maximum():
    ex = Exchange(make("crypto", leverage_cap=3.0))
    ex.add_account("a", cash=100_000)
    assert ex.observe("a").max_leverage == 3.0
    ex.submit("a", TargetExposure(8.0))
    f = ex.step()
    assert expo(f, "a") == pytest.approx(3.0, abs=0.05)
    assert "leverage_clipped" in reasons(ex, "a")
    ex.submit("a", Order("buy", notional=100_000))     # would take it to 4x
    ex.step()
    assert reasons(ex, "a") == ["insufficient_margin"]


def test_approval_delay_holds_large_orders():
    ex = Exchange(make(approval_threshold_frac=0.5, approval_delay_ticks=3))
    ex.add_account("a", cash=100_000)
    ex.add_account("b", cash=100_000)
    ex.submit("a", TargetExposure(1.0))                # 100% of equity: held
    ex.submit("b", TargetExposure(0.3))                # below threshold: immediate
    f = ex.step()                                      # tick 0
    assert reasons(ex, "a") == ["approval_pending"]
    assert expo(f, "a") == 0 and expo(f, "b") == pytest.approx(0.3, abs=0.02)
    ex.submit("a", TargetExposure(1.0))                # resubmitting the same target is not double-counted
    f = ex.step()                                      # tick 1
    assert reasons(ex, "a") == ["approval_pending"] and expo(f, "a") == 0
    f = ex.step()                                      # tick 2
    assert expo(f, "a") == 0
    f = ex.step()                                      # tick 3: released
    assert expo(f, "a") == pytest.approx(1.0, abs=0.02)
    assert len(ex._held) == 0
    # raw orders are held too
    ex.submit("a", Order("sell", notional=80_000))
    f = ex.step()
    assert reasons(ex, "a") == ["approval_pending"]
    for _ in range(3):
        f = ex.step()
    assert expo(f, "a") == pytest.approx(0.2, abs=0.03)


def test_approval_delay_new_target_supersedes_held_one():
    ex = Exchange(make(approval_threshold_frac=0.5, approval_delay_ticks=3))
    ex.add_account("a", cash=100_000)
    ex.submit("a", TargetExposure(1.0))
    ex.step()
    ex.submit("a", TargetExposure(0.2))                # small: executes now, drops the held 1.0
    f = ex.step()
    assert expo(f, "a") == pytest.approx(0.2, abs=0.02)
    for _ in range(3):
        f = ex.step()
    assert expo(f, "a") == pytest.approx(0.2, abs=0.03)


def test_cooldown_between_rebalances():
    ex = Exchange(make(cooldown_ticks=3))
    ex.add_account("a", cash=100_000)
    ex.submit("a", TargetExposure(1.0))
    f = ex.step()                                      # tick 0: executes
    assert expo(f, "a") == pytest.approx(1.0, abs=0.02)
    ex.submit("a", TargetExposure(-1.0))
    f = ex.step()                                      # tick 1: blocked
    assert reasons(ex, "a") == ["cooldown"] and expo(f, "a") > 0.9
    ex.submit("a", Order("sell", qty=1))
    ex.step()                                          # tick 2: blocked
    assert reasons(ex, "a") == ["cooldown"]
    ex.submit("a", TargetExposure(-1.0))
    f = ex.step()                                      # tick 3: allowed again
    assert reasons(ex, "a") == [] and expo(f, "a") == pytest.approx(-1.0, abs=0.02)


def test_account_stop_loss_flattens_and_freezes():
    cfg = regime("equity", seed=0, n_ticks=10)
    cfg.background.enabled = False
    cfg.fundamental.sigma = 0.0
    cfg.fundamental.vol_of_vol = 0.0
    cfg.safeguards = SafeguardConfig(account_stop_loss=0.10)
    ex = Exchange(cfg)
    ex.add_account("mkr", cash=1e7, group="dealer")
    ex.add_account("a", cash=10_000, initial_exposure=1.0)     # 100 shares at 100, cash 0
    ex.submit("mkr", Order("buy", qty=500, type="limit", limit_price=89.0))
    ex.submit("mkr", Order("sell", qty=500, type="limit", limit_price=90.0))
    ex.step()
    a = ex.accounts["a"]
    assert a.position == 0 and a.frozen                         # 10.5% drawdown at mid 89.5: sold at 89
    assert a.cash == pytest.approx(8900 - 8900 * 1e-4)
    assert "stop_loss_triggered" in reasons(ex, "a")
    ex.submit("a", TargetExposure(1.0))
    ex.step()
    assert reasons(ex, "a") == ["stopped_out"] and a.position == 0
    # the dealer is a non-background account too (applies_to=None), but it made money, so it is untouched
    assert not ex.accounts["mkr"].frozen


def test_stop_loss_not_triggered_by_small_drawdown():
    ex = Exchange(make(account_stop_loss=0.2, n_ticks=30))
    ex.add_account("a", cash=100_000, initial_exposure=1.0)
    while not ex.done:
        ex.step()
    assert not ex.accounts["a"].frozen


def test_circuit_breaker_halts_market():
    cfg = make(circuit_breaker_move=0.02, circuit_breaker_window=5, circuit_breaker_halt_ticks=4, n_ticks=40, seed=1)
    cfg.shocks = [Shock(t=10, kind="true", jump=-0.06, news=NewsItem("wire", True, "guidance cut"))]
    ex = Exchange(cfg)
    ex.add_account("a", cash=100_000)
    frames = []
    halted_obs = []
    while not ex.done:
        obs = ex.observe("a")
        halted_obs.append(obs.halted)
        ex.submit("a", TargetExposure(0.5 if ex.t % 2 else 0.4))
        frames.append(ex.step())
    halted = [f["halted"] for f in frames]
    assert any(halted)
    first = halted.index(True)
    assert first > 10
    run = halted[first:first + 4]
    assert run == [True] * 4
    assert halted_obs[first] is True                       # agents are told before they submit
    for f in frames[first:first + 4]:
        assert f["volume"] == 0
        assert f["price"] == frames[first - 1]["price"]
        rej = {r["id"]: r for r in f["agents"]}["a"].get("rejections", [])
        assert [r["reason"] for r in rej] == ["halt"]
    assert frames[first + 4]["volume"] > 0                  # trading resumes
    # the move that triggered it was measured over the window
    j = first - 1
    assert abs(frames[j]["price"] / frames[max(0, j - 5)]["price"] - 1) >= 0.02 or j < 5


def test_circuit_breaker_quiet_market_never_halts():
    ex = Exchange(make(circuit_breaker_move=0.03, n_ticks=200, seed=2))
    halts = 0
    while not ex.done:
        halts += ex.step()["halted"]
    assert halts == 0


def _news_cfg(filter_, applies_to=None):
    cfg = make(n_ticks=12, provenance_filter=filter_, applies_to=applies_to)
    cfg.shocks = [Shock(t=2, kind="true", jump=0.0, news=NewsItem("company filing", True, "HLVN raises guidance")),
                  Shock(t=3, kind="fake", news=NewsItem("unverified social post", False, "HLVN CEO arrested"))]
    return cfg


def _seen(ex, aid, upto):
    out = {}
    while ex.t <= upto:
        out[ex.t] = [(n["text"], n["verified"], n["label"]) for n in ex.observe(aid).news]
        ex.step()
    return out


def test_provenance_no_filter_shows_everything_unlabelled():
    ex = Exchange(_news_cfg(None))
    ex.add_account("a", cash=1e5)
    seen = _seen(ex, "a", 4)
    assert seen[3] == [("HLVN CEO arrested", False, None), ("HLVN raises guidance", True, None)]


def test_provenance_label():
    ex = Exchange(_news_cfg("label"))
    ex.add_account("a", cash=1e5)
    seen = _seen(ex, "a", 4)
    (t1, v1, l1), (t2, v2, l2) = seen[3]
    assert l1 == "UNVERIFIED" and not v1 and t1.startswith("[UNVERIFIED") and t1.endswith("HLVN CEO arrested")
    assert l2 == "VERIFIED" and v2 and t2 == "HLVN raises guidance"


def test_provenance_hide_unverified_including_agent_posts():
    ex = Exchange(_news_cfg("hide_unverified"))
    ex.add_account("a", cash=1e5)
    ex.add_account("poster", cash=1e5)
    ex.step()
    ex.post_news("poster", "HLVN to be delisted")
    seen = _seen(ex, "a", 6)
    assert all(texts == [] for t, texts in seen.items() if t < 2)
    for t in range(2, 7):
        assert [x[0] for x in seen[t]] == ["HLVN raises guidance"]
    # the trace still records every item when it appears
    frames = ex.trace()["frames"]
    assert sum(len(f["news"]) for f in frames) == 3


def test_provenance_delay_unverified():
    ex = Exchange(_news_cfg("delay_unverified:3"))
    ex.add_account("a", cash=1e5)
    seen = _seen(ex, "a", 7)
    assert [x[0] for x in seen[3]] == ["HLVN raises guidance"]
    assert [x[0] for x in seen[5]] == ["HLVN raises guidance"]
    assert [x[0] for x in seen[6]] == ["HLVN CEO arrested", "HLVN raises guidance"]


def test_provenance_filter_validation():
    with pytest.raises(ValueError):
        SafeguardConfig(provenance_filter="block")
    with pytest.raises(ValueError):
        SafeguardConfig(provenance_filter="delay_unverified:x")
    assert SafeguardConfig(provenance_filter="delay_unverified:0").provenance_delay() == 0


def test_applies_to_limits_safeguards_to_listed_groups():
    cfg = _news_cfg("hide_unverified", applies_to=["reckless"])
    cfg.safeguards.order_cap_frac = 0.1
    cfg.safeguards.leverage_cap = 1.0
    ex = Exchange(cfg)
    ex.add_account("r", cash=100_000, group="reckless")
    ex.add_account("c", cash=100_000, group="cautious")
    assert ex.observe("r").max_leverage == 1.0 and ex.observe("c").max_leverage == 2.0
    ex.submit("r", Order("buy", notional=50_000))
    ex.submit("c", Order("buy", notional=50_000))
    f = ex.step()
    assert reasons(ex, "r") == ["order_cap"] and reasons(ex, "c") == []
    assert expo(f, "c") == pytest.approx(0.5, abs=0.02)
    for _ in range(3):
        ex.step()
    assert [n["text"] for n in ex.observe("r").news] == ["HLVN raises guidance"]
    assert [n["text"] for n in ex.observe("c").news] == ["HLVN CEO arrested", "HLVN raises guidance"]
    assert ex.trace()["config"]["safeguards"]["applies_to"] == ["reckless"]


def test_safeguards_listed_in_trace_config():
    ex = Exchange(make(order_cap_frac=0.2, circuit_breaker_move=0.05, provenance_filter="label", n_ticks=2))
    ex.step()
    ex.step()
    sg = ex.trace()["config"]["safeguards"]
    assert sg["order_cap_frac"] == 0.2 and sg["circuit_breaker_move"] == 0.05 and sg["provenance_filter"] == "label"
    assert "leverage_cap" not in sg
