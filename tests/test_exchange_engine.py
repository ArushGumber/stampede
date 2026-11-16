"""Exchange mechanics: matching through the engine, fees, conservation, margin, liquidation, insurance,
bankruptcy, TargetExposure rebalancing, raw orders, news, annotations and initial positions."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from stampede.exchange import (Cancel, Exchange, Hold, NewsItem, Order, SafeguardConfig, Shock, TargetExposure,
                               regime)


def quiet(name="equity", n_ticks=20, **kw):
    """An exchange with no background agents and a flat fundamental: only the test's accounts trade."""
    cfg = regime(name, seed=0, n_ticks=n_ticks, **kw)
    cfg.background.enabled = False
    cfg.fundamental.sigma = 0.0
    cfg.fundamental.vol_of_vol = 0.0
    return cfg


def live(name="equity", n_ticks=40, seed=0, **kw):
    return regime(name, seed=seed, n_ticks=n_ticks, **kw)


def run_to_end(ex):
    while not ex.done:
        ex.step()


def assert_conserved(ex, tol=1e-6):
    a = ex.audit()
    assert abs(a["position_sum"]) < 1e-7, a
    assert abs(a["cash_error"]) < tol * max(1.0, a["cash_deposited"]) * 1e-6 + 1e-4, a


# ---------------------------------------------------------------------------------------------- matching
def test_price_time_priority_across_ticks():
    ex = Exchange(quiet())
    for aid in ("first", "second", "better", "taker"):
        ex.add_account(aid, cash=1e6)
    ex.submit("first", Order("sell", qty=10, type="limit", limit_price=101.0))
    ex.step()
    ex.submit("second", Order("sell", qty=10, type="limit", limit_price=101.0))
    ex.step()
    ex.submit("better", Order("sell", qty=5, type="limit", limit_price=100.5))   # better price, arrives last
    ex.step()
    ex.submit("taker", Order("buy", qty=20, type="market"))
    ex.step()
    pos = {a: ex.accounts[a].position for a in ("first", "second", "better", "taker")}
    assert pos == pytest.approx({"first": -10, "second": -5, "better": -5, "taker": 20})
    obs = ex.observe("second")
    assert obs.asks[0] == [101.0, 5.0]
    assert len(obs.open_orders) == 1 and obs.open_orders[0]["qty"] == pytest.approx(5.0)


def test_partial_fill_and_remainder_rests():
    ex = Exchange(quiet())
    ex.add_account("mkr", cash=1e6)
    ex.add_account("tkr", cash=1e6)
    ex.submit("mkr", Order("sell", qty=3, type="limit", limit_price=100.0))
    ex.step()
    ex.submit("tkr", Order("buy", qty=10, type="limit", limit_price=100.0))   # takes 3, rests 7
    f = ex.step()
    assert ex.accounts["tkr"].position == pytest.approx(3.0)
    assert f["bids"][0] == [100.0, 7.0] and f["asks"] == []
    assert f["volume"] == pytest.approx(3.0)
    oo = ex.observe("tkr").open_orders
    assert len(oo) == 1 and oo[0]["side"] == "buy" and oo[0]["qty"] == pytest.approx(7.0)
    # a market order that finds only part of the liquidity fills what it can
    ex.submit("mkr", Order("sell", qty=10, type="market"))
    ex.step()
    assert ex.accounts["mkr"].position == pytest.approx(-10.0)
    assert ex.accounts["tkr"].position == pytest.approx(10.0)
    assert_conserved(ex)


def test_trades_execute_at_resting_price_and_fees_go_to_fee_account():
    cfg = quiet()
    cfg.fee_bps = 2.0
    ex = Exchange(cfg)
    ex.add_account("mkr", cash=1e6)
    ex.add_account("tkr", cash=1e6)
    ex.submit("mkr", Order("sell", qty=10, type="limit", limit_price=100.0))
    ex.step()
    ex.submit("tkr", Order("buy", qty=10, type="limit", limit_price=105.0))
    ex.step()
    fee = 1000.0 * 2e-4
    assert ex.accounts["tkr"].cash == pytest.approx(1e6 - 1000.0 - fee)
    assert ex.accounts["mkr"].cash == pytest.approx(1e6 + 1000.0 - fee)
    assert ex.audit()["fees"] == pytest.approx(2 * fee)
    assert_conserved(ex)


def test_order_ttl_expiry_and_cancel():
    ex = Exchange(quiet())
    ex.add_account("a", cash=1e6)
    ex.submit("a", Order("buy", qty=1, type="limit", limit_price=90.0, ttl=2))
    ex.submit("a", Order("buy", qty=1, type="limit", limit_price=91.0))
    ex.step()                                                  # tick 0: both rest
    oo = {o["price"]: o for o in ex.observe("a").open_orders}
    assert oo[90.0]["ttl"] == 1 and oo[91.0]["ttl"] is None
    ex.step()                                                  # tick 1: the ttl=2 order expires at the end
    oo = ex.observe("a").open_orders
    assert [o["price"] for o in oo] == [91.0]
    ex.submit("a", Cancel(oo[0]["id"]))
    ex.submit("a", Cancel(12345))
    ex.step()
    assert ex.observe("a").open_orders == []
    rej = ex.last_rejections("a")
    assert [r["reason"] for r in rej] == ["unknown_order"]


def test_order_validation_rejections_do_not_raise():
    ex = Exchange(quiet())
    ex.add_account("a", cash=1e6)
    ex.submit("a", Order("buy", qty=1, notional=100))                # both size fields
    ex.submit("a", Order("buy"))                                     # neither
    ex.submit("a", Order("hold", qty=1))                             # bad side
    ex.submit("a", Order("buy", qty=1, type="limit"))                # no limit price
    ex.submit("a", Order("buy", qty=-1))                             # negative size
    ex.submit("a", TargetExposure(x=float("nan")))
    ex.submit("a", Hold())
    ex.step()
    reasons = [r["reason"] for r in ex.last_rejections("a")]
    assert reasons.count("invalid_order") == 5 and reasons.count("invalid_intent") == 1
    assert ex.observe("a").rejections == ex.last_rejections("a")


def test_insufficient_margin_rejection():
    ex = Exchange(quiet())                                     # equity regime, max leverage 2
    ex.add_account("mkr", cash=1e7)
    ex.add_account("a", cash=10_000)
    ex.submit("mkr", Order("sell", qty=1000, type="limit", limit_price=100.0))
    ex.step()
    ex.submit("a", Order("buy", notional=25_000))              # 2.5x equity
    ex.step()
    assert [r["reason"] for r in ex.last_rejections("a")] == ["insufficient_margin"]
    assert ex.accounts["a"].position == 0
    ex.submit("a", Order("buy", notional=19_000))
    ex.step()
    assert ex.accounts["a"].position == pytest.approx(190.0)


def test_no_liquidity_is_reported():
    ex = Exchange(quiet())
    ex.add_account("a", cash=1e6)
    ex.submit("a", Order("buy", qty=5))
    ex.step()
    assert [r["reason"] for r in ex.last_rejections("a")] == ["no_liquidity"]


# ---------------------------------------------------------------------------------------------- target exposure
@pytest.mark.parametrize("x", [1.0, 1.5, -1.0, -2.0, 0.5])
def test_target_exposure_reaches_target_long_and_short(x):
    ex = Exchange(live("equity", n_ticks=10))
    ex.add_account("a", cash=100_000)
    ex.submit("a", TargetExposure(x=x, style="market"))
    f = ex.step()
    row = f["agents"][0]
    assert row["action"] == x
    assert row["exposure"] == pytest.approx(x, abs=0.02)
    assert math.copysign(1, ex.accounts["a"].position) == math.copysign(1, x)
    # rebalance back to flat
    ex.submit("a", TargetExposure(x=0.0))
    ex.step()
    assert abs(ex.accounts["a"].position) < 1e-9
    assert_conserved(ex)


def test_target_exposure_is_clipped_to_leverage():
    ex = Exchange(live("equity", n_ticks=10))
    ex.add_account("a", cash=100_000)
    ex.add_account("b", cash=100_000)
    ex.submit("a", TargetExposure(x=5.0))
    ex.submit("b", TargetExposure(x=-7.0))
    f = ex.step()
    rows = {r["id"]: r for r in f["agents"]}
    assert rows["a"]["exposure"] == pytest.approx(2.0, abs=0.03)
    assert rows["b"]["exposure"] == pytest.approx(-2.0, abs=0.03)
    assert rows["a"]["action"] == 5.0                          # the requested value is what is recorded
    assert any(r["reason"] == "leverage_clipped" for r in ex.last_rejections("a"))
    assert any(r["reason"] == "leverage_clipped" for r in ex.last_rejections("b"))


def test_target_exposure_without_shorting():
    cfg = live("equity", n_ticks=10)
    cfg.allow_short = False
    ex = Exchange(cfg)
    ex.add_account("a", cash=100_000)
    ex.submit("a", TargetExposure(x=-1.0))
    ex.step()
    assert ex.accounts["a"].position == 0
    assert any(r["reason"] == "leverage_clipped" for r in ex.last_rejections("a"))
    ex.submit("a", Order("sell", qty=1))
    ex.step()
    assert [r["reason"] for r in ex.last_rejections("a")] == ["short_not_allowed"]


def test_crypto_high_leverage_target():
    ex = Exchange(live("crypto", n_ticks=10))
    ex.add_account("a", cash=20_000)
    ex.submit("a", TargetExposure(x=8.0))
    f = ex.step()
    assert f["agents"][0]["exposure"] == pytest.approx(8.0, rel=0.03)
    assert ex.observe("a").max_leverage == 10.0


def test_passive_target_rests_at_the_touch_and_expires():
    cfg = live("equity", n_ticks=20)
    cfg.passive_ttl = 3
    ex = Exchange(cfg)
    ex.add_account("a", cash=100_000)
    bid = ex.observe("a").best_bid
    ex.submit("a", TargetExposure(x=1.0, style="passive"))
    ex.step()
    oo = ex.observe("a").open_orders
    assert ex.accounts["a"].position >= 0
    if oo:                                       # not yet completely filled
        assert oo[0]["side"] == "buy" and abs(oo[0]["price"] / bid - 1) < 1e-3     # at the touch, not crossing
    for _ in range(3):
        ex.step()
    assert ex.observe("a").open_orders == []     # expired after passive_ttl ticks


def test_new_target_replaces_queued_target_within_a_tick():
    ex = Exchange(live("equity", n_ticks=10))
    ex.add_account("a", cash=100_000)
    ex.submit("a", TargetExposure(x=2.0))
    ex.submit("a", TargetExposure(x=0.5))
    f = ex.step()
    assert f["agents"][0]["exposure"] == pytest.approx(0.5, abs=0.02)
    assert f["agents"][0]["action"] == 0.5


# ---------------------------------------------------------------------------------------------- margin, liquidation
def _liquidation_setup(bid, ask):
    ex = Exchange(quiet())
    ex.add_account("mkr", cash=1e7, group="dealer")
    ex.add_account("lng", cash=10_000, group="reckless", initial_exposure=2.0)
    assert ex.accounts["lng"].position == pytest.approx(200.0)
    assert ex.accounts["__custodian__"].position == pytest.approx(-200.0)
    obs = ex.observe("lng")
    assert obs.exposure == pytest.approx(2.0)
    # liquidation when -10000 + 200 p < 0.25 * 200 p, i.e. p < 66.67
    assert obs.liq_price == pytest.approx(10_000 / (200 * 0.75))
    ex.submit("mkr", Order("buy", qty=1000, type="limit", limit_price=bid))
    ex.submit("mkr", Order("sell", qty=10, type="limit", limit_price=ask))
    return ex


def test_forced_liquidation_with_equity_left():
    ex = _liquidation_setup(60.0, 62.0)
    f = ex.step()
    lng = ex.accounts["lng"]
    assert lng.position == 0
    assert f["liquidations"] == 1 and f["liquidation_volume"] == pytest.approx(200.0)
    row = {r["id"]: r for r in f["agents"]}["lng"]
    assert row["liquidated"] is True
    assert lng.cash == pytest.approx(-10_000 + 12_000 - 12_000 * 1e-4)
    assert not lng.bankrupt
    m = ex.metrics()
    assert m["n_liquidations"] == 1 and m["liquidation_volume"] == pytest.approx(200.0)
    assert m["insurance_shortfall"] == 0.0 and m["bankruptcies_by_group"]["reckless"] == 0
    assert m["fire_sale_share"] == pytest.approx(1.0)
    assert_conserved(ex)


def test_liquidation_shortfall_goes_to_insurance_fund_and_bankrupts():
    ex = _liquidation_setup(40.0, 42.0)
    ex.step()
    lng = ex.accounts["lng"]
    short = 10_000 - 8_000 + 8_000 * 1e-4
    assert lng.position == 0 and lng.cash == 0 and lng.bankrupt
    a = ex.audit()
    assert a["insurance"] == pytest.approx(-short) and a["insurance_shortfall"] == pytest.approx(short)
    m = ex.metrics()
    assert m["insurance_shortfall"] == pytest.approx(short)
    assert m["bankruptcies_by_group"] == {"dealer": 0, "reckless": 1}
    assert m["pnl_by_group"]["reckless"] == pytest.approx(-1.0)
    ex.submit("lng", TargetExposure(1.0))
    ex.step()
    assert [r["reason"] for r in ex.last_rejections("lng")] == ["bankrupt"]
    assert_conserved(ex)


def test_short_liquidation():
    ex = Exchange(quiet())
    ex.add_account("mkr", cash=1e7)
    ex.add_account("sht", cash=10_000, initial_exposure=-2.0)       # short 200 @ 100, cash 30000
    obs = ex.observe("sht")
    assert obs.exposure == pytest.approx(-2.0)
    # liquidation when 30000 - 200 p < 0.25 * 200 p, i.e. p > 120
    assert obs.liq_price == pytest.approx(30_000 / (200 * 1.25))
    ex.submit("mkr", Order("buy", qty=10, type="limit", limit_price=124.0))
    ex.submit("mkr", Order("sell", qty=1000, type="limit", limit_price=126.0))
    f = ex.step()
    assert ex.accounts["sht"].position == 0 and f["liquidations"] == 1
    assert_conserved(ex)


def test_liquidation_cascade_with_shock_conserves_everything():
    cfg = live("crypto", n_ticks=40, seed=3)
    cfg.shocks = [Shock(t=10, kind="true", jump=-0.12, news=NewsItem("wire", True, "bad"))]
    ex = Exchange(cfg)
    D = cfg.background.depth_notional
    for i in range(16):
        ex.add_account(f"a{i:02d}", cash=D / 16, group="reckless" if i % 2 else "cautious",
                       initial_exposure=9.0 if i % 2 else 2.0)
    run_to_end(ex)
    m = ex.metrics()
    assert m["n_liquidations"] >= 8          # every 9x long is past its liquidation price (-6.4%)
    assert m["bankruptcies_by_group"]["reckless"] >= 1
    assert m["liquidation_volume"] > 0 and 0 < m["fire_sale_share"] <= 1
    assert_conserved(ex)
    for a in ex._ext:                     # bankrupt accounts were closed out by the insurance fund
        assert a.cash + a.position * ex.price > -0.01 * a.cash0
        if a.bankrupt:
            assert a.position == 0 and a.cash == 0
    assert m["insurance_shortfall"] > 0


def test_background_accounts_conserve_over_long_run():
    ex = Exchange(live("equity", n_ticks=300, seed=5))
    for i in range(8):
        ex.add_account(f"a{i}", cash=50_000, initial_exposure=(i % 5) - 2)
    while not ex.done:
        for i in range(8):
            if (ex.t + i) % 7 == 0:
                ex.submit(f"a{i}", TargetExposure(((ex.t // 7 + i) % 5) - 2))
        ex.step()
    a = ex.audit()
    assert abs(a["position_sum"]) < 1e-7
    assert abs(a["cash_error"]) < 1e-3
    assert a["fees"] > 0


# ---------------------------------------------------------------------------------------------- initial positions
def test_initial_exposure_uses_custodian_and_no_market_impact():
    ex = Exchange(live("equity", n_ticks=5))
    p = ex.price
    book_before = (list(ex.book.bid_px), list(ex.book.ask_px))
    ex.add_account("a", cash=100_000, initial_exposure=1.5)
    ex.add_account("b", cash=50_000, initial_exposure=-1.0)
    ex.add_account("c", cash=50_000, initial_exposure=9.0)      # clipped to the equity cap of 2
    assert (list(ex.book.bid_px), list(ex.book.ask_px)) == book_before
    assert ex.price == p
    for aid, x in (("a", 1.5), ("b", -1.0), ("c", 2.0)):
        obs = ex.observe(aid)
        assert obs.exposure == pytest.approx(x)
        assert obs.equity == pytest.approx(ex.accounts[aid].cash0)
        assert obs.pnl_pct == pytest.approx(0.0)
    cust = ex.accounts["__custodian__"]
    assert cust.position == pytest.approx(-(1.5 * 100_000 + -1.0 * 50_000 + 2.0 * 50_000) / p)
    assert ex.audit()["position_sum"] == pytest.approx(0.0, abs=1e-9)
    trace_ids = [r["id"] for r in ex.trace()["agents"]]
    assert trace_ids == ["a", "b", "c"]                        # the custodian is not an agent
    run_to_end(ex)
    assert cust.position == pytest.approx(-(1.5 * 100_000 + -1.0 * 50_000 + 2.0 * 50_000) / p)  # never trades
    assert_conserved(ex)


# ---------------------------------------------------------------------------------------------- news, annotations
def test_post_news_visible_from_next_tick():
    ex = Exchange(live("equity", n_ticks=10))
    ex.add_account("a", cash=1e5)
    ex.add_account("b", cash=1e5)
    ex.step()
    item = ex.post_news("a", "Halvorsen CFO resigns")
    assert ex.observe("b").news == []
    f1 = ex.step()
    assert f1["news"] == []
    news = ex.observe("b").news
    assert len(news) == 1 and news[0]["text"] == "Halvorsen CFO resigns"
    assert news[0]["author_agent"] == "a" and news[0]["verified"] is False and news[0]["t"] == 2
    assert news[0].id == item.id
    f2 = ex.step()
    assert [n["id"] for n in f2["news"]] == [item.id] and f2["news"][0]["author_agent"] == "a"
    assert ex.step()["news"] == []                             # frames hold only first appearances


def test_shock_news_and_correction():
    cfg = live("equity", n_ticks=12)
    cfg.shocks = [Shock(t=3, kind="fake", news=NewsItem("unverified social post", False, "HLVN fraud probe")),
                  Shock(t=6, kind="correction")]
    ex = Exchange(cfg)
    ex.add_account("a", cash=1e5)
    frames = []
    seen = {}
    while not ex.done:
        seen[ex.t] = [n["text"] for n in ex.observe("a").news]
        frames.append(ex.step())
    assert seen[2] == [] and seen[3] == ["HLVN fraud probe"]
    assert seen[6][0].startswith("Correction") and "HLVN fraud probe" in seen[6][0]
    assert [len(f["news"]) for f in frames] == [0, 0, 0, 1, 0, 0, 1] + [0] * 5
    assert frames[6]["news"][0]["verified"] is True


def test_news_is_newest_first_and_at_most_five():
    cfg = live("equity", n_ticks=10)
    cfg.shocks = [Shock(t=i, kind="fake", news=NewsItem("blog", False, f"item {i}")) for i in range(7)]
    ex = Exchange(cfg)
    ex.add_account("a", cash=1e5)
    for _ in range(7):
        ex.step()
    assert [n["text"] for n in ex.observe("a").news] == [f"item {i}" for i in (6, 5, 4, 3, 2)]


def test_annotate_and_default_action():
    ex = Exchange(live("equity", n_ticks=10))
    for aid in ("a", "b", "c", "d"):
        ex.add_account(aid, cash=1e5)
    ex.submit("a", TargetExposure(1.0))
    ex.annotate("a", proj={"risk": 0.8}, reason="momentum")
    ex.submit("b", TargetExposure(1.0))
    ex.annotate("b", action=0.75)
    ex.submit("c", Order("buy", qty=1))
    ex.annotate("d", reason="waiting")
    f = ex.step()
    rows = {r["id"]: r for r in f["agents"]}
    assert rows["a"]["action"] == 1.0 and rows["a"]["proj"] == {"risk": 0.8} and rows["a"]["reason"] == "momentum"
    assert rows["b"]["action"] == 0.75
    assert rows["c"]["action"] is None                          # raw orders have no target
    assert rows["d"]["action"] is None and rows["d"]["reason"] == "waiting"
    f = ex.step()                                               # annotations do not carry over
    assert all(r["action"] is None and r["proj"] is None and r["reason"] is None for r in f["agents"])


def test_observation_fields_and_cheapness():
    import time
    ex = Exchange(live("equity", n_ticks=5))
    for i in range(32):
        ex.add_account(f"a{i}", cash=1e5, initial_exposure=1.0)
    obs = ex.observe("a0")
    assert obs.symbol == "HLVN" and obs.asset_name == "Halvorsen Industries"
    assert 0 < obs.best_bid < obs.price < obs.best_ask
    assert len(obs.bids) == 5 and len(obs.asks) == 5
    assert obs.bids[0][0] > obs.bids[1][0] and obs.asks[0][0] < obs.asks[1][0]
    assert obs.spread_bps == pytest.approx((obs.best_ask - obs.best_bid) / obs.price * 1e4, rel=1e-6)
    assert obs.margin_ratio == pytest.approx(1.0) and obs.liq_price is None   # 1x long can't be liquidated
    t0 = time.perf_counter()
    for _ in range(100):
        for i in range(32):
            ex.observe(f"a{i}")
    per_call = (time.perf_counter() - t0) / 3200
    assert per_call < 1e-4, per_call
