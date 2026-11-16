"""Order book unit tests: price-time priority, partial fills, cancels, expiry, self-trade prevention."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from stampede.exchange.book import QEPS, OrderBook


def test_empty_book():
    b = OrderBook()
    assert b.best_bid() is None and b.best_ask() is None
    fills, rem = b.match("x", 1, 5.0)
    assert fills == [] and rem == 5.0
    assert b.top(1, 5) == [] and b.top(-1, 5) == []
    assert b.depth(0, 10 ** 9) == 0.0


def test_price_priority_before_time_priority():
    b = OrderBook()
    worse = b.add("a", -1, 102, 10.0)     # arrived first, worse price
    better = b.add("b", -1, 101, 10.0)
    fills, rem = b.match("t", 1, 15.0)
    assert rem == 0.0
    assert [(o.id, q, p) for o, q, p in fills] == [(better.id, 10.0, 101), (worse.id, 5.0, 102)]


def test_time_priority_within_a_level():
    b = OrderBook()
    first = b.add("a", 1, 100, 4.0)
    second = b.add("b", 1, 100, 4.0)
    third = b.add("c", 1, 100, 4.0)
    fills, rem = b.match("t", -1, 6.0)
    assert rem == 0.0
    assert [(o.id, q) for o, q, _ in fills] == [(first.id, 4.0), (second.id, 2.0)]
    assert not first.alive and second.alive and second.qty == 2.0 and third.qty == 4.0
    # the partially filled order keeps its place in the queue
    fills, _ = b.match("t", -1, 3.0)
    assert [(o.id, q) for o, q, _ in fills] == [(second.id, 2.0), (third.id, 1.0)]


def test_partial_fill_leaves_remainder_resting():
    b = OrderBook()
    o = b.add("a", -1, 100, 10.0)
    fills, rem = b.match("t", 1, 3.0)
    assert rem == 0.0 and fills[0][1] == 3.0
    assert o.alive and o.qty == 7.0
    assert b.top(-1, 5) == [[100, 7.0]]
    assert o.id in b.orders


def test_market_order_walks_levels_and_returns_unfilled_remainder():
    b = OrderBook()
    b.add("a", -1, 100, 1.0)
    b.add("a", -1, 101, 2.0)
    b.add("a", -1, 103, 3.0)
    fills, rem = b.match("t", 1, 10.0)
    assert [(q, p) for _, q, p in fills] == [(1.0, 100), (2.0, 101), (3.0, 103)]
    assert rem == 4.0
    assert b.best_ask() is None and not b.orders


def test_limit_price_is_respected_and_trades_at_resting_price():
    b = OrderBook()
    b.add("a", -1, 100, 1.0)
    b.add("a", -1, 105, 1.0)
    fills, rem = b.match("t", 1, 2.0, limit=102)
    assert [(q, p) for _, q, p in fills] == [(1.0, 100)]       # executes at 100, not at the limit 102
    assert rem == 1.0 and b.best_ask() == 105
    b.add("a", 1, 99, 1.0)
    fills, rem = b.match("t", -1, 1.0, limit=100)
    assert fills == [] and rem == 1.0                             # bid 99 is below the sell limit 100


def test_cancel_and_cancel_all():
    b = OrderBook()
    o1 = b.add("a", 1, 100, 1.0)
    o2 = b.add("a", 1, 100, 2.0)
    o3 = b.add("b", 1, 99, 3.0)
    assert not b.cancel(o1.id, acct="b")          # wrong owner
    assert b.cancel(o1.id, acct="a")
    assert not b.cancel(o1.id)                    # already gone
    assert b.top(1, 5) == [[100, 2.0], [99, 3.0]]
    fills, _ = b.match("t", -1, 2.0)              # the cancelled order is skipped, not filled
    assert [(o.id, q) for o, q, _ in fills] == [(o2.id, 2.0)]
    b.cancel_all("b")
    assert b.best_bid() is None and not o3.alive and b.open_orders("b") == []


def test_expiry():
    b = OrderBook()
    o1 = b.add("a", 1, 100, 1.0, expire=3)
    o2 = b.add("a", 1, 100, 1.0, expire=5)
    o3 = b.add("a", 1, 100, 1.0)
    b.expire(2)
    assert o1.alive
    b.expire(3)
    assert not o1.alive and o2.alive and o3.alive
    b.expire(5)
    assert not o2.alive and o3.alive and b.top(1, 1) == [[100, 1.0]]


def test_self_trade_prevention_cancels_the_resting_order():
    b = OrderBook()
    mine = b.add("a", -1, 100, 5.0)
    other = b.add("b", -1, 100, 5.0)
    fills, rem = b.match("a", 1, 3.0)
    assert [(o.id, q) for o, q, _ in fills] == [(other.id, 3.0)]
    assert not mine.alive and rem == 0.0
    assert b.top(-1, 1) == [[100, 2.0]]


def test_top_and_depth():
    b = OrderBook()
    for i, p in enumerate([100, 99, 98, 97, 96, 95, 94]):
        b.add("a", 1, p, 1.0 + i)
    for i, p in enumerate([101, 102, 103]):
        b.add("a", -1, p, 2.0)
    assert b.top(1, 5) == [[100, 1.0], [99, 2.0], [98, 3.0], [97, 4.0], [96, 5.0]]
    assert b.top(-1, 2) == [[101, 2.0], [102, 2.0]]
    assert b.depth(99, 102) == pytest.approx(100 * 1 + 99 * 2 + 101 * 2 + 102 * 2)


def test_levels_are_removed_when_empty():
    b = OrderBook()
    b.add("a", 1, 100, 1.0)
    b.add("a", 1, 99, 1.0)
    b.match("t", -1, 1.0)
    assert b.bid_px == [99] and 100 not in b.bids
    b.cancel_all("a")
    assert b.bid_px == [] and b.bids == {} and b.by_acct["a"] == {}


def test_full_close_is_exact():
    """Selling what rests in several orders consumes them all; float residue stays below QEPS."""
    b = OrderBook()
    for q in (0.1, 0.2, 0.3):
        b.add("a", 1, 100, q)
    total = 0.1 + 0.2 + 0.3
    fills, rem = b.match("t", -1, total)
    assert rem < QEPS and len(fills) == 3 and not b.orders
