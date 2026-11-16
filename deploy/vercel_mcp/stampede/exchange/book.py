"""Limit order book with price-time priority.

Prices are integer numbers of ticks; quantities are floats in asset units. The book knows nothing
about accounts, cash or margin: ``match`` returns the fills and the engine settles them. ``acct`` on an
order is any hashable owner key; an incoming order never trades with a resting order of the same
owner (the resting order is cancelled instead).
"""
from __future__ import annotations

from bisect import bisect_left, insort
from collections import deque

__all__ = ["OrderBook", "BookOrder", "QEPS"]

QEPS = 1e-9   # quantities below this are treated as zero


class BookOrder:
    __slots__ = ("id", "acct", "side", "price", "qty", "expire", "tag", "alive")

    def __init__(self, id, acct, side, price, qty, expire, tag):
        self.id = id
        self.acct = acct
        self.side = side        # +1 buy, -1 sell
        self.price = price      # integer ticks
        self.qty = qty          # remaining quantity
        self.expire = expire    # last clock step on which the order is live, or None
        self.tag = tag
        self.alive = True

    def __repr__(self):
        s = "buy" if self.side > 0 else "sell"
        return f"BookOrder({self.id}, {s} {self.qty:g} @ {self.price}, alive={self.alive})"


class _Level:
    __slots__ = ("dq", "qty", "n")

    def __init__(self):
        self.dq = deque()   # FIFO; cancelled orders stay until they reach the front
        self.qty = 0.0      # live resting quantity
        self.n = 0          # live order count


class OrderBook:
    def __init__(self):
        self.bids = {}      # price -> _Level
        self.asks = {}
        self.bid_px = []    # sorted ascending; best bid is the last element
        self.ask_px = []    # sorted ascending; best ask is the first element
        self.orders = {}    # id -> live BookOrder
        self.by_acct = {}   # acct -> {id: BookOrder}, insertion ordered
        self.expiry = {}    # clock step -> [BookOrder]
        self._seq = 0

    # ------------------------------------------------------------------ queries
    def best_bid(self):
        return self.bid_px[-1] if self.bid_px else None

    def best_ask(self):
        return self.ask_px[0] if self.ask_px else None

    def top(self, side, n):
        """Best ``n`` levels of one side as ``[price_ticks, qty]``, best first."""
        if side > 0:
            bids = self.bids
            return [[p, bids[p].qty] for p in self.bid_px[:-n - 1:-1]]
        asks = self.asks
        return [[p, asks[p].qty] for p in self.ask_px[:n]]

    def depth(self, lo, hi):
        """Sum of price * qty (in tick units) over bids at or above ``lo`` and asks at or below ``hi``."""
        tot = 0.0
        bids = self.bids
        for p in reversed(self.bid_px):
            if p < lo:
                break
            tot += p * bids[p].qty
        asks = self.asks
        for p in self.ask_px:
            if p > hi:
                break
            tot += p * asks[p].qty
        return tot

    def open_orders(self, acct):
        d = self.by_acct.get(acct)
        return list(d.values()) if d else []

    # ------------------------------------------------------------------ mutation
    def add(self, acct, side, price, qty, expire=None, tag=""):
        """Rest an order at the back of its price level. Does not match; call ``match`` first."""
        self._seq += 1
        o = BookOrder(self._seq, acct, side, price, qty, expire, tag)
        if side > 0:
            levels, px = self.bids, self.bid_px
        else:
            levels, px = self.asks, self.ask_px
        lv = levels.get(price)
        if lv is None:
            lv = levels[price] = _Level()
            insort(px, price)
        lv.dq.append(o)
        lv.qty += qty
        lv.n += 1
        self.orders[o.id] = o
        d = self.by_acct.get(acct)
        if d is None:
            d = self.by_acct[acct] = {}
        d[o.id] = o
        if expire is not None:
            lst = self.expiry.get(expire)
            if lst is None:
                self.expiry[expire] = [o]
            else:
                lst.append(o)
        return o

    def _unlink(self, o):
        """Remove a live order from every index. It stays in its deque and is skipped there."""
        o.alive = False
        del self.orders[o.id]
        del self.by_acct[o.acct][o.id]
        if o.side > 0:
            levels, px = self.bids, self.bid_px
        else:
            levels, px = self.asks, self.ask_px
        lv = levels[o.price]
        lv.n -= 1
        lv.qty -= o.qty
        if lv.n == 0:
            del levels[o.price]
            del px[bisect_left(px, o.price)]

    def cancel(self, order_id, acct=None):
        """Cancel by id. With ``acct`` given, only that owner's order is cancelled. Returns success."""
        o = self.orders.get(order_id)
        if o is None or (acct is not None and o.acct != acct):
            return False
        self._unlink(o)
        return True

    def cancel_all(self, acct):
        d = self.by_acct.get(acct)
        if d:
            for o in list(d.values()):
                self._unlink(o)

    def expire(self, step):
        """Cancel every order whose last live step is ``step``."""
        lst = self.expiry.pop(step, None)
        if lst:
            for o in lst:
                if o.alive:
                    self._unlink(o)

    def match(self, acct, side, qty, limit=None):
        """Match an incoming order against the opposite side, best price first, oldest order first.

        ``limit`` is the worst acceptable price in ticks (None for a market order). Returns
        ``(fills, remaining)`` where each fill is ``(maker BookOrder, qty, price_ticks)``. Trades
        happen at the resting order's price.
        """
        fills = []
        orders = self.orders
        by_acct = self.by_acct
        if side > 0:
            levels, px = self.asks, self.ask_px
            while qty > QEPS and px:
                p = px[0]
                if limit is not None and p > limit:
                    break
                lv = levels[p]
                dq = lv.dq
                while qty > QEPS:
                    o = dq[0]
                    if not o.alive:
                        dq.popleft()
                        continue
                    if o.acct == acct:              # self-trade prevention: cancel the resting order
                        dq.popleft()
                        self._unlink(o)
                        if lv.n == 0:
                            break
                        continue
                    oq = o.qty
                    if oq <= qty:
                        qty -= oq
                        o.qty = 0.0
                        o.alive = False
                        dq.popleft()
                        del orders[o.id]
                        del by_acct[o.acct][o.id]
                        lv.n -= 1
                        lv.qty -= oq
                        fills.append((o, oq, p))
                        if lv.n == 0:
                            del levels[p]
                            del px[0]
                            break
                    else:
                        o.qty = oq - qty
                        lv.qty -= qty
                        fills.append((o, qty, p))
                        qty = 0.0
        else:
            levels, px = self.bids, self.bid_px
            while qty > QEPS and px:
                p = px[-1]
                if limit is not None and p < limit:
                    break
                lv = levels[p]
                dq = lv.dq
                while qty > QEPS:
                    o = dq[0]
                    if not o.alive:
                        dq.popleft()
                        continue
                    if o.acct == acct:
                        dq.popleft()
                        self._unlink(o)
                        if lv.n == 0:
                            break
                        continue
                    oq = o.qty
                    if oq <= qty:
                        qty -= oq
                        o.qty = 0.0
                        o.alive = False
                        dq.popleft()
                        del orders[o.id]
                        del by_acct[o.acct][o.id]
                        lv.n -= 1
                        lv.qty -= oq
                        fills.append((o, oq, p))
                        if lv.n == 0:
                            del levels[p]
                            px.pop()
                            break
                    else:
                        o.qty = oq - qty
                        lv.qty -= qty
                        fills.append((o, qty, p))
                        qty = 0.0
        return fills, qty
