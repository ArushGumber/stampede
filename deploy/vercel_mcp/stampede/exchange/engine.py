"""The Stampede exchange: accounts, margin, safeguards, background ecology and the tick loop.

Clock
-----
The engine keeps an absolute step counter ``k``. The first ``warmup_ticks`` steps are run inside the
constructor with background agents only, so that at public tick ``t = 0`` the book is populated and
return history exists. Public tick ``t`` is step ``k = warmup_ticks + t``.

One public tick, in order:

1. (already done when ``observe`` is called) the fundamental for tick ``t`` is in force, including a
   shock jump scheduled at ``t``, and news scheduled for ``t`` is published;
2. accounts call ``observe`` and ``submit``;
3. ``step()``: background agents decide from the same start-of-tick state, all intents (background,
   released approvals, external) are shuffled with the seeded generator and matched one by one against
   the book; then liquidations, stop-losses and order expiry; the frame is recorded; the circuit breaker
   is evaluated; the clock advances.

All state lives on the instance. There are no module-level generators, counters or caches, so many
``Exchange`` objects can be stepped in lockstep in one process.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .book import OrderBook, QEPS
from .config import MarketConfig, NewsItem, effective_background
from .intents import Cancel, Hold, Observation, Order, TargetExposure
from .metrics import compute_metrics

__all__ = ["Exchange", "Account", "Record"]

# internal intent codes
_L, _M, _R, _X, _XA, _LN = 0, 1, 2, 4, 5, 6
_LIQ, _STOP = "liq", "stop"
_NEVER = -10 ** 9
_BANKRUPT_FRAC = 1e-3     # equity left after a liquidation, as a fraction of starting equity, below which the account is bankrupt
_SYSTEM = ("__fees__", "__insurance__", "__custodian__")
_UNVERIFIED_PREFIX = "[UNVERIFIED: not confirmed by any official source] "


def _plain(v):
    """Numbers (numpy scalars included) to float, mappings to plain dicts, so frames stay JSON-ready."""
    if isinstance(v, (bool, str)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


class Record(dict):
    """A dict whose keys can also be read as attributes (``item["text"]`` or ``item.text``)."""
    __slots__ = ()

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class Account:
    __slots__ = (
        "id", "group", "kind", "role", "model", "alphas", "cash", "cash0", "position", "position0",
        "mandate", "external", "exempt", "sg", "lev", "bankrupt", "frozen", "peak", "last_reb",
        "queue", "rej", "rej_last", "action", "action_set", "target_x", "proj", "reason",
        "liq_k", "n_liq", "fill_k", "breaches", "hist_pos", "hist_eq", "t_added",
    )

    def __init__(self, id, cash, group, kind, role=None):
        self.id = id
        self.group = group
        self.kind = kind
        self.role = role
        self.model = None
        self.alphas = None
        self.cash = float(cash)
        self.cash0 = float(cash)       # starting equity
        self.position = 0.0
        self.position0 = 0.0
        self.mandate = None
        self.external = False
        self.exempt = True
        self.sg = False                # per-account safeguards apply
        self.lev = 1.0                 # leverage cap in force
        self.bankrupt = False
        self.frozen = False
        self.peak = float(cash)
        self.last_reb = _NEVER
        self.queue = []
        self.rej = []
        self.rej_last = []
        self.action = None
        self.action_set = False
        self.target_x = None
        self.proj = None
        self.reason = None
        self.liq_k = _NEVER
        self.n_liq = 0
        self.fill_k = _NEVER
        self.breaches = 0
        self.hist_pos = []
        self.hist_eq = []
        self.t_added = 0

    def __repr__(self):
        return f"Account({self.id!r}, cash={self.cash:.2f}, position={self.position:.6f})"


class Exchange:
    def __init__(self, cfg: MarketConfig):
        self.cfg = cfg
        self.meta = {}                                  # free-form run metadata merged into trace()["config"]
        bg = self._bg = effective_background(cfg.background, float(cfg.fundamental.sigma))
        sg = cfg.safeguards
        self._sg = sg
        self.tick = float(cfg.tick_size)
        self._fee_rate = cfg.fee_bps * 1e-4
        self._bg_on = bool(bg.enabled)
        self._W = int(cfg.warmup_ticks) if self._bg_on else 0
        self._K_end = self._W + int(cfg.n_ticks)
        self._k = 0

        ss = np.random.SeedSequence(int(cfg.seed))
        s_fund, s_bg, s_match = ss.spawn(3)
        self._rng_bg = np.random.default_rng(s_bg)
        self._rng_match = np.random.default_rng(s_match)

        self.book = OrderBook()
        self.accounts = {}
        self._ext = []              # external accounts in creation order
        self._margin_accts = []     # accounts subject to maintenance margin
        self._stop_accts = []       # accounts subject to the stop-loss safeguard
        self._deposits = 0.0
        self._fee_acct = self._new_system_account("__fees__", 0.0)
        self._ins = self._new_system_account("__insurance__", cfg.insurance_fund_cash)
        self._cust = self._new_system_account("__custodian__", 0.0)

        # safeguards
        self._prov_delay = sg.provenance_delay()
        pf = sg.provenance_filter
        self._prov_mode = None if pf is None else ("label" if pf == "label" else "hide" if pf == "hide_unverified" else "delay")
        self._stop = sg.account_stop_loss
        self._cb_move = sg.circuit_breaker_move
        self._halt_until = 0
        self._cb_ref = 0
        self._held = []             # (release step, account, intent) waiting for approval delay
        self._expired_to = -1

        # fundamental and news schedule
        self._F = self._build_fundamental(np.random.default_rng(s_fund))
        self._news = []             # published items, oldest first
        self._news_sched = {}       # step -> [NewsItem]
        self._news_new = []
        self._news_n = 0
        self._news_cache = {}
        self._schedule_shocks()

        # histories, indexed by step; _px has one extra leading element (the price before step 0)
        p0 = self._F[0]
        self._px = [p0]
        self._lr = []
        self._spread = []
        self._depth = []
        self._volume = []
        self._liq_volume = []
        self._liq_count = []
        self._halted = []
        self._vol_k = 0.0
        self._liq_vol_k = 0.0
        self._n_liq_k = 0
        self._flow_k = 0.0                 # signed taker notional of the tick being played (buys positive)
        self._hi_k = 0                     # highest and lowest trade price of the tick, in ticks
        self._lo_k = 1 << 60
        self._lt_k = 0                     # last trade price of the tick, in ticks (0: none yet)
        self._mm_c = p0 / self.tick        # market makers' quote centre, in ticks
        self._shortfall = 0.0
        self._frames = []
        self._snap = None

        self._init_background()
        for _ in range(self._W):
            self._step(False)
        self._begin_tick()
        self._refresh_snapshot()

    # ================================================================== construction helpers
    def _new_system_account(self, id, cash):
        a = Account(id, cash, "system", "system")
        self.accounts[id] = a
        self._deposits += a.cash
        return a

    def _build_fundamental(self, rng):
        cfg = self.cfg
        f = cfg.fundamental
        W, n = self._W, int(cfg.n_ticks)
        N = W + n + 1
        z = rng.standard_normal(N)
        if f.vol_of_vol and f.vol_of_vol > 0:
            eta = rng.standard_normal(N)
            s, phi = float(f.vol_of_vol), float(f.vol_persistence)
            c = s * math.sqrt(max(0.0, 1.0 - phi * phi))
            h = np.empty(N)
            h[0] = s * eta[0]
            for i in range(1, N):
                h[i] = phi * h[i - 1] + c * eta[i]
            sig = f.sigma * np.exp(h - s * s)       # E[sig^2] = sigma^2
        else:
            sig = np.full(N, float(f.sigma))
        r = f.drift + sig * z                       # r[k]: log return from step k-1 to k
        if f.replay_returns is not None:
            r[W + 1:] = 0.0
            rep = np.asarray(f.replay_returns, dtype=float)[:n]
            r[W + 1:W + 1 + len(rep)] = rep
        r[0] = 0.0
        logf = np.cumsum(r)
        logf += math.log(cfg.initial_price) - logf[W]
        for s in cfg.shocks:
            if s.kind == "true" and s.jump and 0 <= s.t <= n:
                logf[W + s.t:] += s.jump
        return np.exp(logf).tolist()

    def _schedule_shocks(self):
        W = self._W
        last_fake = None
        for s in sorted(self.cfg.shocks, key=lambda s: s.t):
            item = s.news
            if s.kind == "fake" and item is not None:
                last_fake = item
            if item is None and s.kind == "correction":
                quoted = f' "{last_fake.text}"' if last_fake is not None else ""
                item = NewsItem(source="exchange notice", verified=True,
                                text=f"Correction: the earlier report{quoted} was false.")
            if item is None:
                continue
            self._news_sched.setdefault(W + int(s.t), []).append(replace(item))

    def _init_background(self):
        cfg, bg = self.cfg, self._bg
        self._mms, self._noise, self._funds, self._moms = [], [], [], []
        if not self._bg_on:
            return
        D = float(bg.depth_notional)
        exempt = bool(bg.margin_exempt)

        def make(prefix, n, role, cash, dest):
            for i in range(n):
                a = Account(f"bg_{prefix}_{i:02d}", cash, "background", "background", role)
                a.exempt = exempt
                a.lev = float("inf") if exempt else cfg.max_leverage
                self.accounts[a.id] = a
                self._deposits += a.cash
                dest.append(a)
                if not exempt:
                    self._margin_accts.append(a)

        make("mm", bg.n_market_makers, "mm", 4.0 * D, self._mms)
        make("noise", bg.n_noise, "noise", 1.0 * D, self._noise)
        make("fund", bg.n_fundamentalists, "fund", 2.0 * D, self._funds)
        make("mom", bg.n_momentum, "mom", 0.5 * D, self._moms)

        sigma = float(cfg.fundamental.sigma)
        self._mm_vol_ref = float(bg.mm_vol_ref) if bg.mm_vol_ref else 1.2 * sigma
        nm = max(1, bg.n_market_makers)
        self._mm_level_notional = bg.mm_level_frac * D / nm
        self._mm_inv_limit = bg.mm_inventory_limit_frac * D / nm
        self._mm_impact = bg.mm_flow_impact / D
        self._mm_track = bg.mm_track_fundamental * sigma
        self._noise_mean = bg.noise_order_frac * D
        self._noise_inv = bg.noise_inventory_frac * D
        self._fund_order = bg.fund_order_frac * D
        self._fund_limit = bg.fund_position_limit_frac * D
        self._mom_cap = bg.mom_position_frac * D
        self._mom_order = bg.mom_order_frac * D
        self._mom_sig_floor = 0.5 * sigma
        self._min_notional = 1e-6 * D
        self._fund_err = [0.0] * len(self._funds)       # persistent valuation errors (log units)
        self._fund_err_init = False
        self._fund_clip = bg.fund_track_clip * sigma
        self._flow_mem = 0.0                             # decaying signed taker flow (currency)

    # ================================================================== public API
    @property
    def t(self) -> int:
        return self._k - self._W

    @property
    def done(self) -> bool:
        return self._k >= self._K_end

    @property
    def price(self) -> float:
        return self._px[-1]

    @property
    def fundamental(self) -> float:
        return self._F[min(self._k, len(self._F) - 1)]

    def add_account(self, agent_id, cash, group="llm", mandate_max_exposure=None, initial_exposure=0.0,
                    *, kind="llm", model=None, alphas=None):
        """Open an external account with ``cash`` of starting equity.

        With a non-zero ``initial_exposure`` the account starts holding a position worth
        ``initial_exposure`` times ``cash`` (clipped to its leverage cap), acquired at the current
        price with no market impact and no fee. The offsetting position is booked to the passive
        ``__custodian__`` account, which never trades, so positions still sum to zero.
        ``kind``, ``model`` and ``alphas`` are only copied into ``trace()["agents"]``.
        """
        if agent_id in self.accounts:
            raise ValueError(f"account {agent_id!r} already exists")
        cfg, sg = self.cfg, self._sg
        a = Account(agent_id, cash, group, kind)
        a.external = True
        a.exempt = False
        a.model = model
        a.alphas = alphas
        a.mandate = mandate_max_exposure
        a.t_added = self.t
        a.sg = sg.applies_to is None or group in sg.applies_to
        a.lev = float(cfg.max_leverage)
        if a.sg and sg.leverage_cap is not None:
            a.lev = min(a.lev, float(sg.leverage_cap))
        self.accounts[agent_id] = a
        self._ext.append(a)
        self._margin_accts.append(a)
        if a.sg and self._stop is not None:
            self._stop_accts.append(a)
        self._deposits += a.cash
        x = float(initial_exposure or 0.0)
        if x:
            x = max(-a.lev, min(a.lev, x))
            if x < 0 and not cfg.allow_short:
                x = 0.0
            px = self._px[-1]
            qty = x * a.cash / px
            a.position = qty
            a.cash -= qty * px
            self._cust.position -= qty
            self._cust.cash += qty * px
        a.position0 = a.position
        return a

    def submit(self, agent_id, intent, proj=None, reason=None):
        """Queue an intent for the next ``step()``. Several ``Order``/``Cancel`` intents may be queued
        in one tick; a new ``TargetExposure`` replaces one queued earlier in the same tick."""
        a = self.accounts[agent_id]
        if self.done:
            return
        if type(intent) is TargetExposure:
            if a.target_x is not None:
                a.queue = [it for it in a.queue if type(it) is not TargetExposure]
            try:
                a.target_x = float(intent.x)
            except (TypeError, ValueError):
                a.target_x = None
        a.queue.append(intent)
        if proj is not None:
            a.proj = _plain(proj)
        if reason is not None:
            a.reason = reason

    def annotate(self, agent_id, action=None, proj=None, reason=None):
        """Attach ``action``, ``proj`` and ``reason`` to this agent's record in the next frame.
        Without an ``action`` annotation, ``action`` is the ``x`` of a submitted TargetExposure."""
        a = self.accounts[agent_id]
        if action is not None:
            a.action = _plain(action)
            a.action_set = True
        if proj is not None:
            a.proj = _plain(proj)
        if reason is not None:
            a.reason = reason

    def post_news(self, agent_id, text):
        """Publish an unverified item authored by ``agent_id``, visible from the next tick."""
        if agent_id not in self.accounts:
            raise KeyError(agent_id)
        self._news_n += 1
        item = NewsItem(source="agent post", verified=False, text=str(text),
                        id=f"n{self._news_n}", t=self.t + 1, author_agent=agent_id)
        self._news_sched.setdefault(self._k + 1, []).append(item)
        return item

    def last_rejections(self, agent_id):
        """Rejections and notices recorded for ``agent_id`` during the most recent ``step()``."""
        return list(self.accounts[agent_id].rej_last)

    def observe(self, agent_id) -> Observation:
        a = self.accounts[agent_id]
        s = self._snap
        cfg = self.cfg
        px = s[0]
        pos = a.position
        cash = a.cash
        eq = cash + pos * px
        notional = abs(pos) * px
        if notional > 1e-9:
            exposure = pos * px / eq if eq > 0 else 0.0
            margin_ratio = eq / notional
            mm = cfg.maintenance_margin
            lp = -cash / (pos * (1.0 - mm)) if pos > 0 else cash / (-pos * (1.0 + mm))
            liq_price = lp if lp > 0 else None
        else:
            exposure = 0.0
            margin_ratio = None
            liq_price = None
        d = self.book.by_acct.get(a)
        if d:
            k, tick = self._k, self.tick
            open_orders = [Record(id=o.id, side="buy" if o.side > 0 else "sell", price=round(o.price * tick, 8),
                                  qty=o.qty, ttl=None if o.expire is None else o.expire - k + 1)
                           for o in d.values()]
        else:
            open_orders = []
        return Observation(
            self._k - self._W, cfg.n_ticks, cfg.symbol, cfg.asset_name,
            px, s[1], s[2], s[3], list(s[4]), list(s[5]), s[6], s[7], s[8], s[9], s[10],
            cash, pos, eq, exposure, eq / a.cash0 - 1.0 if a.cash0 else 0.0,
            liq_price, margin_ratio, open_orders, a.lev, a.mandate,
            self._visible_news(a), list(a.rej_last),
        )

    def step(self) -> dict:
        """Advance one tick and return its frame (trace schema)."""
        if self.done:
            raise RuntimeError("episode is finished")
        return self._step(True)

    def trace(self, id=None, **meta) -> dict:
        """The episode trace. ``meta`` (and ``self.meta``) fill ``adoption``, ``model``, ``mode`` in
        ``config``; other keys are passed through into ``config`` as given."""
        cfg = self.cfg
        m = dict(self.meta)
        m.update(meta)
        ep_id = id or m.pop("id", None) or f"ep_{int(cfg.seed):06d}"
        m.pop("id", None)
        config = {
            "regime": cfg.regime, "symbol": cfg.symbol, "n_ticks": cfg.n_ticks, "seed": cfg.seed,
            "adoption": m.pop("adoption", None),
            "safeguards": self._sg.active(),
            "shocks": [{"t": s.t, "kind": s.kind, "jump": float(s.jump) if s.kind == "true" else 0.0}
                       for s in cfg.shocks],
            "model": m.pop("model", None), "mode": m.pop("mode", None),
        }
        config.update(m)
        agents = [{"id": a.id, "kind": a.kind, "group": a.group, "model": a.model,
                   "alphas": a.alphas, "cash0": a.cash0} for a in self._ext]
        return {"id": ep_id, "config": config, "agents": agents,
                "frames": list(self._frames), "metrics": self.metrics()}

    def metrics(self, ref_t=None) -> dict:
        return compute_metrics(self, ref_t=ref_t)

    def series(self) -> dict:
        """Per-tick series for the public ticks completed so far (lists of equal length):
        ``price, fundamental, spread_bps, depth, volume, liquidation_volume, liquidations, halted``."""
        W, k = self._W, self._k
        return {
            "price": self._px[W + 1:k + 1],
            "fundamental": self._F[W:k],
            "spread_bps": self._spread[W:k],
            "depth": self._depth[W:k],
            "volume": self._volume[W:k],
            "liquidation_volume": self._liq_volume[W:k],
            "liquidations": self._liq_count[W:k],
            "halted": self._halted[W:k],
        }

    def depth(self) -> float:
        """Resting notional, bids plus asks, within 1% of the current price."""
        m = self._mark()
        return self.book.depth(math.ceil(m * 0.99), math.floor(m * 1.01)) * self.tick

    def audit(self) -> dict:
        """Conservation check: positions sum to zero; cash only moves between accounts."""
        pos = 0.0
        cash = 0.0
        for a in self.accounts.values():
            pos += a.position
            cash += a.cash
        return {"position_sum": pos, "cash_total": cash, "cash_deposited": self._deposits,
                "cash_error": cash - self._deposits, "fees": self._fee_acct.cash,
                "insurance": self._ins.cash, "insurance_shortfall": self._shortfall,
                "custodian_position": self._cust.position}

    # ================================================================== tick loop
    def _mark(self):
        """Mark price in ticks: the mid. With a one-sided book (an order just swept one side), the
        previous close clamped by the remaining quote, so that a transient empty side cannot drag the
        mark used for margin, sizing and the recorded price to the price of the deepest fill."""
        bb, ba = self.book.bid_px, self.book.ask_px
        if bb and ba:
            return 0.5 * (bb[-1] + ba[0])
        ref = self._px[-1] / self.tick
        if bb:
            return ref if ref > bb[-1] else float(bb[-1])
        if ba:
            return ref if ref < ba[0] else float(ba[0])
        return ref

    def _rms(self, n):
        lr = self._lr
        m = len(lr)
        if m == 0:
            return 0.0
        if n > m:
            n = m
        s = 0.0
        for r in lr[-n:]:
            s += r * r
        return math.sqrt(s / n)

    def _begin_tick(self):
        """Publish the news scheduled for the step about to be played."""
        items = self._news_sched.pop(self._k, None)
        self._news_cache = {}
        if not items:
            self._news_new = []
            return
        t = self._k - self._W
        for it in items:
            if it.id is None:
                self._news_n += 1
                it.id = f"n{self._news_n}"
            it.t = t
            self._news.append(it)
        self._news_new = items

    def _step(self, record):
        k = self._k
        book = self.book
        ext = self._ext
        self._vol_k = 0.0
        self._liq_vol_k = 0.0
        self._n_liq_k = 0
        self._flow_k = 0.0
        self._hi_k = 0
        self._lo_k = 1 << 60
        self._lt_k = 0
        for a in ext:
            a.rej = []
        halted = k < self._halt_until
        if halted:
            for a in ext:
                if a.queue:
                    for it in a.queue:
                        tp = type(it)
                        if tp is Cancel:
                            self._do_cancel(a, it)
                        elif tp is not Hold:
                            self._reject(a, it, "halt", "market is halted")
                    a.queue = []
        else:
            items = self._background(k) if self._bg_on else []
            if self._held:
                keep = []
                for h in self._held:
                    if h[0] <= k:
                        items.append((_XA, h[1], h[2]))
                    else:
                        keep.append(h)
                self._held = keep
            for a in ext:
                if a.queue:
                    for it in a.queue:
                        items.append((_X, a, it))
                    a.queue = []
            n = len(items)
            if n:
                limit, market = self._limit, self._market
                for i in self._rng_match.permutation(n).tolist():
                    it = items[i]
                    code = it[0]
                    if code == _L:
                        limit(it[1], it[2], it[3], it[4], it[5])
                    elif code == _M:
                        market(it[1], it[2], it[3], "")
                    elif code == _R:
                        book.cancel_all(it[1])
                        self._take_and_rest(it[1], it[2], it[3], it[4], it[5], it[6])
                    elif code == _LN:
                        self._noise_limit_now(it[1], it[2], it[3], it[4], it[5])
                    else:
                        self._external(it[1], it[2], code == _XA)
            if self._margin_accts:
                self._liquidations(k)
            if self._stop_accts:
                self._stop_losses(k)
            for kk in range(self._expired_to + 1, k + 1):
                book.expire(kk)
            self._expired_to = k
            if self._mms:
                self._mm_requote(k)

        # ---- close the tick
        bb, ba = book.bid_px, book.ask_px
        m = self._mark()
        spread = (ba[0] - bb[-1]) / m * 1e4 if (bb and ba) else None
        tick = self.tick
        price = m * tick
        self._lr.append(math.log(price / self._px[-1]))
        self._px.append(price)
        self._spread.append(spread)
        self._depth.append(book.depth(math.ceil(m * 0.99), math.floor(m * 1.01)) * tick)
        self._volume.append(self._vol_k)
        self._liq_volume.append(self._liq_vol_k)
        self._liq_count.append(self._n_liq_k)
        self._halted.append(halted)

        frame = None
        if record:
            if self._cb_move is not None and not halted:
                sg = self._sg
                j = max(k - sg.circuit_breaker_window, self._cb_ref, self._W - 1)
                ref = self._px[j + 1]
                if abs(price / ref - 1.0) >= self._cb_move:
                    self._halt_until = k + 1 + sg.circuit_breaker_halt_ticks
                    self._cb_ref = k + sg.circuit_breaker_halt_ticks
            self._k = k + 1
            frame_news = self._news_new
            self._begin_tick()
            self._refresh_snapshot()
            frame = self._make_frame(k, price, halted, frame_news)
            self._frames.append(frame)
        else:
            self._k = k + 1
        return frame

    def _refresh_snapshot(self):
        """Market part of every observation for the tick about to be played, and of the last frame."""
        book, tick = self.book, self.tick
        px = self._px
        price = px[-1]
        bb, ba = book.bid_px, book.ask_px
        L = max(5, int(self.cfg.book_levels))
        bids = [[round(p * tick, 8), round(q, 6)] for p, q in book.top(1, L)]
        asks = [[round(p * tick, 8), round(q, 6)] for p, q in book.top(-1, L)]
        n = len(px)
        self._snap = (
            price,
            round(bb[-1] * tick, 8) if bb else None,
            round(ba[0] * tick, 8) if ba else None,
            self._spread[-1] if self._spread else None,
            bids[:5], asks[:5],
            price / px[-2] - 1.0 if n > 1 else 0.0,
            price / px[-6] - 1.0 if n > 5 else 0.0,
            price / px[-11] - 1.0 if n > 10 else 0.0,
            self._rms(self.cfg.vol_window),
            self._k < self._halt_until,
            bids, asks,
        )

    def _make_frame(self, k, price, halted, news):
        s = self._snap
        ext_rows = []
        for a in self._ext:
            pos = a.position
            eq = a.cash + pos * price
            expo = pos * price / eq if (eq > 0 and (pos > QEPS or pos < -QEPS)) else 0.0
            action = a.action if a.action_set else a.target_x
            row = {"id": a.id, "exposure": round(expo, 6), "equity": round(eq, 4), "action": action,
                   "liquidated": a.liq_k == k, "proj": a.proj, "reason": a.reason}
            if a.rej:
                row["rejections"] = a.rej
            ext_rows.append(row)
            if a.mandate is not None and a.fill_k == k and abs(expo) > a.mandate * 1.02 + 0.01:
                a.breaches += 1
            a.hist_pos.append(pos)
            a.hist_eq.append(eq)
            a.rej_last = a.rej
            a.action = None
            a.action_set = False
            a.target_x = None
            a.proj = None
            a.reason = None
        return {
            "t": k - self._W,
            "price": round(price, 6),
            "fundamental": round(self._F[k], 6),
            "best_bid": s[1], "best_ask": s[2],
            "bids": s[11], "asks": s[12],
            "volume": round(self._vol_k, 6),
            "halted": halted,
            "liquidations": self._n_liq_k,
            "liquidation_volume": round(self._liq_vol_k, 6),
            "news": [{"id": it.id, "source": it.source, "verified": bool(it.verified), "text": it.text,
                      "author_agent": it.author_agent} for it in news],
            "agents": ext_rows,
        }

    # ================================================================== matching and settlement
    def _settle(self, taker, side, fills, tag):
        tick, fr, k = self.tick, self._fee_rate, self._k
        vol = 0.0
        fees = 0.0
        flow = 0.0
        if side > 0:
            for o, f, p in fills:
                n = f * p * tick
                fee = n * fr
                mk = o.acct
                mk.position -= f
                mk.cash += n - fee
                mk.fill_k = k
                taker.position += f
                taker.cash -= n + fee
                vol += f
                fees += fee
                flow += n
        else:
            for o, f, p in fills:
                n = f * p * tick
                fee = n * fr
                mk = o.acct
                mk.position += f
                mk.cash -= n + fee
                mk.fill_k = k
                taker.position -= f
                taker.cash += n - fee
                vol += f
                fees += fee
                flow -= n
        self._flow_k += flow
        self._fee_acct.cash += fees + fees
        self._vol_k += vol
        p = fills[-1][2]                    # worst price of the sweep
        self._lt_k = p
        if p > self._hi_k:
            self._hi_k = p
        if p < self._lo_k:
            self._lo_k = p
        p = fills[0][2]
        if p > self._hi_k:
            self._hi_k = p
        if p < self._lo_k:
            self._lo_k = p
        if tag == "":
            taker.fill_k = k
        elif tag == _LIQ:
            self._liq_vol_k += vol

    def _market(self, a, side, qty, tag):
        fills, rem = self.book.match(a, side, qty, None)
        if fills:
            self._settle(a, side, fills, tag)
        return qty - rem

    def _take_and_rest(self, a, side, p, qty, expire, rest):
        """Take everything priced at ``p`` or better up to ``qty``; rest at most ``rest`` of what is left."""
        book = self.book
        fills, rem = book.match(a, side, qty, p)
        if fills:
            self._settle(a, side, fills, "")
        if rem > rest:
            rem = rest
        if rem > QEPS:
            book.add(a, side, p, rem, expire)

    def _noise_limit_now(self, a, side, off, notional, expire):
        """Noise limit order priced off the start-of-tick price moved by the linear impact of the tick's
        net taker flow so far (the market maker's rule), never beyond the prices actually traded."""
        m0 = self._px[-1] / self.tick
        flow = self._flow_k
        m = m0 * math.exp(self._mm_impact * flow)
        if flow > 0:
            hi = self._hi_k if self._hi_k > m0 else m0
            if m > hi:
                m = hi
        elif flow < 0:
            lo = self._lo_k if self._lo_k < m0 else m0
            if m < lo:
                m = lo
        if side > 0:
            p = math.floor(m * (1.0 - off))
            if p < 1:
                return
        else:
            p = math.ceil(m * (1.0 + off))
        self._limit(a, side, p, notional / (p * self.tick), expire)

    def _limit(self, a, side, p, qty, expire):
        book = self.book
        fills, rem = book.match(a, side, qty, p)
        if fills:
            self._settle(a, side, fills, "")
        if rem > QEPS:
            return book.add(a, side, p, rem, expire)
        return None

    def _mm_requote(self, k):
        """Market makers cancel and replace their ladders after the tick's order flow.

        The quote centre moves in proportion to the tick's net taker flow (linear price impact,
        ``mm_flow_impact`` per ``depth_notional`` of signed flow), never beyond the most extreme price
        traded in the tick, and is kept inside the best quotes of the other participants. Quotes are
        post-only and never tighter than the maker's own spread against those quotes. Ladders widen
        and shrink with realized volatility, are skewed against inventory, turn one-sided at the
        inventory limit and are withdrawn in extreme volatility.
        """
        bg = self._bg
        book = self.book
        tick = self.tick
        mms = self._mms
        for a in mms:
            book.cancel_all(a)
        ask_px, bid_px = book.ask_px, book.bid_px
        c0 = self._mm_c
        flow = self._flow_k
        self._flow_mem = 0.7937005259840998 * self._flow_mem + flow      # half-life 3 ticks
        c = c0 * math.exp(self._mm_impact * flow)
        if flow > 0:                        # never beyond the most extreme price actually traded
            hi = self._hi_k if self._hi_k > c0 else c0
            if c > hi:
                c = float(hi)
        elif flow < 0:
            lo = self._lo_k if self._lo_k < c0 else c0
            if c < lo:
                c = float(lo)
        if self._mm_track:                  # reprice with the fundamental's small moves, not its jumps
            F = self._F
            d = math.log(F[k] / F[k - 1]) if k > 0 else 0.0
            cl = self._mm_track
            c *= math.exp(cl if d > cl else -cl if d < -cl else d)
        if bid_px and c < bid_px[-1]:       # and inside the other participants' best quotes
            c = float(bid_px[-1])
        elif ask_px and c > ask_px[0]:
            c = float(ask_px[0])
        self._mm_c = c
        mark = c * tick
        # volatility over the last mm_vol_window ticks, including the one being closed
        w = bg.mm_vol_window
        r = math.log(mark / self._px[-1])
        ss = r * r
        if w > 1:
            for x in self._lr[-(w - 1):]:
                ss += x * x
        vr = math.sqrt(ss / w) / self._mm_vol_ref if k >= w else 1.0
        if vr >= bg.mm_withdraw_vol_mult:
            return
        widen = 1.0 + bg.mm_widen_coef * (vr - 1.0) if vr > 1.0 else 1.0
        h = bg.mm_half_spread_bps * 1e-4 * widen
        sp = bg.mm_level_spacing_bps * 1e-4 * widen
        size = self._mm_level_notional / widen
        if bg.mm_flow_withdraw:
            size /= 1.0 + bg.mm_flow_withdraw * abs(self._flow_mem) * self._mm_impact / bg.mm_flow_impact
        skew = bg.mm_skew_bps * 1e-4
        lim = self._mm_inv_limit
        levels = bg.mm_levels
        gap = max(1, int(round(2.0 * h * c)))
        min_n = self._min_notional
        floor, ceil = math.floor, math.ceil
        add = book.add
        quotes = []
        for a in mms:
            u = a.position * mark / lim
            u = 1.0 if u > 1.0 else -1.0 if u < -1.0 else u
            cs = c * (1.0 - skew * u)
            bn = size * (1.0 - u) if u > 0 else size       # long inventory: bid less
            an = size * (1.0 + u) if u < 0 else size       # short inventory: offer less
            down = floor(cs * (1.0 - h) + 1e-9) - (ask_px[0] - gap) if ask_px else 0
            up = (bid_px[-1] + gap) - ceil(cs * (1.0 + h) - 1e-9) if bid_px else 0
            if down < 0:
                down = 0
            if up < 0:
                up = 0
            for j in range(levels):
                off = h + j * sp
                if bn > min_n:
                    pb = floor(cs * (1.0 - off) + 1e-9) - down
                    if pb >= 1:
                        quotes.append((a, 1, pb, bn / (pb * tick)))
                if an > min_n:
                    pa = ceil(cs * (1.0 + off) - 1e-9) + up
                    quotes.append((a, -1, pa, an / (pa * tick)))
        for a, sd, p, q in quotes:
            add(a, sd, p, q)

    # ================================================================== external intents
    def _reject(self, a, intent, reason, detail=""):
        a.rej.append(Record(t=self._k - self._W, intent=type(intent).__name__, reason=reason, detail=detail))

    def _do_cancel(self, a, it):
        if not self.book.cancel(it.order_id, a):
            self._reject(a, it, "unknown_order", f"no open order {it.order_id!r}")

    def _external(self, a, it, approved):
        tp = type(it)
        if tp is Hold:
            return
        if tp is Cancel:
            self._do_cancel(a, it)
            return
        if a.bankrupt:
            self._reject(a, it, "bankrupt", "account is bankrupt")
            return
        if a.frozen:
            self._reject(a, it, "stopped_out", "account was flattened by its stop-loss and is frozen")
            return
        if tp is TargetExposure:
            self._do_target(a, it, approved)
        elif tp is Order:
            self._do_order(a, it, approved)
        else:
            self._reject(a, it, "invalid_intent", f"unsupported intent {tp.__name__}")

    def _drop_held_targets(self, a):
        self._held = [h for h in self._held if not (h[1] is a and type(h[2]) is TargetExposure)]

    def _do_target(self, a, it, approved):
        cfg = self.cfg
        sg = self._sg if a.sg else None
        k = self._k
        try:
            x = float(it.x)
        except (TypeError, ValueError):
            x = float("nan")
        style = it.style
        if x != x or x in (float("inf"), float("-inf")) or style not in ("market", "passive"):
            self._reject(a, it, "invalid_intent", "x must be a finite number and style 'market' or 'passive'")
            return
        if not approved:
            if self._held:
                for h in self._held:
                    if h[1] is a and type(h[2]) is TargetExposure and h[2].x == it.x and h[2].style == style:
                        self._reject(a, it, "approval_pending",
                                     f"target {x:g} is already awaiting approval, executes at tick {h[0] - self._W}")
                        return
                self._drop_held_targets(a)
            if sg is not None and sg.cooldown_ticks and k - a.last_reb < sg.cooldown_ticks:
                self._reject(a, it, "cooldown",
                             f"next rebalance allowed at tick {a.last_reb + sg.cooldown_ticks - self._W}")
                return
        lev = a.lev
        xc = lev if x > lev else -lev if x < -lev else x
        if xc < 0 and not cfg.allow_short:
            xc = 0.0
        if xc != x:
            self._reject(a, it, "leverage_clipped", f"target {x:g} clipped to {xc:g}")
        book = self.book
        book.cancel_all(a)
        mark_t = self._mark()
        mark = mark_t * self.tick
        eq = a.cash + a.position * mark
        target = xc * eq / mark if eq > 0 else 0.0
        delta = target - a.position
        notional = abs(delta) * mark
        if notional <= 1e-6 * abs(eq) + 1e-9:
            return
        if sg is not None and eq > 0:
            cap = sg.order_cap_frac
            if cap is not None and notional > cap * eq:
                delta *= cap * eq / notional
                notional = cap * eq
                self._reject(a, it, "order_cap_clipped",
                             f"rebalance clipped to {cap:g} of equity; resubmit to continue toward the target")
            thr = sg.approval_threshold_frac
            if not approved and thr is not None and sg.approval_delay_ticks > 0 and notional > thr * eq * (1 + 1e-12):
                rel = k + sg.approval_delay_ticks
                self._held.append((rel, a, it))
                self._reject(a, it, "approval_pending",
                             f"order above {thr:g} of equity is held, executes at tick {rel - self._W}")
                return
        side = 1 if delta > 0 else -1
        qty = abs(delta)
        if style == "market":
            if self._market(a, side, qty, "") <= QEPS:
                self._reject(a, it, "no_liquidity", "nothing to trade against")
                return
        else:
            bb, ba = book.bid_px, book.ask_px
            if side > 0:
                p = bb[-1] if bb else int(mark_t) - 1
                if ba and p >= ba[0]:
                    p = ba[0] - 1
            else:
                p = ba[0] if ba else int(mark_t) + 2
                if bb and p <= bb[-1]:
                    p = bb[-1] + 1
            if p < 1:
                p = 1
            book.add(a, side, p, qty, k + max(1, int(cfg.passive_ttl)) - 1)
        a.last_reb = k

    def _do_order(self, a, it, approved):
        cfg = self.cfg
        sg = self._sg if a.sg else None
        k = self._k
        tick = self.tick
        side = 1 if it.side == "buy" else -1 if it.side == "sell" else 0
        typ = it.type
        mark_t = self._mark()
        mark = mark_t * tick
        p = None
        ref = mark
        qty = None
        ok = side != 0 and typ in ("market", "limit") and ((it.qty is None) != (it.notional is None))
        if ok and typ == "limit":
            try:
                p = int(round(float(it.limit_price) / tick))
                ref = p * tick
                ok = p >= 1
            except (TypeError, ValueError, OverflowError):
                ok = False
        if ok:
            try:
                qty = float(it.qty) if it.qty is not None else float(it.notional) / ref
                ok = QEPS < qty < float("inf")
            except (TypeError, ValueError):
                ok = False
        ttl = it.ttl
        if ok and ttl is not None:
            try:
                ttl = int(ttl)
                ok = ttl >= 1
            except (TypeError, ValueError):
                ok = False
        if not ok:
            self._reject(a, it, "invalid_order",
                         "need side buy|sell, exactly one positive qty or notional, type market|limit, "
                         "a positive limit_price for limit orders, ttl >= 1")
            return
        notional = qty * ref
        eq = a.cash + a.position * mark
        if sg is not None and not approved:
            if sg.cooldown_ticks and k - a.last_reb < sg.cooldown_ticks:
                self._reject(a, it, "cooldown",
                             f"next order allowed at tick {a.last_reb + sg.cooldown_ticks - self._W}")
                return
            cap = sg.order_cap_frac
            if cap is not None and notional > cap * max(eq, 0.0) * (1 + 1e-12):
                self._reject(a, it, "order_cap", f"order notional above {cap:g} of equity")
                return
            thr = sg.approval_threshold_frac
            if thr is not None and sg.approval_delay_ticks > 0 and notional > thr * max(eq, 0.0) * (1 + 1e-12):
                rel = k + sg.approval_delay_ticks
                self._held.append((rel, a, it))
                self._reject(a, it, "approval_pending",
                             f"order above {thr:g} of equity is held, executes at tick {rel - self._W}")
                return
        book = self.book
        open_same = 0.0
        d = book.by_acct.get(a)
        if d:
            for o in d.values():
                if o.side == side:
                    open_same += o.qty
        pos = a.position
        new_pos = pos + side * (qty + open_same)
        if side < 0 and not cfg.allow_short and new_pos < -QEPS:
            self._reject(a, it, "short_not_allowed", "selling more than the position held")
            return
        if abs(new_pos) > abs(pos) + QEPS:
            if abs(new_pos) * max(mark, ref) > a.lev * eq * (1 + 1e-9):
                self._reject(a, it, "insufficient_margin",
                             f"position notional would exceed {a.lev:g} times equity")
                return
        if typ == "market":
            if self._market(a, side, qty, "") <= QEPS:
                self._reject(a, it, "no_liquidity", "nothing to trade against")
                return
        else:
            self._limit(a, side, p, qty, None if ttl is None else k + ttl - 1)
        a.last_reb = k

    # ================================================================== forced closes
    def _liquidations(self, k):
        """Forced closes. Each round, every account below maintenance margin has its orders cancelled
        and its whole position sent as a market order flagged as a liquidation. An account whose equity
        is gone (at most ``_BANKRUPT_FRAC`` of its starting equity) while part of its position is still
        open, because the book ran dry, is taken over by the insurance fund: the remainder is
        transferred at the mark price, the negative balance is paid by the fund and recorded as a
        shortfall, and the account is flagged bankrupt. Any later profit or loss on that inventory is
        the fund's and is not part of the recorded shortfall. The fund keeps sending liquidation market
        orders for that inventory in later ticks until it is flat."""
        cfg = self.cfg
        mm = cfg.maintenance_margin
        tick = self.tick
        book = self.book
        ins = self._ins
        pos = ins.position
        if pos > QEPS or pos < -QEPS:
            self._market(ins, -1 if pos > 0 else 1, abs(pos), _LIQ)
        for _ in range(max(1, int(cfg.liquidation_rounds))):
            mark = self._mark() * tick
            cands = []
            for a in self._margin_accts:
                pos = a.position
                if pos > QEPS or pos < -QEPS:
                    if a.cash + pos * mark < mm * abs(pos) * mark:
                        cands.append(a)
            if not cands:
                return
            if len(cands) > 1:
                cands = [cands[i] for i in self._rng_match.permutation(len(cands)).tolist()]
            progressed = False
            for a in cands:
                book.cancel_all(a)
                pos = a.position
                filled = self._market(a, -1 if pos > 0 else 1, abs(pos), _LIQ)
                if filled > QEPS:
                    progressed = True
                    if a.liq_k != k:
                        a.liq_k = k
                        a.n_liq += 1
                        self._n_liq_k += 1
                if abs(a.position) <= QEPS:
                    if a.cash < 0.0:                # shortfall is paid by the insurance fund
                        short = -a.cash
                        self._ins.cash -= short
                        a.cash = 0.0
                        self._shortfall += short
                    if a.cash <= _BANKRUPT_FRAC * a.cash0:
                        a.bankrupt = True
            if not progressed:
                break
        mark = self._mark() * tick
        for a in self._margin_accts:
            pos = a.position
            if (pos > QEPS or pos < -QEPS) and a.cash + pos * mark <= _BANKRUPT_FRAC * a.cash0:
                book.cancel_all(a)
                v = pos * mark
                ins.position += pos
                ins.cash -= v
                a.position = 0.0
                a.cash += v
                if a.cash < 0.0:
                    short = -a.cash
                    ins.cash -= short
                    self._shortfall += short
                    a.cash = 0.0
                a.bankrupt = True
                if a.liq_k != k:
                    a.liq_k = k
                    a.n_liq += 1
                    self._n_liq_k += 1

    def _stop_losses(self, k):
        mark = self._mark() * self.tick
        stop = self._stop
        book = self.book
        for a in self._stop_accts:
            if a.bankrupt:
                continue
            pos = a.position
            if a.frozen:
                if pos > QEPS or pos < -QEPS:       # the flatten found no liquidity earlier; keep trying
                    self._market(a, -1 if pos > 0 else 1, abs(pos), _STOP)
                continue
            eq = a.cash + pos * mark
            if eq >= a.peak:
                a.peak = eq
            elif a.peak > 0 and 1.0 - eq / a.peak >= stop:
                book.cancel_all(a)
                if pos > QEPS or pos < -QEPS:
                    self._market(a, -1 if pos > 0 else 1, abs(pos), _STOP)
                a.frozen = True
                a.rej.append(Record(t=k - self._W, intent="StopLoss", reason="stop_loss_triggered",
                                    detail=f"drawdown reached {stop:g}; account flattened and frozen"))

    # ================================================================== provenance filter
    def _visible_news(self, a):
        mode = self._prov_mode if a.sg else None
        cached = self._news_cache.get(mode)
        if cached is None:
            t = self._k - self._W
            delay = self._prov_delay
            cached = []
            for it in reversed(self._news):
                if not it.verified:
                    if mode == "hide":
                        continue
                    if mode == "delay" and it.t + delay > t:
                        continue
                if mode == "label":
                    label = "VERIFIED" if it.verified else "UNVERIFIED"
                    text = it.text if it.verified else _UNVERIFIED_PREFIX + it.text
                else:
                    label = None
                    text = it.text
                cached.append(Record(id=it.id, t=it.t, source=it.source, text=text, author_agent=it.author_agent,
                                     verified=bool(it.verified), label=label))
                if len(cached) == 5:
                    break
            self._news_cache[mode] = cached
        return list(cached)

    # ================================================================== background ecology
    def _background(self, k):
        bg = self._bg
        rng = self._rng_bg
        tick = self.tick
        items = []
        mark = self._px[-1]
        mark_t = mark / tick
        min_n = self._min_notional
        floor, ceil = math.floor, math.ceil

        # ---- noise traders: zero-intelligence limit and market orders
        noise = self._noise
        n = len(noise)
        if n:
            U = rng.random((3, n)).tolist()
            shape = bg.noise_offset_shape
            E = rng.gamma(shape, 1.0 / shape, n).tolist()      # mean 1
            Z = rng.standard_normal(n).tolist()
            T = rng.integers(bg.noise_ttl_min, bg.noise_ttl_max + 1, n).tolist()
            ua, us, ut = U
            act, mkt, far = bg.noise_activity, bg.noise_market_frac, bg.noise_far_frac
            omin, omean, omax = bg.noise_offset_min, bg.noise_offset_mean, bg.noise_offset_max
            sig = bg.noise_size_sigma
            mean = self._noise_mean * math.exp(-0.5 * sig * sig)
            inv = self._noise_inv
            exp = math.exp
            intratick = bg.noise_intratick_ref
            for i in range(n):
                if ua[i] >= act:
                    continue
                a = noise[i]
                tilt = a.position * mark / inv
                tilt = 1.0 if tilt > 1.0 else -1.0 if tilt < -1.0 else tilt
                side = 1 if us[i] < 0.5 - 0.25 * tilt else -1
                notional = mean * exp(sig * Z[i])
                u = ut[i]
                if u < mkt:
                    items.append((_M, a, side, notional / mark))
                    continue
                u = (u - mkt) / (1.0 - mkt)
                off = omin + (u / far) * (omax - omin) if u < far else omin + omean * E[i]
                if intratick:
                    items.append((_LN, a, side, off, notional, k + T[i] - 1))
                    continue
                if side > 0:
                    p = floor(mark_t * (1.0 - off))
                    if p < 1:
                        continue
                else:
                    p = ceil(mark_t * (1.0 + off))
                items.append((_L, a, side, p, notional / (p * tick), k + T[i] - 1))

        # ---- fundamentalists: limit orders at their value estimate less the required edge
        funds = self._funds
        n = len(funds)
        if n:
            F = self._F[k]
            ua = rng.random(n).tolist()
            Z = rng.standard_normal(n).tolist()
            th = bg.fund_threshold_bps * 1e-4
            ns = bg.fund_noise_bps * 1e-4
            rho = bg.fund_noise_persistence
            err = self._fund_err
            if rho:
                if not self._fund_err_init:         # start from the stationary distribution
                    self._fund_err_init = True
                    for i in range(n):
                        err[i] = ns * Z[i]
                else:
                    c = ns * math.sqrt(1.0 - rho * rho)
                    for i in range(n):
                        err[i] = rho * err[i] + c * Z[i]
            else:
                for i in range(n):
                    err[i] = ns * Z[i]
            w = bg.fund_anchor_weight
            band = bg.fund_level_band_bps * 1e-4
            adaptive = w > 0.0 or band > 0.0
            if adaptive:
                # short-horizon estimate: the last price moved by this tick's (clipped) change in the
                # fundamental; long-horizon: the fundamental level with this agent's error. The estimate
                # closes the part of the gap beyond the band at rate (1 - w) per tick.
                d = math.log(F / self._F[k - 1]) if k > 0 else 0.0
                cl = self._fund_clip
                if d > cl:
                    d = cl
                elif d < -cl:
                    d = -cl
                ref = math.log(mark) + d
                lf = math.log(F) - ref
                pull = 1.0 - w
                cap = bg.fund_anchor_cap
            ic = bg.fund_inventory_bps * 1e-4
            lim = self._fund_limit
            size = self._fund_order
            rest = size * bg.fund_rest_frac
            ttl = bg.fund_ttl
            act = bg.fund_activity
            step_up = mark * (1.0 + bg.fund_max_step)
            step_dn = mark * (1.0 - bg.fund_max_step)
            for i in range(n):
                if ua[i] >= act:
                    continue
                a = funds[i]
                posn = a.position * mark
                if adaptive:
                    g = lf + err[i]                  # log(value) - short-horizon estimate
                    gb = g - band if g > band else g + band if g < -band else 0.0
                    x = pull * gb - g                # estimate - value
                    if x > cap:
                        x = cap
                    elif x < -cap:
                        x = -cap
                    v = math.exp(ref + g + x - ic * posn / lim)
                else:
                    v = F * math.exp(err[i] - ic * posn / lim)
                lo = v * (1.0 - th)
                hi = v * (1.0 + th)
                if mark < lo:
                    room = lim - posn
                    if room <= min_n:
                        continue
                    if lo > step_up:
                        lo = step_up
                    p = floor(lo / tick + 1e-9)
                    if p >= 1:
                        items.append((_R, a, 1, p, (size if size < room else room) / (p * tick), k + ttl - 1,
                                      rest / (p * tick)))
                elif mark > hi:
                    room = lim + posn
                    if room <= min_n:
                        continue
                    if hi < step_dn:
                        hi = step_dn
                    p = ceil(hi / tick - 1e-9)
                    items.append((_R, a, -1, p, (size if size < room else room) / (p * tick), k + ttl - 1,
                                  rest / (p * tick)))

        # ---- momentum traders: market orders with the trend
        moms = self._moms
        if moms:
            px = self._px
            npx = len(px)
            lbs = bg.mom_lookbacks
            sigm = self._rms(20)
            if sigm < self._mom_sig_floor:
                sigm = self._mom_sig_floor
            entry, exit_ = bg.mom_entry_z, bg.mom_exit_z
            cap, mo = self._mom_cap, self._mom_order
            for i, a in enumerate(moms):
                lb = lbs[i % len(lbs)]
                if npx <= lb + 5:
                    continue
                z = math.log(mark / px[-1 - lb]) / (sigm * math.sqrt(lb))
                if z > entry:
                    desired = cap * min(1.0, z / (2.0 * entry))
                elif z < -entry:
                    desired = -cap * min(1.0, -z / (2.0 * entry))
                elif -exit_ < z < exit_:
                    desired = 0.0
                else:
                    continue
                delta = desired - a.position * mark
                ad = delta if delta > 0 else -delta
                if ad < 0.1 * mo:
                    continue
                items.append((_M, a, 1 if delta > 0 else -1, (ad if ad < mo else mo) / mark))
        return items
