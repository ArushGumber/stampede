"""The four intent types an account can submit, and the per-agent observation."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional

__all__ = ["TargetExposure", "Order", "Cancel", "Hold", "Observation"]


@dataclass
class TargetExposure:
    """Rebalance so that position notional equals ``x`` times account equity.

    ``x`` may be negative (short) and is clipped to the allowed leverage. ``style="market"`` crosses
    the spread at once; ``style="passive"`` rests a limit order at the touch. Either way the account's
    other open orders are cancelled first.
    """
    x: float
    style: str = "market"


@dataclass
class Order:
    """A raw order. Exactly one of ``qty`` (asset units) and ``notional`` (currency) must be given.

    ``ttl`` is the number of ticks a limit order stays in the book (None = until cancelled).
    """
    side: str
    qty: Optional[float] = None
    notional: Optional[float] = None
    type: str = "market"
    limit_price: Optional[float] = None
    ttl: Optional[int] = None


@dataclass
class Cancel:
    order_id: int


@dataclass
class Hold:
    pass


@dataclass(slots=True)
class Observation:
    """What one account sees at the start of a tick (before it submits).

    Market fields describe the book at the end of the previous tick. ``ret_k`` are simple returns of
    the price over the last ``k`` ticks; ``realized_vol`` is the root mean square of one-tick log
    returns over ``MarketConfig.vol_window`` ticks (per-tick units). ``exposure`` is position notional
    over equity; ``pnl_pct`` is ``equity / starting equity - 1`` as a fraction (0.01 is one percent,
    like ``ret_k``); ``margin_ratio`` is equity
    over position notional (None when flat; liquidation happens below ``maintenance_margin``);
    ``liq_price`` is the price at which that happens (None when no such price exists).
    ``max_leverage`` is the cap in force for this account (regime maximum or the broker cap).
    ``news`` are dicts ``{id, t, source, text, author_agent, verified, label}`` (keys also readable as
    attributes), newest first, at most five, after the provenance filter. ``t`` is the tick the item
    appeared. ``label`` is None unless the account's filter is ``"label"``; then it is "VERIFIED" or
    "UNVERIFIED" and the text of an unverified item starts with an explicit warning.
    ``open_orders`` are dicts ``{id, side, price, qty, ttl}``. ``rejections`` are dicts
    ``{t, intent, reason, detail}`` from the previous tick.
    """
    t: int
    n_ticks: int
    symbol: str
    asset_name: str
    price: float
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread_bps: Optional[float]
    bids: list
    asks: list
    ret_1: float
    ret_5: float
    ret_10: float
    realized_vol: float
    halted: bool
    cash: float
    position: float
    equity: float
    exposure: float
    pnl_pct: float
    liq_price: Optional[float]
    margin_ratio: Optional[float]
    open_orders: list
    max_leverage: float
    mandate_max_exposure: Optional[float]
    news: list = field(default_factory=list)
    rejections: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}
