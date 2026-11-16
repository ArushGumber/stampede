"""Stampede exchange: a tick-based limit order book with leverage, liquidation, safeguards and a
background ecology. Pure Python and numpy; deterministic for a given seed.

    from stampede.exchange import Exchange, regime, TargetExposure
    ex = Exchange(regime("equity", seed=0, n_ticks=60))
    ex.add_account("a01", cash=100_000.0, group="reckless", mandate_max_exposure=1.0)
    while not ex.done:
        obs = ex.observe("a01")
        ex.submit("a01", TargetExposure(x=1.5, style="market"))
        frame = ex.step()
    trace, metrics = ex.trace(), ex.metrics()
"""
from .book import BookOrder, OrderBook
from .config import (BackgroundConfig, FundamentalConfig, MarketConfig, NewsItem, SafeguardConfig,
                     Shock, regime)
from .engine import Account, Exchange, Record
from .intents import Cancel, Hold, Observation, Order, TargetExposure
from .metrics import METRIC_KEYS, compute_metrics

__all__ = [
    "Exchange", "MarketConfig", "BackgroundConfig", "SafeguardConfig", "FundamentalConfig",
    "Shock", "NewsItem", "TargetExposure", "Order", "Cancel", "Hold", "Observation", "regime",
    "OrderBook", "BookOrder", "Account", "Record", "compute_metrics", "METRIC_KEYS",
]
