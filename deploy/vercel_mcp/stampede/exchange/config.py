"""Configuration objects for the Stampede exchange.

Everything here is plain data. Nothing in this module holds state, so any number of
``Exchange`` instances can be built from (copies of) these objects in one process.

Units used throughout the package
---------------------------------
* one tick of the clock is thought of as roughly one minute;
* prices are in currency units, quantities in asset units (shares or coins);
* ``volume`` and ``liquidation_volume`` are in asset units;
* ``depth_notional``, ``min_depth_post`` and ``insurance_shortfall`` are in currency units;
* all returns and jumps are fractions (0.01 is one percent); shock jumps and replayed
  fundamental returns are log returns.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from typing import Optional

__all__ = [
    "NewsItem", "Shock", "SafeguardConfig", "FundamentalConfig", "BackgroundConfig",
    "MarketConfig", "regime", "REGIMES", "RESILIENCE", "effective_background",
]


@dataclass
class NewsItem:
    """A headline. ``id`` and ``t`` are filled in by the exchange when the item is published."""
    source: str
    verified: bool
    text: str
    id: Optional[str] = None
    t: Optional[int] = None
    author_agent: Optional[str] = None


@dataclass
class Shock:
    """A scheduled event at tick ``t``.

    kind
        ``"true"``: the fundamental moves by the log return ``jump`` at ``t`` and ``news`` is published.
        ``"fake"``: ``news`` is published, the fundamental does not move (``jump`` is ignored).
        ``"correction"``: news that an earlier fake item was false; the fundamental does not move.
        If ``news`` is None for a correction, a verified exchange notice quoting the most recent
        fake item is generated.
    """
    t: int
    kind: str = "true"
    jump: float = 0.0
    news: Optional[NewsItem] = None

    def __post_init__(self):
        if self.kind not in ("true", "fake", "correction"):
            raise ValueError(f"Shock.kind must be 'true', 'fake' or 'correction', got {self.kind!r}")


@dataclass
class SafeguardConfig:
    """Broker-side and market-wide safeguards. Everything is off by default.

    A safeguard is active when its primary field is not None (``cooldown_ticks`` also when > 0).

    order_cap_frac
        Maximum notional of one order as a fraction of account equity. A raw ``Order`` above the cap
        is rejected (``order_cap``); a ``TargetExposure`` rebalance is clipped to the cap and proceeds
        (notice ``order_cap_clipped``), so an account moves toward its target at a bounded rate.
    leverage_cap
        Broker-side leverage cap below the regime maximum.
    approval_threshold_frac, approval_delay_ticks
        Orders whose notional exceeds the threshold (fraction of equity) are held and executed
        ``approval_delay_ticks`` ticks later (notice ``approval_pending``). A held TargetExposure is
        superseded by any later TargetExposure with a different target.
    cooldown_ticks
        Minimum number of ticks between two rebalances (accepted orders) of one account.
    account_stop_loss
        Drawdown from the account's equity high-water mark at which the account is flattened by a
        market order and frozen for the rest of the episode.
    circuit_breaker_move, circuit_breaker_window, circuit_breaker_halt_ticks
        Market-wide halt: if the price moved by at least ``move`` (absolute simple return) against the
        price ``window`` ticks earlier, trading halts for ``halt_ticks`` ticks. Applies to everyone.
    provenance_filter
        Applied inside ``observe()``: ``None`` (items as published), ``"label"`` (each item carries
        ``label`` "VERIFIED"/"UNVERIFIED" and unverified text is prefixed with an explicit warning),
        ``"hide_unverified"`` (unverified items are never shown), or ``"delay_unverified:<ticks>"``
        (unverified items appear that many ticks late). Frames in the trace always hold the raw items.
    applies_to
        Groups the per-account safeguards apply to. ``None`` means every non-background account.
    """
    order_cap_frac: Optional[float] = None
    leverage_cap: Optional[float] = None
    approval_threshold_frac: Optional[float] = None
    approval_delay_ticks: int = 3
    cooldown_ticks: Optional[int] = None
    account_stop_loss: Optional[float] = None
    circuit_breaker_move: Optional[float] = None
    circuit_breaker_window: int = 5
    circuit_breaker_halt_ticks: int = 5
    provenance_filter: Optional[str] = None
    applies_to: Optional[list] = None

    def __post_init__(self):
        self.provenance_delay()  # validates the filter string

    def provenance_delay(self) -> int:
        """Delay in ticks for the ``delay_unverified:<ticks>`` filter, else 0."""
        pf = self.provenance_filter
        if pf is None or pf in ("label", "hide_unverified"):
            return 0
        if isinstance(pf, str) and pf.startswith("delay_unverified:"):
            try:
                d = int(pf.split(":", 1)[1])
            except ValueError:
                d = -1
            if d >= 0:
                return d
        raise ValueError(
            "provenance_filter must be None, 'label', 'hide_unverified' or 'delay_unverified:<ticks>', "
            f"got {pf!r}")

    def active(self) -> dict:
        """The active safeguards as a plain dict (empty when everything is off)."""
        out = {}
        if self.order_cap_frac is not None:
            out["order_cap_frac"] = self.order_cap_frac
        if self.leverage_cap is not None:
            out["leverage_cap"] = self.leverage_cap
        if self.approval_threshold_frac is not None:
            out["approval_threshold_frac"] = self.approval_threshold_frac
            out["approval_delay_ticks"] = self.approval_delay_ticks
        if self.cooldown_ticks:
            out["cooldown_ticks"] = self.cooldown_ticks
        if self.account_stop_loss is not None:
            out["account_stop_loss"] = self.account_stop_loss
        if self.circuit_breaker_move is not None:
            out["circuit_breaker_move"] = self.circuit_breaker_move
            out["circuit_breaker_window"] = self.circuit_breaker_window
            out["circuit_breaker_halt_ticks"] = self.circuit_breaker_halt_ticks
        if self.provenance_filter is not None:
            out["provenance_filter"] = self.provenance_filter
        if out and self.applies_to is not None:
            out["applies_to"] = list(self.applies_to)
        return out


@dataclass
class FundamentalConfig:
    """The fundamental value process.

    Random walk in logs: ``log F[t+1] = log F[t] + drift + sigma[t] * z``. With ``vol_of_vol > 0`` the
    per-tick volatility follows a stationary log-AR(1) (news arrives in clusters); the unconditional
    per-tick variance stays ``sigma**2``. If ``replay_returns`` is given, those log returns are used
    instead from tick 0 on (``F[t] = F[t-1] * exp(replay_returns[t-1])``, zero once the list runs out);
    the random walk is then used only for the warm-up before tick 0.
    """
    sigma: float = 0.0010
    drift: float = 0.0
    vol_of_vol: float = 0.0
    vol_persistence: float = 0.95
    replay_returns: Optional[list] = None


@dataclass
class BackgroundConfig:
    """The background ecology. Background agents never read news; only the fundamental moves them.

    ``depth_notional`` is the anchor quantity. It is the time-average resting notional, bids plus asks,
    at prices within 1% of the mid, in a background-only market (currency units). Every order size and
    inventory limit below is a fraction of it, so changing ``depth_notional`` rescales the whole
    ecology and leaves prices, spreads and volatility unchanged. Experiments size agent capital as a
    multiple of it ("adoption share"). ``scripts/validate_exchange.py`` reports measured against nominal.

    ``resilience`` ("high" | "medium" | "low") sets how fast the background closes a dislocation that
    news-reading agents cause; see ``RESILIENCE`` below. ``regime()`` uses "medium". The fields below
    describe "high"; "medium" and "low" override some of them when the Exchange is built.
    """
    enabled: bool = True
    depth_notional: float = 2_000_000.0
    resilience: str = "high"             # "high" | "medium" | "low"; see RESILIENCE below. regime() uses "medium"

    # --- zero-intelligence noise traders: random limit orders around the mid, some market orders
    n_noise: int = 30
    noise_activity: float = 0.6          # probability that a noise trader sends an order in a tick
    noise_market_frac: float = 0.08      # share of those orders that are market orders
    noise_order_frac: float = 0.0077     # mean order notional / depth_notional
    noise_size_sigma: float = 0.7        # lognormal sigma of order size
    noise_offset_min: float = 0.0002     # minimum distance of a limit price from the mid (fraction)
    noise_offset_mean: float = 0.0100    # mean of the gamma-distributed near component
    noise_offset_shape: float = 3.0      # gamma shape; above 1 gives a book that thickens away from the touch
    noise_far_frac: float = 0.20         # share of limit orders from the uniform far component
    noise_offset_max: float = 0.06       # upper end of the far component
    noise_ttl_min: int = 5
    noise_ttl_max: int = 30
    noise_inventory_frac: float = 0.25   # inventory (fraction of depth) at which side choice is fully tilted
    noise_intratick_ref: bool = False    # True: limit prices are set when the order is processed, from the
                                         # start-of-tick price moved by the linear impact (mm_flow_impact) of
                                         # the tick's net taker flow so far, so noise traders see an earlier sweep

    # --- market maker: ladder of quotes around the mid, skewed against inventory
    n_market_makers: int = 1
    mm_levels: int = 5
    mm_half_spread_bps: float = 1.5
    mm_level_spacing_bps: float = 2.0
    mm_level_frac: float = 0.010         # notional per level per side / depth_notional
    mm_inventory_limit_frac: float = 0.5 # inventory limit / depth_notional; one-sided quoting beyond it
    mm_skew_bps: float = 6.0             # shift of the quote centre at the inventory limit
    mm_flow_impact: float = 0.020        # quote-centre move per depth_notional of net taker flow in a tick
    mm_vol_ref: Optional[float] = None   # normal per-tick volatility; None = 1.2 * fundamental sigma
    mm_vol_window: int = 5
    mm_widen_coef: float = 0.5          # spread multiplier = 1 + coef * max(0, vol / vol_ref - 1)
    mm_withdraw_vol_mult: float = 6.0    # no quotes while vol / vol_ref is above this
    mm_flow_withdraw: float = 0.0        # quote size multiplier 1 / (1 + this * one-sided flow / depth_notional),
                                         # flow being an exponentially decaying sum (half-life 3 ticks)
    mm_track_fundamental: float = 0.0    # > 0: the quote centre also moves with each tick's change in the log
                                         # fundamental, clipped at this many sigmas (small moves are public
                                         # information, jumps are not); 0: off

    # --- fundamentalists: buy below and sell above their estimate of the fundamental
    n_fundamentalists: int = 8
    fund_activity: float = 0.75          # probability of acting in a tick
    fund_order_frac: float = 0.040       # order notional / depth_notional (their per-tick capacity)
    fund_threshold_bps: float = 1.5      # required edge before trading
    fund_noise_bps: float = 1.0          # idiosyncratic valuation error (standard deviation)
    fund_noise_persistence: float = 0.0  # AR(1) coefficient of each agent's error per tick (0: fresh every tick)
    fund_anchor_weight: float = 0.0      # w: weight of the short-horizon estimate (last price moved by this
                                         # tick's change in the fundamental, clipped at fund_track_clip sigmas);
                                         # 1 - w is the share of the gap to the fundamental level closed per tick
    fund_level_band_bps: float = 0.0     # gaps to the fundamental level smaller than this are ignored
    fund_anchor_cap: float = 1.0         # the estimate is never further than this (log) from the agent's
                                         # value, so prices far from value meet full resistance
    fund_track_clip: float = 6.0         # sigmas; larger per-tick changes (jumps) are learned only via the level
    fund_position_limit_frac: float = 0.5  # per-agent position limit / depth_notional
    fund_inventory_bps: float = 5.0      # reservation-price shift at the position limit
    fund_rest_frac: float = 0.2          # share of an order that may rest at the limit price if unfilled
    fund_ttl: int = 2                    # ticks that resting remainder stays in the book
    fund_max_step: float = 0.006         # limit prices are at most this far from the last price, which
                                         # bounds how fast they can walk the price toward the fundamental

    # --- momentum traders: trade with the recent trend using market orders
    n_momentum: int = 6
    mom_lookbacks: tuple = (3, 5, 8, 12, 20, 30)
    mom_entry_z: float = 1.5
    mom_exit_z: float = 0.5
    mom_position_frac: float = 0.04      # maximum position notional / depth_notional
    mom_order_frac: float = 0.010        # maximum order notional per tick / depth_notional

    margin_exempt: bool = True           # background accounts are never liquidated


# Background resilience presets. "high" leaves every BackgroundConfig field as given (the original
# calibration: fundamentalists know the fundamental to about a basis point and close 75% of any gap
# within one or two ticks). "medium" and "low" replace the listed fields when the Exchange is built;
# the config object itself is not modified. Units, so that one preset fits both regimes:
#   keys ending in "_bps" are multiples of the fundamental's per-tick sigma (equity 9.2 bps, crypto 22 bps);
#   fund_order_frac and fund_position_limit_frac are multiples of noise_order_frac (the mean noise order);
#   ("x", m) multiplies the field's own (regime) value by m; everything else is used as given.
# What changes: fundamentalists' value estimates carry persistent idiosyncratic errors of tens of basis
# points; each estimate is mostly the last price moved by the fundamental's small per-tick change
# (weight fund_anchor_weight), and only partly the fundamental level, so a dislocation (or a jump, which
# the short-horizon part does not see) closes by a fraction 1 - fund_anchor_weight per tick; noise
# traders stop accumulating large inventories; noise limit orders are priced off the current mark, so a
# sweep is not absorbed by orders priced as if it had not happened (they quote around the start-of-tick
# price moved by the linear impact of the tick's net flow so far); the market maker reprices with the
# fundamental's small moves and quotes less after strong one-sided flow.
RESILIENCE = {
    "high": {},
    "medium": {
        "fund_anchor_weight": 0.85, "fund_anchor_cap": 0.02, "fund_level_band_bps": 0.5, "fund_noise_bps": 3.0,
        "fund_noise_persistence": 0.97, "fund_threshold_bps": 0.45, "fund_inventory_bps": 0.5,
        "fund_activity": 0.6, "fund_order_frac": 4.0, "fund_position_limit_frac": 33.0,
        "noise_inventory_frac": 0.05, "noise_intratick_ref": True,
        "mm_flow_withdraw": 1.0, "mm_track_fundamental": 6.0, "mm_withdraw_vol_mult": 30.0,
    },
    "low": {
        "fund_anchor_weight": 0.90, "fund_anchor_cap": 0.03, "fund_level_band_bps": 0.75, "fund_noise_bps": 4.0,
        "fund_noise_persistence": 0.98, "fund_threshold_bps": 0.45, "fund_inventory_bps": 0.5,
        "fund_activity": 0.5, "fund_order_frac": 4.0, "fund_position_limit_frac": 33.0,
        "noise_inventory_frac": 0.05, "noise_intratick_ref": True,
        "mm_flow_withdraw": 2.0, "mm_track_fundamental": 6.0, "mm_withdraw_vol_mult": 30.0,
        "mm_flow_impact": ("x", 1.5),
    },
}
_NOISE_UNITS = ("fund_order_frac", "fund_position_limit_frac")


def effective_background(bg: "BackgroundConfig", sigma: float) -> "BackgroundConfig":
    """A copy of ``bg`` with its resilience preset applied (``sigma``: fundamental per-tick sigma)."""
    try:
        preset = RESILIENCE[bg.resilience]
    except KeyError:
        raise ValueError(f"BackgroundConfig.resilience must be one of {sorted(RESILIENCE)}, "
                         f"got {bg.resilience!r}") from None
    out = copy.copy(bg)
    for key, value in preset.items():
        if isinstance(value, tuple):          # ("x", m): multiply the field's own value
            value = getattr(bg, key) * value[1]
        elif key.endswith("_bps"):
            value = value * sigma * 1e4
        elif key in _NOISE_UNITS:
            value = value * bg.noise_order_frac
        setattr(out, key, value)
    return out


@dataclass
class MarketConfig:
    regime: str = "equity"
    symbol: str = "HLVN"
    asset_name: str = "Halvorsen Industries"
    initial_price: float = 100.0
    tick_size: float = 0.01
    max_leverage: float = 2.0
    maintenance_margin: float = 0.25
    allow_short: bool = True
    fee_bps: float = 1.0                 # charged on the notional of every fill, to both sides
    n_ticks: int = 60
    seed: int = 0
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    fundamental: FundamentalConfig = field(default_factory=FundamentalConfig)
    shocks: list = field(default_factory=list)
    safeguards: SafeguardConfig = field(default_factory=SafeguardConfig)
    # --- engine details beyond the contract (all have working defaults)
    warmup_ticks: int = 30               # background-only ticks run before tick 0 to fill the book
    passive_ttl: int = 5                 # ticks a passive TargetExposure order rests before expiring
    vol_window: int = 20                 # window of Observation.realized_vol
    book_levels: int = 10                # levels per side recorded in each frame
    liquidation_rounds: int = 3          # liquidation passes inside one tick (cascade depth per tick)
    insurance_fund_cash: float = 0.0     # starting balance of the insurance fund (may go negative)

    def copy(self) -> "MarketConfig":
        return copy.deepcopy(self)


def _equity(seed: int, n_ticks: int) -> MarketConfig:
    return MarketConfig(
        regime="equity", symbol="HLVN", asset_name="Halvorsen Industries",
        initial_price=100.0, tick_size=0.01,
        max_leverage=2.0, maintenance_margin=0.25, allow_short=True, fee_bps=1.0,
        n_ticks=n_ticks, seed=seed,
        fundamental=FundamentalConfig(sigma=0.00092, vol_of_vol=0.45, vol_persistence=0.95),
        background=BackgroundConfig(depth_notional=2_000_000.0, resilience="medium"),
    )


def _crypto(seed: int, n_ticks: int) -> MarketConfig:
    return MarketConfig(
        regime="crypto", symbol="MRD", asset_name="Meridian",
        initial_price=2000.0, tick_size=0.1,
        max_leverage=10.0, maintenance_margin=0.05, allow_short=True, fee_bps=2.5,
        n_ticks=n_ticks, seed=seed,
        fundamental=FundamentalConfig(sigma=0.0022, vol_of_vol=0.45, vol_persistence=0.95),
        background=BackgroundConfig(
            depth_notional=1_000_000.0, resilience="medium",
            noise_order_frac=0.0118, noise_offset_min=0.0004, noise_offset_mean=0.0120, noise_offset_max=0.10,
            mm_half_spread_bps=3.5, mm_level_spacing_bps=3.5, mm_level_frac=0.012, mm_skew_bps=12.0,
            mm_flow_impact=0.030,
            fund_threshold_bps=3.0, fund_noise_bps=2.0, fund_inventory_bps=10.0, fund_max_step=0.008,
            fund_order_frac=0.05,
        ),
    )


REGIMES = {"equity": _equity, "crypto": _crypto}


def regime(name: str, seed: int = 0, n_ticks: int = 60, **overrides) -> MarketConfig:
    """Calibrated ``MarketConfig`` for ``"equity"`` (Halvorsen Industries, HLVN) or ``"crypto"``
    (Meridian, MRD). Extra keyword arguments override ``MarketConfig`` fields."""
    try:
        cfg = REGIMES[name](seed, n_ticks)
    except KeyError:
        raise ValueError(f"unknown regime {name!r}; choose from {sorted(REGIMES)}") from None
    names = {f.name for f in fields(MarketConfig)}
    for key, value in overrides.items():
        if key not in names:
            raise TypeError(f"regime() got an unexpected MarketConfig field {key!r}")
        setattr(cfg, key, value)
    return cfg
