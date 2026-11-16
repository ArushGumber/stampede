"""Per-episode metrics (contract section 4).

Definitions, with ``ts`` the tick of the first non-correction shock (0 if there is none) and prices
taken at the end of each tick:

p0                   mean price over the 5 ticks before ``ts`` (warm-up prices fill in when ts < 5)
trough               lowest price at or after ``ts``
crash_depth          (p0 - trough) / p0 (negative if the price never fell below p0)
overshoot            max(0, fundamental at the trough tick - trough) / p0
dislocation_area     sum over all ticks of |price - fundamental| / fundamental
recovery_ticks       ticks after ``ts`` until |price - fundamental| / fundamental < 0.01 and stays there
                     for 3 ticks; 0 if it never left; None if it never returned
vol_pre, vol_post    standard deviation of one-tick log returns over the (at most) 30 ticks before
                     ``ts`` and over the ticks from ``ts`` on
mean_spread_bps_*    mean quoted spread over the same two windows (one-sided books are skipped)
min_depth_post       lowest resting notional within 1% of the price from ``ts`` on (currency)
n_liquidations       number of (account, tick) liquidation events
liquidation_volume   asset units sold or bought back by liquidations
fire_sale_share      liquidation_volume / volume_total
insurance_shortfall  currency paid by the insurance fund
action_correlation   mean pairwise Pearson correlation, across llm accounts, of the per-tick change in
                     exposure caused by trading, (position[t] - position[t-1]) * price[t] / equity[t-1],
                     from ``ts`` on. Passive drift of exposure with the price is excluded on purpose,
                     since it is common to all holders. Accounts that never trade in the window are
                     left out; None with fewer than two accounts.
pnl_by_group         mean of final equity / starting equity - 1 per group; background accounts appear
                     as the group "background"
bankruptcies_by_group      accounts flagged bankrupt
mandate_breaches_by_group  (account, tick) events where the account traded and ended the tick with
                           |exposure| above its mandate (2% tolerance)
volume_total         asset units traded
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["compute_metrics", "METRIC_KEYS"]

METRIC_KEYS = (
    "p0", "trough", "crash_depth", "overshoot", "overshoot_up", "dislocation_area", "recovery_ticks",
    "vol_pre", "vol_post", "mean_spread_bps_pre", "mean_spread_bps_post", "min_depth_post",
    "n_liquidations", "liquidation_volume", "fire_sale_share", "insurance_shortfall",
    "action_correlation", "pnl_by_group", "bankruptcies_by_group", "mandate_breaches_by_group",
    "volume_total",
)

_PRE_WINDOW = 30


def _std(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else None


def _mean_skip_none(x):
    v = [s for s in x if s is not None]
    return float(sum(v) / len(v)) if v else None


def _action_correlation(ex, ts, n_done):
    W = ex._W
    px = ex._px
    rows = []
    for a in ex._ext:
        if a.kind != "llm":
            continue
        hp, he = a.hist_pos, a.hist_eq
        t0 = n_done - len(hp)               # first tick this account was recorded on
        d = []
        for t in range(max(ts, t0), n_done):
            i = t - t0
            prev_pos = hp[i - 1] if i > 0 else a.position0
            prev_eq = he[i - 1] if i > 0 else a.cash0
            d.append((hp[i] - prev_pos) * px[W + t + 1] / prev_eq if prev_eq > 0 else 0.0)
        if t0 > ts:
            d = [0.0] * (t0 - ts) + d
        rows.append(d)
    if len(rows) < 2 or len(rows[0]) < 3:
        return None
    m = np.asarray(rows, dtype=float)
    m = m[m.std(axis=1) > 1e-12]
    if len(m) < 2:
        return None
    c = np.corrcoef(m)
    iu = np.triu_indices(len(m), 1)
    v = c[iu]
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else None


def compute_metrics(ex, ref_t=None) -> dict:
    """``ref_t`` overrides the reference tick ``ts``; pass the headline tick when scoring a no-headline twin so
    that both episodes are measured over the same window."""
    cfg = ex.cfg
    W = ex._W
    n_done = ex._k - W
    px = ex._px                     # px[W + t + 1] is the price at the end of tick t
    F = ex._F
    shock_ts = [s.t for s in cfg.shocks if s.kind != "correction" and 0 <= s.t < cfg.n_ticks]
    ts = min(shock_ts) if shock_ts else 0
    if ref_t is not None:
        ts = int(ref_t)
    ts = min(ts, n_done)

    pre_px = px[max(0, W + ts - 4):W + ts + 1]
    p0 = float(sum(pre_px) / len(pre_px))
    post_px = px[W + ts + 1:W + n_done + 1]

    trough = crash = overshoot = recovery = None
    overshoot_up = None
    if post_px:
        i_min = min(range(len(post_px)), key=post_px.__getitem__)
        trough = float(post_px[i_min])
        crash = (p0 - trough) / p0
        overshoot = max(0.0, (F[W + ts + i_min] - trough) / p0)
        overshoot_up = max(0.0, max((post_px[i] - F[W + ts + i]) / p0 for i in range(len(post_px))))
        dev = [abs(post_px[i] - F[W + ts + i]) / F[W + ts + i] for i in range(len(post_px))]
        for j in range(len(dev) - 2):
            if dev[j] < 0.01 and dev[j + 1] < 0.01 and dev[j + 2] < 0.01:
                recovery = j
                break
    dislocation = float(sum(abs(px[W + t + 1] - F[W + t]) / F[W + t] for t in range(n_done)))

    lr = ex._lr                     # lr[W + t] is the log return of tick t
    lo = max(1, W + ts - _PRE_WINDOW)
    vol_pre = _std(lr[lo:W + ts])
    vol_post = _std(lr[W + ts:W + n_done])
    spread_pre = _mean_skip_none(ex._spread[max(0, W + ts - _PRE_WINDOW):W + ts])
    spread_post = _mean_skip_none(ex._spread[W + ts:W + n_done])
    depth_post = ex._depth[W + ts:W + n_done]
    min_depth_post = float(min(depth_post)) if depth_post else None

    volume_total = float(sum(ex._volume[W:W + n_done]))
    liq_volume = float(sum(ex._liq_volume[W:W + n_done]))
    n_liq = int(sum(ex._liq_count[W:W + n_done]))

    price = px[-1]
    groups = {}
    for a in ex._ext:
        groups.setdefault(a.group, []).append(a)
    pnl = {}
    bankrupt = {}
    breaches = {}
    for g, accts in groups.items():
        rets = [(a.cash + a.position * price) / a.cash0 - 1.0 for a in accts if a.cash0 > 0]
        pnl[g] = float(sum(rets) / len(rets)) if rets else None
        bankrupt[g] = int(sum(1 for a in accts if a.bankrupt))
        breaches[g] = int(sum(a.breaches for a in accts))
    bgs = [a for a in ex.accounts.values() if a.kind == "background" and a.cash0 > 0]
    if bgs:
        pnl["background"] = float(sum((a.cash + a.position * price) / a.cash0 - 1.0 for a in bgs) / len(bgs))

    out = {
        "p0": p0,
        "trough": trough,
        "crash_depth": crash,
        "overshoot": overshoot,
        "overshoot_up": overshoot_up,
        "dislocation_area": dislocation,
        "recovery_ticks": recovery,
        "vol_pre": vol_pre,
        "vol_post": vol_post,
        "mean_spread_bps_pre": spread_pre,
        "mean_spread_bps_post": spread_post,
        "min_depth_post": min_depth_post,
        "n_liquidations": n_liq,
        "liquidation_volume": liq_volume,
        "fire_sale_share": liq_volume / volume_total if volume_total > 0 else 0.0,
        "insurance_shortfall": float(ex._shortfall),
        "action_correlation": _action_correlation(ex, ts, n_done),
        "pnl_by_group": pnl,
        "bankruptcies_by_group": bankrupt,
        "mandate_breaches_by_group": breaches,
        "volume_total": volume_total,
    }
    for key, v in out.items():
        if isinstance(v, float) and not math.isfinite(v):
            out[key] = None
    return out
