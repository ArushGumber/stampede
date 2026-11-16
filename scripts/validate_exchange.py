"""E0: is the exchange sane? Background-only calibration plus shock scenarios, for both regimes and all
three background resilience settings ("high", "medium" = the regime() default, "low").

    .venv/bin/python scripts/validate_exchange.py            # full run, about a minute on one CPU core
    .venv/bin/python scripts/validate_exchange.py --quick    # fewer seeds and ticks

Prints a table and writes results/engine_validation.json (served as /api/results/engine_validation).

Scenarios (background agents only, apart from the dumping accounts):
  calm     long background-only runs: per-tick return sd, spread, depth within 1% of mid, excess kurtosis,
           lag-1 autocorrelation of absolute and raw returns, tracking error to the fundamental
  fake     an unverified headline at t=20 with no fundamental move, corrected at t=35: largest price change
           versus the same seed without the headline (background agents do not read news: should be zero)
  true     the fundamental jumps by -5% (log) at t=20: recovery_ticks (ticks until |price/fundamental - 1|
           < 1% for 3 ticks), overshoot past the new fundamental
  dump     8 external accounts holding a combined 20% or 100% of depth_notional (initial_exposure=1) all go
           flat with market orders at t=20. gap = price / price of the same seed without the dump - 1,
           averaged over seeds. Reported: the peak gap (t=20..23), its half-life (ticks from the peak until
           the mean gap is half the peak), and the whole mean gap curve with 10th and 90th percentiles.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import kurtosis

from stampede.exchange import Exchange, NewsItem, Shock, TargetExposure, regime

SETTINGS = ("high", "medium", "low")
SHOCK_T = 20
N_TICKS = 60
# targets apply to the default ("medium") unless a setting is named
TARGETS = {
    "equity": {"ret_sd_pct": (0.091, 0.111), "spread_bps": (2.0, 5.0), "true_recovery_ticks": (10, 20),
               "dump100_peak_pct": (-3.0, -1.5), "dump100_half_life": (4, 10)},
    "crypto": {"ret_sd_pct": (0.200, 0.244), "spread_bps": (5.0, 10.0), "true_recovery_ticks": (10, 20),
               "dump100_peak_pct": (-3.0, -1.5), "dump100_half_life": (4, 10)},
}
HIGH_TARGETS = {"true_recovery_ticks": (5, 15)}


def cfg_for(name, res, seed, n_ticks=N_TICKS, vol_of_vol=None):
    cfg = regime(name, seed=seed, n_ticks=n_ticks)
    cfg.background.resilience = res
    if vol_of_vol is not None:
        cfg.fundamental.vol_of_vol = vol_of_vol
    return cfg


def acf1(x):
    x = np.asarray(x, float)
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def run(cfg, dump_frac=0.0):
    ex = Exchange(cfg)
    if dump_frac:
        D = cfg.background.depth_notional
        for i in range(8):
            ex.add_account(f"d{i}", cash=dump_frac * D / 8, group="dump", initial_exposure=1.0)
    while not ex.done:
        if dump_frac and ex.t == SHOCK_T:
            for i in range(8):
                ex.submit(f"d{i}", TargetExposure(0.0, "market"))
        ex.step()
    return ex


def calm(name, res, seeds, n_ticks, vol_of_vol=None):
    rows = []
    for seed in seeds:
        cfg = cfg_for(name, res, seed, n_ticks, vol_of_vol)
        ex = run(cfg)
        s = ex.series()
        p, F = np.array(s["price"]), np.array(s["fundamental"])
        r = np.diff(np.log(p))
        sp = np.array([x for x in s["spread_bps"] if x is not None])
        rows.append({
            "ret_sd_pct": r.std() * 100, "fund_ret_sd_pct": np.diff(np.log(F)).std() * 100,
            "spread_bps_median": float(np.median(sp)), "spread_bps_mean": float(sp.mean()),
            "depth_over_nominal": float(np.mean(s["depth"]) / cfg.background.depth_notional),
            "depth_p10_over_nominal": float(np.percentile(s["depth"], 10) / cfg.background.depth_notional),
            "excess_kurtosis": float(kurtosis(r)), "abs_ret_acf1": acf1(np.abs(r)), "ret_acf1": acf1(r),
            "tracking_rms_bps": float(np.sqrt(np.mean(np.log(p / F) ** 2)) * 1e4),
            "volume_notional_per_tick_over_depth": float(np.mean(s["volume"]) * p.mean()
                                                         / cfg.background.depth_notional),
        })
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    out.update(n_seeds=len(seeds), n_ticks=n_ticks)
    return out


def fake(name, res, seeds):
    worst = 0.0
    for seed in seeds:
        a = run(cfg_for(name, res, seed))
        cfg = cfg_for(name, res, seed)
        cfg.shocks = [Shock(t=SHOCK_T, kind="fake", news=NewsItem("unverified social post", False, "Unconfirmed: raid")),
                      Shock(t=SHOCK_T + 15, kind="correction")]
        b = run(cfg)
        pa, pb = np.array(a.series()["price"]), np.array(b.series()["price"])
        worst = max(worst, float(np.max(np.abs(pb / pa - 1))))
    return {"max_abs_price_diff_vs_no_news_pct": worst * 100, "n_seeds": len(seeds)}


def true_shock(name, res, seeds, jump=-0.05):
    rec, over, curves = [], [], []
    for seed in seeds:
        cfg = cfg_for(name, res, seed)
        cfg.shocks = [Shock(t=SHOCK_T, kind="true", jump=jump, news=NewsItem("company filing", True, "Guidance cut"))]
        ex = run(cfg)
        m = ex.metrics()
        s = ex.series()
        rec.append(m["recovery_ticks"])
        over.append(m["overshoot"])
        curves.append(np.array(s["price"]) / np.array(s["fundamental"]) - 1)
    r = [x for x in rec if x is not None]
    return {"jump": jump, "recovery_ticks_mean": float(np.mean(r)) if r else None,
            "recovery_ticks_min": min(r) if r else None, "recovery_ticks_max": max(r) if r else None,
            "never_recovered": sum(x is None for x in rec), "overshoot_pct_mean": float(np.mean(over)) * 100,
            "n_seeds": len(seeds),
            "price_over_fundamental_pct_mean": [round(float(x) * 100, 4) for x in np.mean(curves, axis=0)]}


def dump(name, res, seeds, frac):
    gaps = []
    for seed in seeds:
        a = run(cfg_for(name, res, seed))
        b = run(cfg_for(name, res, seed), dump_frac=frac)
        gaps.append(np.array(b.series()["price"]) / np.array(a.series()["price"]) - 1)
    G = np.array(gaps) * 100
    g = G.mean(axis=0)
    ipk = SHOCK_T + int(np.argmin(g[SHOCK_T:SHOCK_T + 4]))
    peak = float(g[ipk])
    hl = next((j - ipk for j in range(ipk, len(g)) if g[j] >= peak / 2), None) if peak < 0 else None
    c75 = next((j - ipk for j in range(ipk, len(g)) if g[j] >= peak / 4), None) if peak < 0 else None
    worst = [float(x[SHOCK_T:SHOCK_T + 4].min()) for x in G]
    return {"dump_notional_over_depth": frac, "peak_gap_pct": peak, "peak_tick": ipk,
            "half_life_ticks": hl, "ticks_to_close_75pct": c75,
            "per_seed_worst_gap_pct_min": min(worst), "per_seed_worst_gap_pct_max": max(worst),
            "gap_after_10_ticks_pct": float(g[min(len(g) - 1, ipk + 10)]), "n_seeds": len(seeds),
            "curve": {"t": list(range(len(g))), "dump_t": SHOCK_T,
                      "gap_pct_mean": [round(float(x), 4) for x in g],
                      "gap_pct_p10": [round(float(x), 4) for x in np.percentile(G, 10, axis=0)],
                      "gap_pct_p90": [round(float(x), 4) for x in np.percentile(G, 90, axis=0)]}}


def timing(n=10):
    out = {}
    for res in SETTINGS:
        t0 = time.perf_counter()
        for seed in range(n):
            ex = Exchange(cfg_for("equity", res, seed))
            for i in range(32):
                ex.add_account(f"a{i:02d}", cash=50_000.0, initial_exposure=1.0)
            while not ex.done:
                for i in range(32):
                    ex.observe(f"a{i:02d}")
                    ex.submit(f"a{i:02d}", TargetExposure(x=((i + ex.t) % 5 - 2) * 0.5))
                ex.step()
            ex.trace()
        out[res] = (time.perf_counter() - t0) / n * 1000
    return out


def reference():
    path = ROOT / "data" / "replay" / "calibration.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())["series"]
    out = {}
    for k in ("SPY_1m", "AAPL_1m", "MSFT_1m", "BTCUSDT_1m_calm", "BTCUSDT_1m_stress", "LINKUSDT_1m_calm",
              "LINKUSDT_1m_stress"):
        if k in d:
            s = d[k].get("excluding_suspected_bad_ticks", d[k])
            out[k] = {"ret_sd_pct": s["return_std"] * 100, "excess_kurtosis": s["excess_kurtosis"],
                      "abs_ret_acf1": s["abs_return_lag1_autocorr"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "engine_validation.json"))
    args = ap.parse_args()
    calm_seeds, calm_ticks = ((1, 2), 1500) if args.quick else ((1, 2, 3, 4), 3000)
    seeds = range(6) if args.quick else range(16)

    t_start = time.perf_counter()
    res = {"settings": list(SETTINGS), "default_setting": regime("equity").background.resilience,
           "regimes": {}, "targets": {"medium": TARGETS, "high": HIGH_TARGETS},
           "reference_minute_bars": reference()}
    for name in ("equity", "crypto"):
        cfg = regime(name)
        entry = {"config": {"symbol": cfg.symbol, "asset_name": cfg.asset_name, "initial_price": cfg.initial_price,
                            "tick_size": cfg.tick_size, "max_leverage": cfg.max_leverage,
                            "maintenance_margin": cfg.maintenance_margin, "fee_bps": cfg.fee_bps,
                            "depth_notional": cfg.background.depth_notional,
                            "fundamental_sigma": cfg.fundamental.sigma,
                            "fundamental_vol_of_vol": cfg.fundamental.vol_of_vol},
                 "settings": {}}
        for r in SETTINGS:
            entry["settings"][r] = {
                "calm": calm(name, r, calm_seeds, calm_ticks),
                "fake_shock": fake(name, r, seeds),
                "true_shock": true_shock(name, r, seeds),
                "dump_20pct_depth": dump(name, r, seeds, 0.2),
                "dump_100pct_depth": dump(name, r, seeds, 1.0),
            }
            print(f"  done {name} {r} ({time.perf_counter() - t_start:.0f}s)", flush=True)
        entry["medium_calm_without_stochastic_vol"] = calm(name, "medium", calm_seeds[:2], calm_ticks, vol_of_vol=0.0)
        res["regimes"][name] = entry
    res["timing_ms_60_ticks_32_accounts"] = timing()
    res["dump_recovery_curves"] = {
        name: {r: res["regimes"][name]["settings"][r]["dump_100pct_depth"]["curve"] for r in SETTINGS}
        for name in res["regimes"]}

    rows = []
    METRICS = [
        ("per-tick return sd (%)", lambda s: s["calm"]["ret_sd_pct"], "ret_sd_pct", "{:.4f}"),
        ("median spread (bps)", lambda s: s["calm"]["spread_bps_median"], "spread_bps", "{:.2f}"),
        ("depth within 1% / depth_notional", lambda s: s["calm"]["depth_over_nominal"], None, "{:.3f}"),
        ("excess kurtosis", lambda s: s["calm"]["excess_kurtosis"], None, "{:.2f}"),
        ("lag-1 autocorr |r|", lambda s: s["calm"]["abs_ret_acf1"], None, "{:.3f}"),
        ("lag-1 autocorr r", lambda s: s["calm"]["ret_acf1"], None, "{:+.3f}"),
        ("tracking error rms (bps)", lambda s: s["calm"]["tracking_rms_bps"], None, "{:.1f}"),
        ("fake shock max |move| (%)", lambda s: s["fake_shock"]["max_abs_price_diff_vs_no_news_pct"], None, "{:.4f}"),
        ("true -5%: recovery_ticks mean", lambda s: s["true_shock"]["recovery_ticks_mean"], "true_recovery_ticks", "{:.1f}"),
        ("true -5%: recovery_ticks max", lambda s: s["true_shock"]["recovery_ticks_max"], None, "{:.0f}"),
        ("true -5%: never recovered (of n)", lambda s: s["true_shock"]["never_recovered"], None, "{:.0f}"),
        ("dump 20%: peak gap (%)", lambda s: s["dump_20pct_depth"]["peak_gap_pct"], None, "{:.2f}"),
        ("dump 20%: half-life (ticks)", lambda s: s["dump_20pct_depth"]["half_life_ticks"], None, "{:.0f}"),
        ("dump 100%: peak gap (%)", lambda s: s["dump_100pct_depth"]["peak_gap_pct"], "dump100_peak_pct", "{:.2f}"),
        ("dump 100%: half-life (ticks)", lambda s: s["dump_100pct_depth"]["half_life_ticks"], "dump100_half_life", "{:.0f}"),
        ("dump 100%: gap 10 ticks after peak (%)", lambda s: s["dump_100pct_depth"]["gap_after_10_ticks_pct"], None, "{:.2f}"),
    ]
    head = "".join(f"{n[:2]}-{r:<7s}" for n in ("equity", "crypto") for r in SETTINGS)
    print(f"\n{'metric':40s} " + " ".join(f"{n[:2] + ' ' + r:>9s}" for n in ("equity", "crypto") for r in SETTINGS))
    print("-" * 102)
    misses = []
    for label, get, tkey, fmt in METRICS:
        cells = []
        for name in ("equity", "crypto"):
            for r in SETTINGS:
                v = get(res["regimes"][name]["settings"][r])
                tgt = None
                if tkey and r == "medium":
                    tgt = TARGETS[name].get(tkey)
                elif tkey and r == "high":
                    tgt = HIGH_TARGETS.get(tkey)
                ok = None if (tgt is None or v is None) else bool(tgt[0] <= v <= tgt[1])
                rows.append({"regime": name, "setting": r, "metric": label, "value": v, "target": tgt, "ok": ok})
                if ok is False:
                    misses.append(rows[-1])
                cell = "n/a" if v is None else fmt.format(v)
                cells.append(f"{cell + ('*' if ok is False else ''):>9s}")
        print(f"{label:40s} " + " ".join(cells))
    t = res["timing_ms_60_ticks_32_accounts"]
    print(f"{'60 ticks, 32 accounts, wall time (ms)':40s} " + " ".join(f"{t[r]:9.1f}" for r in SETTINGS))
    for name in ("equity", "crypto"):
        nv = res["regimes"][name]["medium_calm_without_stochastic_vol"]
        print(f"{name} medium, fundamental without stochastic vol: kurtosis {nv['excess_kurtosis']:.2f}, "
              f"|r| acf1 {nv['abs_ret_acf1']:.3f}")
    if res["reference_minute_bars"]:
        print("\nreal 1-minute bars (data/replay/calibration.json):")
        for k, v in res["reference_minute_bars"].items():
            print(f"  {k:20s} return sd {v['ret_sd_pct']:.4f}%  excess kurtosis {v['excess_kurtosis']:6.2f}  "
                  f"|r| acf1 {v['abs_ret_acf1']:.3f}")
    res["table"] = rows
    res["runtime_s"] = time.perf_counter() - t_start
    res["quick"] = bool(args.quick)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, allow_nan=False))
    print(f"\nwrote {out} in {res['runtime_s']:.1f}s; {len(misses)} target(s) missed (marked *)")
    for r in misses:
        print("  MISS:", r["regime"], r["setting"], r["metric"], r["value"], r["target"])


if __name__ == "__main__":
    main()
