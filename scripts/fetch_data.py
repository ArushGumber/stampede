#!/usr/bin/env python
"""Fetch minute-bar data for Stampede regime calibration.

Crypto: Binance public data (data.binance.vision daily 1m kline zips; falls back to
the public REST klines API if a daily file is missing). Equity: yfinance 1m (last ~7
days) and 5m (last ~60 days) bars. Writes CSVs and calibration.json to data/replay/.
Nothing is synthesised; failures are recorded in calibration.json "notes".

Run: .venv/bin/python scripts/fetch_data.py
"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

OUT = Path(__file__).resolve().parent.parent / "data" / "replay"
OUT.mkdir(parents=True, exist_ok=True)

CRYPTO_PAIRS = ["BTCUSDT", "LINKUSDT"]  # LINKUSDT is the mid-cap pair
# Chosen from BTCUSDT daily (high-low)/close ranges: lowest ranges since 2024 (calm)
# and the largest ranges since 2024 (stress).
CALM_DAYS = ["2025-09-21", "2026-01-10", "2026-08-15"]
STRESS_DAYS = ["2024-08-05", "2025-10-10", "2026-02-05"]
EQUITIES = ["SPY", "AAPL", "MSFT"]  # broad ETF + two large single stocks
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
notes = []


def fetch_day_vision(sym, day):
    url = f"https://data.binance.vision/data/spot/daily/klines/{sym}/1m/{sym}-1m-{day}.zip"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(z.open(z.namelist()[0]), header=None, names=COLS)
    return df, url


def fetch_day_api(sym, day):
    t0 = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
    frames = []
    for start in (t0, t0 + 1000 * 60000):
        r = requests.get("https://api.binance.com/api/v3/klines", timeout=30,
                         params=dict(symbol=sym, interval="1m", startTime=start,
                                     endTime=t0 + 86400000 - 1, limit=1000))
        r.raise_for_status()
        frames.append(pd.DataFrame(r.json(), columns=COLS))
    return pd.concat(frames), "api.binance.com/api/v3/klines"


def to_time(col):
    col = col.astype("int64")
    unit = "us" if col.iloc[0] > 1e14 else "ms"  # vision switched to microseconds in 2025
    return pd.to_datetime(col, unit=unit, utc=True)


def stats(ret, hl_bps):
    ret = ret.dropna()
    a = ret.abs()
    return dict(
        n_bars=int(len(ret)),
        return_std=float(ret.std()),
        excess_kurtosis=float(ret.kurt()),
        abs_return_lag1_autocorr=float(a.autocorr(1)),
        median_bar_range_bps=float(hl_bps.median()),
    )


calib = {"series": {}, "notes": notes,
         "definitions": {
             "return": "log close-to-close return per bar; equity returns exclude overnight gaps",
             "excess_kurtosis": "pandas Fisher kurtosis (normal = 0)",
             "median_bar_range_bps": "median of (high-low)/close * 1e4"}}

# ---- crypto ----
for sym in CRYPTO_PAIRS:
    allbars = []
    for regime, days in (("calm", CALM_DAYS), ("stress", STRESS_DAYS)):
        for day in days:
            try:
                try:
                    df, src = fetch_day_vision(sym, day)
                except Exception as e:  # noqa: BLE001
                    notes.append(f"{sym} {day}: data.binance.vision failed ({e}); used REST API")
                    df, src = fetch_day_api(sym, day)
                df["time"] = to_time(df["open_time"])
                df = df[["time", "open", "high", "low", "close", "volume"]].copy()
                df = df.sort_values("time").drop_duplicates("time")
                for c in ["open", "high", "low", "close", "volume"]:
                    df[c] = df[c].astype(float)
                df["regime"] = regime
                df["day"] = day
                allbars.append(df)
            except Exception as e:  # noqa: BLE001
                notes.append(f"FAILED {sym} {day} ({regime}): {e}")
    if not allbars:
        continue
    d = pd.concat(allbars).reset_index(drop=True)
    d.to_csv(OUT / f"{sym}_1m.csv", index=False)
    d["ret"] = np.log(d["close"]).groupby(d["day"]).diff()
    d["hl"] = (d["high"] - d["low"]) / d["close"] * 1e4
    for regime in ("calm", "stress"):
        s = d[d.regime == regime]
        # autocorrelation computed within each day, then pooled by concatenation
        calib["series"][f"{sym}_1m_{regime}"] = stats(s["ret"], s["hl"]) | {
            "days": sorted(s["day"].unique().tolist())}

# ---- equities ----
try:
    import yfinance as yf
except Exception as e:  # noqa: BLE001
    yf = None
    notes.append(f"yfinance import failed: {e}")
if yf:
    for interval, period in (("1m", "7d"), ("5m", "60d")):
        for sym in EQUITIES:
            try:
                d = yf.download(sym, period=period, interval=interval, progress=False,
                                auto_adjust=False)
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                d = d.dropna().rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
                if d.empty:
                    raise RuntimeError("empty frame")
                d.index = d.index.tz_convert("UTC")
                d.index.name = "time"
                d.to_csv(OUT / f"{sym}_{interval}.csv")
                day = d.index.tz_convert("America/New_York").date
                ret = np.log(d["close"]).groupby(day).diff()
                hl = (d["high"] - d["low"]) / d["close"] * 1e4
                # suspected bad ticks: a >0.5% bar that is >75% reversed by the next bar
                nxt = ret.shift(-1)
                bad = (ret.abs() > 0.005) & ((ret + nxt).abs() < 0.25 * ret.abs())
                entry = stats(ret, hl) | {"window": [str(d.index.min()), str(d.index.max())]}
                bad = bad | bad.shift(1, fill_value=False)  # drop the reversal bar too
                if bad.any():
                    entry["suspected_bad_tick_bars"] = [str(t) for t in d.index[bad.values]]
                    entry["excluding_suspected_bad_ticks"] = stats(ret[~bad], hl[~bad])
                    notes.append(f"{sym} {interval}: {int(bad.sum())} suspected vendor bad-tick bar(s) "
                                 "flagged (single-bar spike reversed next bar); raw stats kept, "
                                 "cleaned stats in excluding_suspected_bad_ticks")
                calib["series"][f"{sym}_{interval}"] = entry
            except Exception as e:  # noqa: BLE001
                notes.append(f"FAILED {sym} {interval}: {e}")

notes.append("Crypto calm days are weekend days; BTCUSDT spot bar ranges on them are near the "
             "one-tick floor (median range under 1 bp), so BTC calm vs stress differ by ~2 orders of magnitude.")
notes.append("Equity 1m window is the last ~6 sessions, 5m window the last ~60 calendar days; "
             "vendor data is unadjusted-for-splits and may contain bad ticks.")
(OUT / "calibration.json").write_text(json.dumps(calib, indent=1))
print(json.dumps(calib, indent=1))
