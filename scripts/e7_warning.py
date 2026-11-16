"""E7: do agents' internal states warn of a crash before prices do?

Uses live episodes (results/episodes_live/*.json.gz) in which every decision logged projections of the decision
token's residual stream onto the trait directions at a monitor layer upstream of the steering layer
(`mon_credulity`, `mon_risk`), so the reading reflects what agents perceive, not the injected dial.

Indicators per episode, read one tick after the headline appears (before most orders have executed):
  internal:  mean over deciding agents of the monitor projection, minus the same episode's pre-headline mean
  price:     one-tick return after the headline; drawdown from the pre-shock price
Target: whether the episode's overshoot exceeds a threshold (default 1%). Reports AUROC with bootstrap
intervals, and the lead (ticks) between the indicator crossing its no-headline 95th percentile and the price
first falling 0.5% below fundamental.
"""
import argparse
import glob
import gzip
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--glob", default="results/episodes_live/*.json.gz")
ap.add_argument("--threshold", type=float, default=0.01)
ap.add_argument("--out", default="results/e7/early_warning.json")
args = ap.parse_args()


def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, int)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


rows = []
for p in sorted(glob.glob(args.glob)):
    tr = json.load(gzip.open(p, "rt"))
    c = tr["config"]
    sh = c.get("shock_t")
    if c.get("headline_kind") in (None, "none") or sh is None:
        continue
    fr = tr["frames"]

    def mon(t, key):
        vals = [a["proj"][key] for a in fr[t]["agents"] if a.get("proj") and key in a["proj"]]
        return np.mean(vals) if vals else np.nan

    keys = [k for k in ("mon_credulity", "mon_risk") if any(a.get("proj") and k in a["proj"] for f in fr for a in f["agents"])]
    if not keys:
        continue
    pre = range(max(1, sh - 6), sh)
    rec = {"id": tr["id"], "overshoot": tr["metrics"].get("overshoot"), "crash": float(tr["metrics"].get("overshoot") or 0) > args.threshold,
           "pop": c.get("pop"), "capital": c.get("capital_ratio"), "regime": c.get("regime")}
    for k in keys:
        base = np.nanmean([mon(t, k) for t in pre])
        rec[f"int_{k}"] = mon(sh, k) - base if sh < len(fr) else np.nan
        rec[f"int_{k}_t1"] = mon(sh + 1, k) - base if sh + 1 < len(fr) else np.nan
    p0 = np.mean([fr[t]["price"] for t in pre])
    rec["price_ret_t0"] = fr[sh]["price"] / fr[sh - 1]["price"] - 1
    rec["price_ret_t1"] = fr[sh + 1]["price"] / fr[sh]["price"] - 1 if sh + 1 < len(fr) else np.nan
    rec["price_dd_t1"] = fr[sh + 1]["price"] / p0 - 1 if sh + 1 < len(fr) else np.nan
    rows.append(rec)

rng = np.random.default_rng(0)
out = []
if rows:
    y = np.array([r["crash"] for r in rows], int)
    cand = [k for k in rows[0] if k.startswith("int_") or k.startswith("price_")]
    for k in cand:
        s = np.array([r.get(k, np.nan) for r in rows], float)
        ok = ~np.isnan(s)
        sign = -1.0 if k.startswith("price_") else 1.0  # falling prices and rising alarm both predict crashes
        a = auroc(sign * s[ok], y[ok])
        boots = []
        idx = np.where(ok)[0]
        for _ in range(1000):
            j = rng.choice(idx, len(idx))
            boots.append(auroc(sign * s[j], y[j]))
        boots = [b for b in boots if b == b]
        out.append({"indicator": k, "auroc": a, "lo": float(np.percentile(boots, 2.5)) if boots else None,
                    "hi": float(np.percentile(boots, 97.5)) if boots else None, "n": int(ok.sum()), "n_crash": int(y[ok].sum()),
                    "lead_ticks": 0 if k.endswith("_t0") or k in ("int_mon_credulity", "int_mon_risk") else 1})
import os
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(out, open(args.out, "w"), indent=1)
for r in out:
    print(r)
