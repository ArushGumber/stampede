"""Place closed models on the open model's dial: for each metric, the steering strength at which the steered
open model matches the closed model's measured value (linear interpolation along the dose curve).

Intervals: bootstrap over items of the closed model's battery (arrays saved by e3_closed.py), each resample
mapped through the same curve. Values outside the curve's range are reported as censored ("below" / "above").

    python scripts/e3_equivalent.py --model gemma-4-12b --layer 24
Writes results/e3/alpha_equivalent.json
"""
import argparse
import glob
import json
import os

import numpy as np

from stampede.steering.battery import Battery, market_states

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemma-4-12b")
ap.add_argument("--layer", type=int, default=24)
ap.add_argument("--method", default="syslast")
ap.add_argument("--boot", type=int, default=1000)
args = ap.parse_args()

PAIRS = [("risk", "kelly_fraction"), ("risk", "mean_exposure"), ("risk", "crra_gamma"), ("credulity", "credulity"),
         ("credulity", "react_unverified"), ("integrity", "unethical_rate")]
rng = np.random.default_rng(0)


def curve(trait, metric):
    p = f"results/e1/{args.model}/dose_{trait}_{args.method}_L{args.layer}.json"
    if not os.path.exists(p):
        return None
    recs = [r for r in json.load(open(p))["records"] if r["kind"] == "steer" and metric in r and r[metric] == r[metric]]
    recs.sort(key=lambda r: r["alpha"])
    if len(recs) < 3:
        return None
    return np.array([r["alpha"] for r in recs]), np.array([r[metric] for r in recs])


def invert(a, v, target):
    """Smallest-|alpha| crossing of the curve with target; censored if none."""
    best = None
    for i in range(len(a) - 1):
        lo, hi = sorted((v[i], v[i + 1]))
        if lo <= target <= hi and v[i] != v[i + 1]:
            x = a[i] + (target - v[i]) * (a[i + 1] - a[i]) / (v[i + 1] - v[i])
            if best is None or abs(x) < abs(best):
                best = x
    if best is not None:
        return float(best), None
    return None, ("below" if target < v.min() else "above")


# item structure needed to recompute metrics from resampled arrays
states = market_states("equity")
cats = np.array([s["news"] for s in states])
keys = [(s["ret5"], s["exposure"]) for s in states]
maxlev = 2.0


def credulity_from(expo):
    base = {k: expo[i] for i, k in enumerate(keys) if cats[i] == "none"}
    def react(c):
        return np.mean([(base[keys[i]] - expo[i]) / maxlev for i in range(len(expo)) if cats[i] == c])
    ru, rv = react("unv_bear_fresh"), react("ver_bear_fresh")
    return ru / rv if abs(rv) > 1e-3 else np.nan, ru


out = []
for p in sorted(glob.glob("results/e3/*/battery.json")):
    d = json.load(open(p))
    agent, s = d["meta"]["agent"], d["summary"]
    for trait, metric in PAIRS:
        c = curve(trait, metric)
        if c is None or metric not in s:
            continue
        a, v = c
        x, cens = invert(a, v, s[metric])
        samples = []
        if metric in ("credulity", "react_unverified") and "arr_exposure" in s and len(s["arr_exposure"]) == len(states):
            ex = np.array(s["arr_exposure"])
            # resample state groups: (ret5, exposure) blocks keep the matched pairs together
            blocks = sorted(set(keys))
            for _ in range(args.boot):
                pick = [blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]
                idx = [i for b in pick for i in range(len(keys)) if keys[i] == b]
                sub_keys = [keys[i] for i in idx]
                sub_cats = cats[idx]
                sub_ex = ex[idx]
                base = {}
                for k2, c2, e2 in zip(sub_keys, sub_cats, sub_ex):
                    if c2 == "none":
                        base.setdefault(k2, []).append(e2)
                base = {k2: np.mean(vv) for k2, vv in base.items()}
                def rx(c3):
                    vals = [(base[k2] - e2) / maxlev for k2, c2, e2 in zip(sub_keys, sub_cats, sub_ex) if c2 == c3 and k2 in base]
                    return np.mean(vals) if vals else np.nan
                ru, rv = rx("unv_bear_fresh"), rx("ver_bear_fresh")
                val = (ru / rv if abs(rv) > 1e-3 else np.nan) if metric == "credulity" else ru
                xb, _ = invert(a, v, val)
                if xb is not None:
                    samples.append(xb)
        elif metric == "kelly_fraction" and "arr_stake" in s:
            from stampede.steering.battery import kelly_items
            k = np.array([it["kelly"] for it in kelly_items()])
            st = np.array(s["arr_stake"])
            pos = np.where(k > 0)[0]
            for _ in range(args.boot):
                j = rng.choice(pos, len(pos))
                xb, _ = invert(a, v, st[j].sum() / k[j].sum())
                if xb is not None:
                    samples.append(xb)
        lo, hi = (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))) if len(samples) > 20 else (None, None)
        out.append({"agent": agent, "trait": trait, "metric": metric, "value": s[metric], "alpha_equivalent": x,
                    "censored": cens, "lo": lo, "hi": hi, "curve_range": [float(v.min()), float(v.max())],
                    "harness": d["meta"]["harness"], "date": d["meta"]["date"]})
json.dump(out, open("results/e3/alpha_equivalent.json", "w"), indent=1)
for r in out:
    print(f"{r['agent']:14s} {r['metric']:16s} value {r['value']:+.3f} -> alpha {r['alpha_equivalent'] if r['alpha_equivalent'] is None else round(r['alpha_equivalent'], 2)} "
          f"{r['censored'] or ''} [{r['lo'] and round(r['lo'], 2)}, {r['hi'] and round(r['hi'], 2)}]")
