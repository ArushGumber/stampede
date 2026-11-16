"""Aggregate sweep rows into the tipping-point result set, with headline-minus-twin effects and bootstrap intervals.

    python scripts/e4_analyze.py results/e4/<name> [more dirs] --out results/e4/tipping_point.json
"""
import argparse
import collections
import gzip
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--out", default="results/e4/tipping_point.json")
ap.add_argument("--boot", type=int, default=2000)
args = ap.parse_args()

rows = []
for d in args.dirs:
    p = os.path.join(d, "rows.jsonl.gz")
    if os.path.exists(p):
        rows += [json.loads(l) for l in gzip.open(p, "rt")]

KEYS = ("tag", "pop", "regime", "hcode", "capital", "safeguard_name", "resilience", "depth_scale")
pairs = collections.defaultdict(dict)
for r in rows:
    k = tuple(r.get(x) for x in KEYS) + (r["seed"],)
    pairs[k]["twin" if r["headline_kind"] == "none" else "head"] = r

cells = collections.defaultdict(list)
for k, pr in pairs.items():
    if "twin" in pr and "head" in pr:
        cells[k[:-1]].append(pr)
for k in cells:  # seed order, so episode_ids start with the seed whose trace was saved (seed 0)
    cells[k].sort(key=lambda pr: int(pr["head"]["seed"]))

rng = np.random.default_rng(0)


def boot(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float(x[0]), float(x[0])
    idx = rng.integers(0, len(x), size=(args.boot, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def f(r, k):
    v = r.get(k)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


out = []
for k, prs in sorted(cells.items(), key=lambda kv: [str(x) for x in kv[0]]):
    rec = dict(zip(KEYS, k))
    rec["n"] = len(prs)
    for m in ("crash_depth", "overshoot", "overshoot_up", "dislocation_area", "n_liquidations", "liquidation_volume", "action_correlation",
              "volume_total", "mean_spread_bps_post", "min_depth_post"):
        head = [f(p["head"], m) for p in prs]
        twin = [f(p["twin"], m) for p in prs]
        v, lo, hi = boot(np.array(head) - np.array(twin))
        rec[f"excess_{m}"], rec[f"excess_{m}_lo"], rec[f"excess_{m}_hi"] = v, lo, hi
        rec[m] = float(np.nanmean(head))
    rec["recovery_ticks"] = float(np.nanmean([f(p["head"], "recovery_ticks") for p in prs]))
    rec["p_crash_5pct"] = float(np.mean([f(p["head"], "overshoot") > 0.05 for p in prs]))
    rec["episode_ids"] = [p["head"]["id"] for p in prs][:3]
    rec["headline"] = rec["hcode"]
    rec["adoption"] = rec["capital"]
    rec["safeguard"] = rec["safeguard_name"]
    out.append(rec)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(out, open(args.out, "w"), indent=0, default=float)
print(f"{len(rows)} rows -> {len(out)} cells -> {args.out}")
