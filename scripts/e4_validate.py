"""Validate response-table agents against live GPU agents on matched episodes.

Both sides share configuration, seed, headline, group sizes and agent order (so the same random streams). Each
trace is scored from its saved price path with the same reference tick (stampede.sim.trace_metrics), and each
headline episode is paired with its own no-headline twin.

    python scripts/e4_validate.py
Writes results/e4/surrogate_validation.json.
"""
import collections
import glob
import gzip
import json
import os

import numpy as np

from stampede.sim.trace_metrics import trace_metrics


def load(d):
    out = {}
    for p in glob.glob(os.path.join(d, "*.json.gz")):
        tr = json.load(gzip.open(p, "rt"))
        c = tr["config"]
        pop = c.get("pop") or c["groups"][0]["name"].split("#")[0]
        key = (pop, c["regime"], float(c.get("capital_ratio") or c.get("adoption")), int(c["seed"]), c["headline_kind"] != "none")
        out[key] = trace_metrics(tr, c.get("shock_t") or 15)
    return out


L, S = load("results/episodes_live"), load("results/episodes_live_surr")
cells = collections.defaultdict(lambda: {"live": [], "surr": [], "live_path": [], "surr_path": []})
pairs = []
for k in sorted(L):
    pop, reg, cap, seed, head = k
    if not head:
        continue
    tw = (pop, reg, cap, seed, False)
    if k in S and tw in L and tw in S:
        lv = {m: L[k][m] - L[tw][m] for m in ("overshoot", "overshoot_up", "crash_depth", "dislocation_area", "n_liquidations")}
        sv = {m: S[k][m] - S[tw][m] for m in lv}
        pairs.append({"pop": pop, "regime": reg, "capital": cap, "seed": seed, **{f"live_{m}": v for m, v in lv.items()},
                      **{f"surr_{m}": v for m, v in sv.items()}})
        c = cells[(pop, reg, cap)]
        c["live"].append(lv["overshoot"])
        c["surr"].append(sv["overshoot"])
        if "mean_exposure_path" in L[k] and "mean_exposure_path" in S[k]:
            c["live_path"].append(L[k]["mean_exposure_path"])
            c["surr_path"].append(S[k]["mean_exposure_path"])
summary = {"n_pairs": len(pairs), "n_cells": len(cells)}
if cells:
    a = np.array([np.mean(v["live"]) for v in cells.values()])
    b = np.array([np.mean(v["surr"]) for v in cells.values()])
    summary.update(cell_corr=float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else None,
                   cell_mae_pct=float(100 * np.abs(a - b).mean()), live_mean_pct=float(100 * a.mean()),
                   surr_mean_pct=float(100 * b.mean()))
    path_err = [float(np.abs(np.mean(v["live_path"], 0) - np.mean(v["surr_path"], 0)).mean()) for v in cells.values() if v["live_path"]]
    summary["mean_exposure_path_mae"] = float(np.mean(path_err)) if path_err else None
out = {"summary": summary, "pairs": pairs,
       "cells": [{"pop": k[0], "regime": k[1], "capital": k[2], "live_pct": 100 * float(np.mean(v["live"])),
                  "surr_pct": 100 * float(np.mean(v["surr"])), "n": len(v["live"]),
                  "live_path": np.mean(v["live_path"], 0).round(4).tolist() if v["live_path"] else None,
                  "surr_path": np.mean(v["surr_path"], 0).round(4).tolist() if v["surr_path"] else None}
                 for k, v in cells.items()]}
os.makedirs("results/e4", exist_ok=True)
json.dump(out, open("results/e4/surrogate_validation.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
for c in out["cells"]:
    print(f"{c['pop']:8s} {c['regime']:6s} cap {c['capital']:g}: live {c['live_pct']:.2f}% surr {c['surr_pct']:.2f}% (n={c['n']})")
