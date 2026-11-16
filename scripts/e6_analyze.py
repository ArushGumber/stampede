"""E6: what each safeguard buys and what it costs.

Benefit: reduction in excess overshoot after FALSE headlines (unverified and verified-looking), relative to no safeguard.
Cost:    (a) slower pricing of TRUE news: excess dislocation area after a true headline, relative to no safeguard;
         (b) trading volume lost in normal markets (no-headline twins), as a ratio to no safeguard.
Input: results/e6/safeguards_cells.json (cells from e4_analyze) and the raw rows for the twins' volume.
Output: results/e6/safeguards.json, rows per (safeguard, pop, regime, capital).
"""
import collections
import glob
import gzip
import json
import os

import numpy as np

cells = json.load(open("results/e6/safeguards_cells.json"))
rows = []
for d in ("results/e4/safeguards", "results/e4/safeguards_steer"):
    p = os.path.join(d, "rows.jsonl.gz")
    if os.path.exists(p):
        rows += [json.loads(l) for l in gzip.open(p, "rt")]
twin_vol = collections.defaultdict(list)
for r in rows:
    if r["headline_kind"] == "none":
        twin_vol[(r["pop"], r["regime"], float(r["capital"]), r["safeguard_name"])].append(float(r["volume_total"] or 0))

STEER = {"r50_shift": ("r50", "steer_cautious"), "base_shift": ("gemma-base", "steer_cautious")}
idx = {}
for c in cells:
    pop, sg = c["pop"], c["safeguard_name"]
    if pop in STEER:
        pop, sg = STEER[pop]
    idx[(sg, pop, c["regime"], float(c["capital"]), c["hcode"])] = c

out = []
keys = sorted({(k[1], k[2], k[3]) for k in idx})
sgs = sorted({k[0] for k in idx})
for pop, regime, cap in keys:
    base = {h: idx.get(("none", pop, regime, cap, h)) for h in ("fake_unv", "fake_ver", "true_ver")}
    if not all(base.values()):
        continue
    for sg in sgs:
        cur = {h: idx.get((sg, pop, regime, cap, h)) for h in ("fake_unv", "fake_ver", "true_ver")}
        if not all(cur.values()):
            continue
        pop_key = {"steer_cautious": {"r50": "r50_shift", "gemma-base": "base_shift"}.get(pop, pop)}.get(sg, pop)
        v_sg = np.mean(twin_vol.get((pop_key, regime, cap, sg if sg != "steer_cautious" else "none"), [np.nan]))
        v_0 = np.mean(twin_vol.get((pop, regime, cap, "none"), [np.nan]))
        out.append({
            "safeguard": sg, "pop": pop, "regime": regime, "capital": cap,
            "fake_unv_overshoot": cur["fake_unv"]["excess_overshoot"], "fake_unv_lo": cur["fake_unv"]["excess_overshoot_lo"],
            "fake_unv_hi": cur["fake_unv"]["excess_overshoot_hi"],
            "fake_ver_overshoot": cur["fake_ver"]["excess_overshoot"], "fake_ver_lo": cur["fake_ver"]["excess_overshoot_lo"],
            "fake_ver_hi": cur["fake_ver"]["excess_overshoot_hi"],
            "reduction_unv": 1 - cur["fake_unv"]["excess_overshoot"] / base["fake_unv"]["excess_overshoot"] if base["fake_unv"]["excess_overshoot"] > 1e-4 else None,
            "reduction_ver": 1 - cur["fake_ver"]["excess_overshoot"] / base["fake_ver"]["excess_overshoot"] if base["fake_ver"]["excess_overshoot"] > 1e-4 else None,
            "true_dislocation": cur["true_ver"]["excess_dislocation_area"],
            "true_dislocation_base": base["true_ver"]["excess_dislocation_area"],
            "volume_ratio": float(v_sg / v_0) if v_0 and v_0 == v_0 else None,
            "liquidations_unv": cur["fake_unv"]["n_liquidations"],
            # contract section 6 fields for the front end
            "crash_depth": cur["fake_unv"]["excess_overshoot"], "lo": cur["fake_unv"]["excess_overshoot_lo"],
            "hi": cur["fake_unv"]["excess_overshoot_hi"], "population": pop,
            "depth_ratio": None,
        })
os.makedirs("results/e6", exist_ok=True)
json.dump(out, open("results/e6/safeguards.json", "w"), indent=1)
for r in out:
    if r["capital"] == 2.0:
        print(f"{r['regime']:6s} {r['pop']:13s} {r['safeguard']:16s} unv {100 * r['fake_unv_overshoot']:5.2f}% ver {100 * r['fake_ver_overshoot']:5.2f}% "
              f"true-disloc {r['true_dislocation']:+.3f} vol {r['volume_ratio'] if r['volume_ratio'] is None else round(r['volume_ratio'], 2)} liq {r['liquidations_unv']:.1f}")
