"""E4/E6: market sweeps with response-table agents (CPU, parallel).

Every headline episode is paired with a twin that has the same seed and no headline, so effects are
reported as excess over the market's own noise.

    python scripts/e4_sweep.py --name tip_risk --pops 'risk-1=gemma-4-12b_risk-1;risk0=gemma-4-12b_base;...' \
        --capital 0.05,0.1,0.2,0.35,0.5,0.75,1,1.5 --seeds 8 --regimes equity,crypto --headlines fake_unv,true_ver

A population spec is name=table[:fraction],table[:fraction],... where table is a surrogate name without the
regime suffix. Safeguard variants: --safeguards 'none={};cap=...json...' and --steer_shift for the
"steer everyone" safeguard (requires tables for the shifted dispositions).
"""
import argparse
import gzip
import itertools
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from e4_sweep_shared import HEAD, HKINDS, SYM, headlines_for  # noqa: E402

_SURR = None


def _surr():
    global _SURR
    if _SURR is None:
        from stampede.sim.surrogate import load_surrogates

        _SURR = load_surrogates("results/surrogates")
    return _SURR


def surrogate_fn(name, obs, vals, spec):
    s = _surr()[f"{name}_{spec.regime}"]
    bull = (spec.headline or {}).get("direction") == "bullish" if spec.headline else None
    return s.probs_for(obs, vals, bullish=bull)


def run_one(job):
    from stampede.sim.episode import EpisodeSpec, Group, run_episodes, save_trace, summary_row

    specs = []
    for kind in (job["hkind"], "none"):
        sp = EpisodeSpec(regime=job["regime"], seed=job["seed"], n_ticks=job["n_ticks"], shock_t=job["shock_t"],
                         correction_t=job["correction_t"] if kind == "fake" else None,
                         headline_kind=kind, headline=job["headline"] if kind != "none" else None,
                         source_verified=job["verified"], source_label=job["source"],
                         capital_ratio=job["capital"], safeguards=job["safeguards"], act_prob=job["act_prob"],
                         resilience=job.get("resilience") or None, depth_scale=job.get("depth_scale", 1.0),
                         model="surrogate", tag=job["tag"],
                         groups=[Group(name=f"{g['table']}", n=g["n"], surrogate=g["table"], init_exposure=1.0)
                                 for g in job["groups"]])
        specs.append(sp)
    trs = run_episodes(specs, decider=None, surrogate_fn=surrogate_fn)
    rows = []
    for tr in trs:
        row = summary_row(tr)
        row.update({k: job[k] for k in ("pop", "capital", "hcode", "safeguard_name", "tag")})
        row["resilience"] = job.get("resilience") or "medium"
        row["depth_scale"] = job.get("depth_scale", 1.0)
        rows.append(row)
        if job.get("save") and tr["config"]["headline_kind"] != "none":
            tr["config"].update(pop=job["pop"], hcode=job["hcode"], safeguard=job["safeguard_name"], adoption=job["capital"])
            save_trace(tr, "results/episodes")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pops", required=True)
    ap.add_argument("--capital", default="0.05,0.1,0.2,0.35,0.5,0.75,1,1.5")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--regimes", default="equity,crypto")
    ap.add_argument("--headlines", default="fake_unv,true_ver")
    ap.add_argument("--n_headlines", type=int, default=3)
    ap.add_argument("--severity", default="severe,moderate")
    ap.add_argument("--n_agents", type=int, default=24)
    ap.add_argument("--n_ticks", type=int, default=50)
    ap.add_argument("--shock_t", type=int, default=15)
    ap.add_argument("--correction_t", type=int, default=30)
    ap.add_argument("--act_prob", type=float, default=0.5)
    ap.add_argument("--safeguards", default="none={}")
    ap.add_argument("--save_seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--resilience", default="", help="comma list of resilience settings; empty = regime default")
    ap.add_argument("--depth_scale", default="1")
    args = ap.parse_args()

    pops = {}
    for part in args.pops.split(";"):
        if not part.strip():
            continue
        name, spec = part.split("=", 1)
        groups = []
        items = [x for x in spec.split(",") if x]
        fr = []
        for x in items:
            t, f = (x.split(":") + ["1"])[:2]
            fr.append((t, float(f)))
        tot = sum(f for _, f in fr)
        left = args.n_agents
        for i, (t, f) in enumerate(fr):
            n = left if i == len(fr) - 1 else int(round(args.n_agents * f / tot))
            left -= n
            if n > 0:
                groups.append({"table": t, "n": n})
        pops[name.strip()] = groups
    sgs = {}
    for part in args.safeguards.split(";"):
        if part.strip():
            k, v = part.split("=", 1)
            sgs[k.strip()] = json.loads(v)
    jobs = []
    for regime, hcode in itertools.product(args.regimes.split(","), args.headlines.split(",")):
        kind, verified, source, direction = HKINDS[hcode]
        hs = headlines_for(regime, direction, args.severity.split(","), args.n_headlines)
        res_list = [x for x in args.resilience.split(",") if x] or [""]
        ds_list = [float(x) for x in args.depth_scale.split(",")]
        for (pname, groups), cap, seed, (sgname, sg), res, ds in itertools.product(
                pops.items(), [float(c) for c in args.capital.split(",")], range(args.seeds), sgs.items(), res_list, ds_list):
            h = hs[seed % len(hs)]
            jobs.append({"regime": regime, "hkind": kind, "verified": verified, "source": source, "headline": h,
                         "hcode": hcode, "pop": pname, "groups": groups, "capital": cap, "seed": seed,
                         "safeguards": sg, "safeguard_name": sgname, "n_ticks": args.n_ticks, "shock_t": args.shock_t,
                         "correction_t": args.correction_t, "act_prob": args.act_prob, "tag": args.name,
                         "resilience": res, "depth_scale": ds,
                         "save": seed == args.save_seed})
    out_dir = f"results/e4/{args.name}"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for i, rs in enumerate(pool.imap_unordered(run_one, jobs, chunksize=4)):
            rows.extend(rs)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(jobs)} jobs in {time.time() - t0:.0f}s", flush=True)
    with gzip.open(f"{out_dir}/rows.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    print(f"done {len(jobs)} jobs, {len(rows)} episodes in {time.time() - t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
