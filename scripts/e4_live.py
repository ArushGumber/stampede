"""Live market episodes with steered open-model agents on the GPU (no surrogates).

Uses: (a) validating response-table surrogates against the real policy in the market, (b) early-warning
analysis (E7) from activations read at a monitor layer upstream of the steering layer, (c) showcase
episodes for the site.

    python scripts/e4_live.py --name live_val --model gemma-4-12b --layer 24 --monitor_layer 18 \
        --pops 'base={};risk+1={"risk":1}' --capital 0.5,2 --seeds 3 --regimes equity,crypto --headlines fake_unv,true_ver
"""
import argparse
import gzip
import itertools
import json
import os
import time

import torch

from stampede.llm.runner import ModelRunner, TraitVector
from stampede.sim.episode import ChoiceCache, Decider, EpisodeSpec, Group, run_episodes, save_trace, summary_row

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--model", default="gemma-4-12b")
ap.add_argument("--layer", type=int, default=24)
ap.add_argument("--monitor_layer", type=int, default=18)
ap.add_argument("--method", default="syslast")
ap.add_argument("--pops", required=True, help="name=json-alphas[@n] separated by ';'. A mixture: name=json|json")
ap.add_argument("--capital", default="0.5,2")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--regimes", default="equity")
ap.add_argument("--headlines", default="fake_unv")
ap.add_argument("--n_agents", type=int, default=24)
ap.add_argument("--n_ticks", type=int, default=50)
ap.add_argument("--shock_t", type=int, default=15)
ap.add_argument("--correction_t", type=int, default=30)
ap.add_argument("--batch_eps", type=int, default=16)
ap.add_argument("--twins", action="store_true")
args = ap.parse_args()

from e4_sweep_shared import HKINDS, headlines_for  # noqa: E402

r = ModelRunner(args.model, "cuda:0")
vec = torch.load(f"results/e1/{args.model}/vectors/{args.method}.pt")
for t, raw in vec.items():
    r.set_vector(TraitVector(t, args.layer, raw[args.layer].float(), args.method))
    r.set_vector(TraitVector("mon_" + t, args.monitor_layer, raw[args.monitor_layer].float(), args.method))
os.makedirs("results/cache", exist_ok=True)
cache = ChoiceCache(f"results/cache/{args.model}_L{args.layer}_choices.pkl")
dec = Decider({args.model: r}, cache, log_proj=True, bs=64)

pops = {}
for part in args.pops.split(";"):
    if part.strip():
        name, spec = part.split("=", 1)
        mix = [json.loads(x) for x in spec.split("|")]
        pops[name.strip()] = mix

jobs = []
for regime, hcode in itertools.product(args.regimes.split(","), args.headlines.split(",")):
    kind, verified, source, direction = HKINDS[hcode]
    hs = headlines_for(regime, direction, ["severe", "moderate"], 3)
    for (pname, mix), cap, seed in itertools.product(pops.items(), [float(c) for c in args.capital.split(",")], range(args.seeds)):
        n_each = [args.n_agents // len(mix)] * len(mix)
        n_each[-1] += args.n_agents - sum(n_each)
        groups = [Group(name=f"{pname}#{i}", n=n, alphas=a, init_exposure=1.0) for i, (a, n) in enumerate(zip(mix, n_each))]
        for k in ([kind, "none"] if args.twins else [kind]):
            jobs.append((EpisodeSpec(regime=regime, seed=seed, n_ticks=args.n_ticks, shock_t=args.shock_t,
                                     correction_t=args.correction_t if k == "fake" else None, headline_kind=k,
                                     headline=hs[seed % len(hs)] if k != "none" else None, source_verified=verified,
                                     source_label=source, capital_ratio=cap, groups=groups, model=args.model,
                                     tag=args.name), {"pop": pname, "capital": cap, "hcode": hcode, "safeguard_name": "none"}))

out_dir = f"results/e4/{args.name}"
os.makedirs(out_dir, exist_ok=True)
t0 = time.time()
rows = []
for i in range(0, len(jobs), args.batch_eps):
    chunk = jobs[i:i + args.batch_eps]
    trs = run_episodes([s for s, _ in chunk], dec)
    for tr, (_, meta) in zip(trs, chunk):
        tr["config"].update(pop=meta["pop"], hcode=meta["hcode"], safeguard="none", adoption=meta["capital"], live=True)
        save_trace(tr, "results/episodes_live")
        row = summary_row(tr)
        row.update(meta)
        row["tag"] = args.name
        rows.append(row)
    cache.save()
    print(f"[{time.time() - t0:5.0f}s] {i + len(chunk)}/{len(jobs)} episodes; cache hits {cache.hits} misses {cache.misses}", flush=True)
    with gzip.open(f"{out_dir}/rows.jsonl.gz", "wt") as f:
        for row in rows:
            f.write(json.dumps(row, default=float) + "\n")
print("done", round(time.time() - t0), "s", flush=True)
