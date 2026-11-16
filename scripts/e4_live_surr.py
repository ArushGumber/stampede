"""Surrogate twin of e4_live.py: the same episode configurations (same seeds, headlines, group sizes and agent order,
hence the same per-agent random streams), with each live steered agent replaced by its response table.
Traces go to results/episodes_live_surr/, so live and surrogate runs can be compared tick by tick.

    python scripts/e4_live_surr.py --pops 'base=gemma-4-12b_base;risk+1=gemma-4-12b_risk+1;mix=gemma-4-12b_base|gemma-4-12b_risk+1.5' \
        --capital 1,2 --seeds 3 --regimes equity,crypto --headlines fake_unv --twins
"""
import argparse
import itertools
import sys

sys.path.insert(0, "scripts")
from e4_sweep import surrogate_fn  # noqa: E402
from e4_sweep_shared import HKINDS, headlines_for  # noqa: E402
from stampede.sim.episode import EpisodeSpec, Group, run_episodes, save_trace  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pops", required=True)
ap.add_argument("--capital", default="1,2")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--regimes", default="equity")
ap.add_argument("--headlines", default="fake_unv")
ap.add_argument("--n_agents", type=int, default=24)
ap.add_argument("--n_ticks", type=int, default=50)
ap.add_argument("--shock_t", type=int, default=15)
ap.add_argument("--correction_t", type=int, default=30)
ap.add_argument("--twins", action="store_true")
args = ap.parse_args()

pops = {}
for part in args.pops.split(";"):
    if part.strip():
        name, spec = part.split("=", 1)
        pops[name.strip()] = spec.split("|")
specs = []
for regime, hcode in itertools.product(args.regimes.split(","), args.headlines.split(",")):
    kind, verified, source, direction = HKINDS[hcode]
    hs = headlines_for(regime, direction, ["severe", "moderate"], 3)
    for (pname, tables), cap, seed in itertools.product(pops.items(), [float(c) for c in args.capital.split(",")], range(args.seeds)):
        n_each = [args.n_agents // len(tables)] * len(tables)
        n_each[-1] += args.n_agents - sum(n_each)
        groups = [Group(name=f"{pname}#{i}", n=n, surrogate=t, init_exposure=1.0) for i, (t, n) in enumerate(zip(tables, n_each))]
        for k in ([kind, "none"] if args.twins else [kind]):
            specs.append(EpisodeSpec(regime=regime, seed=seed, n_ticks=args.n_ticks, shock_t=args.shock_t,
                                     correction_t=args.correction_t if k == "fake" else None, headline_kind=k,
                                     headline=hs[seed % len(hs)] if k != "none" else None, source_verified=verified,
                                     source_label=source, capital_ratio=cap, groups=groups, model="surrogate",
                                     tag="live_val_surr"))
trs = run_episodes(specs, decider=None, surrogate_fn=surrogate_fn)
for tr, sp in zip(trs, specs):
    tr["config"].update(pop=sp.groups[0].name.split("#")[0], adoption=sp.capital_ratio)
    save_trace(tr, "results/episodes_live_surr")
print("saved", len(trs))
