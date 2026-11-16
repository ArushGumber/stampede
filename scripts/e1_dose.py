"""E1 stage 2: dose-response of one trait direction at one layer, with controls.

Series (contract section 6, dose_response):
  steer    the trait direction at alpha in --alphas (alpha = multiples of the raw mean difference)
  random   matched-norm random directions (3 seeds), same alphas
  unrelated  a formality direction (non-financial), matched norm, same alphas
  prompt   prompt personas: the trait's positive and negative instructions (plotted at alpha = +1 and -1)

Writes results/e1/<model>/dose_<trait>_<method>_L<layer>.json
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from stampede.llm.runner import ModelRunner, TraitVector
from stampede.steering import extract as X
from stampede.steering.battery import Battery

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemma-4-12b")
ap.add_argument("--trait", required=True)
ap.add_argument("--method", default="syslast")
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--alphas", default="-3,-2,-1,-0.5,0,0.5,1,2,3")
ap.add_argument("--controls", default="random,unrelated,prompt")
ap.add_argument("--parts", default="hl,kelly,market,capability,integrity,ab")
ap.add_argument("--vectors", default="")
ap.add_argument("--regimes", default="equity")
ap.add_argument("--stakes", default="1,100,10000")
ap.add_argument("--random_alphas", default="-2,-1,1,2")
ap.add_argument("--random_seeds", type=int, default=2)
ap.add_argument("--unrelated_alphas", default="-1,1,2")
ap.add_argument("--tag", default="")
args = ap.parse_args()

out_dir = f"results/e1/{args.model}"
os.makedirs(out_dir, exist_ok=True)
T = json.load(open("data/batteries/traits.json"))["traits"]
EQ = json.load(open("data/batteries/extract_questions.json"))["items"]
CAA = json.load(open("data/batteries/caa_pairs.json"))["items"]
C = json.load(open("data/controls/controls.json"))

r = ModelRunner(args.model, "cuda:0")
heldout = {}
train = {}
for t in ("risk", "credulity", "integrity"):
    items = [it for it in CAA if it["trait"] == t]
    random.Random(0).shuffle(items)
    k = (2 * len(items)) // 3
    train[t], heldout[t] = items[:k], items[k:]
bat = Battery(r, regimes=tuple(args.regimes.split(",")), ab_heldout=heldout,
              stakes=tuple(float(x) for x in args.stakes.split(",")))
parts = tuple(args.parts.split(","))


def instr(t, pole):
    p = T[t][pole]
    return p["instructions"] if isinstance(p, dict) else p


vec_path = args.vectors or f"{out_dir}/vectors/{args.method}.pt"
if os.path.exists(vec_path) and args.trait in torch.load(vec_path):
    raw = torch.load(vec_path)[args.trait]
else:
    if args.method == "syslast":
        qs = [q["question"] for q in EQ if q["trait"] == args.trait and q.get("split") == "extract"]
        raw = X.syslast_vectors(r, instr(args.trait, "positive"), instr(args.trait, "negative"), qs[:24])
    elif args.method == "caa":
        raw = X.caa_vectors(r, train[args.trait])
    elif args.method == "nonfin":
        ins = C["nonfinancial_instructions"][args.trait]
        raw = X.syslast_vectors(r, ins["positive"], ins["negative"], C["nonfinancial_questions"][args.trait])
    else:
        raise SystemExit("unknown method")
    os.makedirs(os.path.dirname(vec_path), exist_ok=True)
    blob = torch.load(vec_path) if os.path.exists(vec_path) else {}
    blob[args.trait] = raw
    torch.save(blob, vec_path)
v = raw[args.layer].float()
neutral = [r.format(X.NEUTRAL_SYSTEM, q["question"]) for q in EQ[:60]]
sigma = float(X.projection_sigma(r, neutral, raw)[args.layer])

alphas = [float(a) for a in args.alphas.split(",")]
records = []
t0 = time.time()


def run(kind, alpha, steer=None, persona=None, extra=None):
    if steer is not None:
        r.set_vector(TraitVector("x", args.layer, steer, kind))
        res = bat.run({"x": alpha} if alpha else None, parts=parts)
    else:
        res = bat.run(None, persona=persona, parts=parts)
    rec = {"kind": kind, "alpha": alpha, **(extra or {}),
           **{k: float(val) for k, val in res.items() if isinstance(val, (float, int, np.floating))}}
    rec["p_risky_by_row"] = [float(x) for x in res.get("p_risky_by_row", [])]
    for k in ("p_risky", "stake", "exposure", "unethical"):
        if k in res:
            rec["arr_" + k] = [round(float(x), 5) for x in np.asarray(res[k]).ravel()]
    records.append(rec)
    print(f"[{time.time() - t0:5.0f}s] {kind:9s} a={alpha:+.2f} {extra or ''} safe={rec.get('safe_choices', float('nan')):.2f} "
          f"gamma={rec.get('crra_gamma', float('nan')):+.2f} kelly={rec.get('kelly_fraction', float('nan')):.2f} "
          f"expo={rec.get('mean_exposure', float('nan')):.2f} lev={rec.get('leverage_rate', float('nan')):.2f} "
          f"unv={rec.get('react_unverified', float('nan')):+.3f} ver={rec.get('react_verified', float('nan')):+.3f} "
          f"uneth={rec.get('unethical_rate', float('nan')):.2f} ab={rec.get('ab_' + args.trait, float('nan')):.2f} "
          f"cap={rec.get('capability', float('nan')):.2f}", flush=True)
    meta = {"model": args.model, "trait": args.trait, "method": args.method, "layer": args.layer,
            "vector_norm": float(v.norm()), "sigma": sigma, "alpha_unit": "one raw mean difference"}
    json.dump({"meta": meta, "records": records},
              open(f"{out_dir}/dose_{args.trait}_{args.method}_L{args.layer}{args.tag}.json", "w"), indent=1)


for a in alphas:
    run("steer", a, steer=v)
ctrl = args.controls.split(",") if args.controls else []
if "random" in ctrl:
    for seed in range(1, args.random_seeds + 1):
        rv = X.random_like(v.cpu(), seed)
        for a in [float(x) for x in args.random_alphas.split(",")]:
            run("random", a, steer=rv, extra={"seed": seed})
if "unrelated" in ctrl:
    u = C["unrelated"]
    raw_u = X.syslast_vectors(r, u["positive"], u["negative"], u["questions"])[args.layer].float()
    uv = raw_u / raw_u.norm() * v.norm()
    for a in [float(x) for x in args.unrelated_alphas.split(",")]:
        run("unrelated", a, steer=uv)
if "prompt" in ctrl:
    for pole, a in (("negative", -1.0), ("positive", 1.0)):
        for i, ins in enumerate(instr(args.trait, pole)):
            run("prompt", a, persona=ins, extra={"instruction": i})
print("done", round(time.time() - t0), "s", flush=True)
