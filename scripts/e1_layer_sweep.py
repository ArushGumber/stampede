"""E1 stage 1: extract trait directions by several methods and pick a layer per (trait, method).

For each trait and extraction method, steer at +/-alpha on a set of layers and record the trait's target
metrics plus capability. Writes results/e1/<model>/layer_sweep.json and vectors/<method>.pt.

Methods
  caa       A/B items from the trait battery (answer-letter activations)
  syslast   finance-framed system instructions x finance questions (last prompt token)
  nonfin    non-financial instructions x non-financial questions (last prompt token)
Controls (computed at the chosen layer in stage 2): matched-norm random directions, an unrelated trait (formality).
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
ap.add_argument("--layers", default="")
ap.add_argument("--alpha", type=float, default=0.2)
ap.add_argument("--traits", default="risk,credulity,integrity")
ap.add_argument("--methods", default="caa,syslast,nonfin")
args = ap.parse_args()

T = json.load(open("data/batteries/traits.json"))["traits"]
EQ = json.load(open("data/batteries/extract_questions.json"))["items"]
CAA = json.load(open("data/batteries/caa_pairs.json"))["items"]
C = json.load(open("data/controls/controls.json"))
out_dir = f"results/e1/{args.model}"
os.makedirs(out_dir + "/vectors", exist_ok=True)

r = ModelRunner(args.model, "cuda:0")
L = r.n_layers
layers = [int(x) for x in args.layers.split(",")] if args.layers else sorted({int(L * f) for f in (0.25, 0.33, 0.42, 0.5, 0.58, 0.67)})
traits = args.traits.split(",")

# split CAA items: 2/3 for extraction, 1/3 held out for evaluation
caa_split = {}
for t in traits:
    items = [it for it in CAA if it["trait"] == t]
    random.Random(0).shuffle(items)
    k = (2 * len(items)) // 3
    caa_split[t] = (items[:k], items[k:])

bat = Battery(r, ab_heldout={t: caa_split[t][1] for t in traits})
norm_prompts = [r.format(X.NEUTRAL_SYSTEM, q["question"]) for q in EQ[:60]]
norms = X.resid_norms(r, norm_prompts)
torch.save(norms, f"{out_dir}/vectors/resid_norms.pt")


def instr(t, pole):
    p = T[t][pole]
    return p["instructions"] if isinstance(p, dict) else p


def build(method, t):
    if method == "caa":
        raw = X.caa_vectors(r, caa_split[t][0])
    elif method == "syslast":
        qs = [q["question"] for q in EQ if q["trait"] == t and q.get("split") == "extract"]
        raw = X.syslast_vectors(r, instr(t, "positive"), instr(t, "negative"), qs[:24])
    elif method == "nonfin":
        ins = C["nonfinancial_instructions"][t]
        raw = X.syslast_vectors(r, ins["positive"], ins["negative"], C["nonfinancial_questions"][t])
    else:
        raise ValueError(method)
    return X.to_unit_scaled(raw, norms)


PARTS = {"risk": ("hl", "kelly", "market", "capability", "ab"),
         "credulity": ("market", "capability", "ab"),
         "integrity": ("capability", "ab", "integrity")}
KEYS = {"risk": ["safe_choices", "crra_gamma", "kelly_fraction", "mean_exposure", "leverage_rate", "monotonicity"],
        "credulity": ["react_unverified", "react_verified", "credulity", "credulity_gap", "mean_exposure"],
        "integrity": ["unethical_rate", "mean_exposure"]}

t0 = time.time()
results = {"model": args.model, "n_layers": L, "layers": layers, "alpha": args.alpha, "rows": [], "baseline": {}}
base = bat.run(parts=("hl", "kelly", "market", "capability", "ab"))
results["baseline"] = {k: float(v) for k, v in base.items() if isinstance(v, (float, int))}
print("baseline", {k: round(v, 3) for k, v in results["baseline"].items()}, flush=True)
for method in args.methods.split(","):
    vecs = {}
    for t in traits:
        try:
            vecs[t] = build(method, t)
        except Exception as e:
            print("skip", method, t, repr(e)[:200], flush=True)
            continue
        # cosine similarity between layers' directions across traits, for leakage diagnostics later
    torch.save({t: v for t, v in vecs.items()}, f"{out_dir}/vectors/{method}.pt")
    for t, vec in vecs.items():
        for l in layers:
            r.set_vector(TraitVector(t, l, vec[l], method))
            row = {"method": method, "trait": t, "layer": l}
            for a in (-args.alpha, args.alpha):
                res = bat.run({t: a}, parts=PARTS[t])
                for k, v in res.items():
                    if isinstance(v, (float, int)):
                        row[f"{k}@{a:+.2f}"] = float(v)
            results["rows"].append(row)
            show = {k: round(row.get(f"{k}@{a:+.2f}", float("nan")), 3) for k in KEYS[t] + ["capability", f"ab_{t}"] for a in (args.alpha,)}
            lo = {k: round(row.get(f"{k}@{-args.alpha:+.2f}", float("nan")), 3) for k in KEYS[t] + ["capability", f"ab_{t}"]}
            print(f"[{time.time()-t0:5.0f}s] {method:7s} {t:9s} L{l:02d}  -: {lo}  +: {show}", flush=True)
            json.dump(results, open(f"{out_dir}/layer_sweep.json", "w"), indent=1)
# cross-trait cosine at every layer, per method
cos = {}
for method in args.methods.split(","):
    p = f"{out_dir}/vectors/{method}.pt"
    if not os.path.exists(p):
        continue
    v = torch.load(p)
    ts = list(v)
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            a, b = v[ts[i]], v[ts[j]]
            cos[f"{method}:{ts[i]}~{ts[j]}"] = torch.nn.functional.cosine_similarity(a, b, dim=-1).tolist()
for t in traits:
    ms = [m for m in args.methods.split(",") if os.path.exists(f"{out_dir}/vectors/{m}.pt")]
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            a, b = torch.load(f"{out_dir}/vectors/{ms[i]}.pt").get(t), torch.load(f"{out_dir}/vectors/{ms[j]}.pt").get(t)
            if a is not None and b is not None:
                cos[f"{t}:{ms[i]}~{ms[j]}"] = torch.nn.functional.cosine_similarity(a, b, dim=-1).tolist()
results["cosine"] = cos
json.dump(results, open(f"{out_dir}/layer_sweep.json", "w"), indent=1)
print("done", round(time.time() - t0), "s", flush=True)
