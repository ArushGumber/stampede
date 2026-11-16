"""Response tables (surrogate policies) for steered open-model agents.

For each disposition (a dict of trait -> alpha) and regime, read the model's choice distribution on every
state of the grid, averaged over ascending and descending menus. Saves
results/surrogates/<model>_<name>_<regime>.json.

    python scripts/make_tables.py --model gemma-4-12b --layer 24 --size medium \
        --dispositions 'base={};reckless1={"risk":1}' --regimes equity,crypto
"""
import argparse
import json
import os
import time

import torch

from stampede.llm.runner import ModelRunner, TraitVector
from stampede.sim.surrogate import save_table, table_from_open_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemma-4-12b")
ap.add_argument("--layer", type=int, default=24)
ap.add_argument("--method", default="syslast")
ap.add_argument("--size", default="medium")
ap.add_argument("--regimes", default="equity,crypto")
ap.add_argument("--dispositions", required=True, help="name=json;name=json")
ap.add_argument("--persona", default="", help="optional name=instruction-text pairs separated by ';;'")
ap.add_argument("--persona_ref", default="", help="name=trait:pole:index into data/batteries/traits.json, separated by ';'")
args = ap.parse_args()

r = ModelRunner(args.model, "cuda:0")
import os as _os
_vp = f"results/e1/{args.model}/vectors/{args.method}.pt"
if _os.path.exists(_vp):
    vec = torch.load(_vp)
    for t, raw in vec.items():
        r.set_vector(TraitVector(t, args.layer, raw[args.layer].float(), args.method))
disp = {}
for part in args.dispositions.split(";"):
    if part.strip():
        k, v = part.split("=", 1)
        disp[k.strip()] = json.loads(v)
personas = {}
for part in args.persona.split(";;"):
    if part.strip():
        k, v = part.split("=", 1)
        personas[k.strip()] = v.strip()
if args.persona_ref:
    TR = json.load(open("data/batteries/traits.json"))["traits"]
    for part in args.persona_ref.split(";"):
        if part.strip():
            k, ref = part.split("=", 1)
            trait, pole, i = ref.split(":")
            personas[k.strip()] = TR[trait][pole]["instructions"][int(i)]
t0 = time.time()
for g in args.regimes.split(","):
    for name, al in disp.items():
        out = f"results/surrogates/{args.model}_{name}_{g}.json"
        if os.path.exists(out):
            print("exists", out, flush=True)
            continue
        tab = table_from_open_model(r, g, alphas=al, size=args.size)
        tab.update(model=args.model, layer=args.layer, method=args.method, disposition=name)
        save_table(tab, out)
        print(f"[{time.time() - t0:5.0f}s] {out} states={len(tab['states'])}", flush=True)
    for name, text in personas.items():
        out = f"results/surrogates/{args.model}_{name}_{g}.json"
        if os.path.exists(out):
            continue
        tab = table_from_open_model(r, g, alphas={}, size=args.size, persona=text)
        tab.update(model=args.model, persona=text, disposition=name)
        save_table(tab, out)
        print(f"[{time.time() - t0:5.0f}s] {out} (persona)", flush=True)
