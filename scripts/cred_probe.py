"""Search for a credulity direction that makes the agent discount unverified headlines in the market format.

For each extraction method and layer, steer toward scepticism (negative alpha) and measure, on the market
states, the reaction to an unverified post versus a verified wire story, plus capability. Saves vectors for
the CAA method alongside the syslast ones and writes results/e1/<model>/cred_probe.json.
"""
import json
import random
import sys
import time

import torch

from stampede.llm.runner import ModelRunner, TraitVector
from stampede.steering import extract as X
from stampede.steering.battery import Battery

name = sys.argv[1] if len(sys.argv) > 1 else "gemma-4-12b"
r = ModelRunner(name, "cuda:0")
T = json.load(open("data/batteries/traits.json"))["traits"]
EQ = json.load(open("data/batteries/extract_questions.json"))["items"]
CAA = json.load(open("data/batteries/caa_pairs.json"))["items"]
items = [it for it in CAA if it["trait"] == "credulity"]
random.Random(0).shuffle(items)
train = items[: (2 * len(items)) // 3]
cap = None
bat = Battery(r)
bat.cap = bat.cap[:30]
vec_dir = f"results/e1/{name}/vectors"
sys_vec = torch.load(f"{vec_dir}/syslast.pt")["credulity"]
caa_raw = X.caa_vectors(r, train)
blob = {}
try:
    blob = torch.load(f"{vec_dir}/caa.pt")
except Exception:
    pass
blob["credulity"] = caa_raw
torch.save(blob, f"{vec_dir}/caa.pt")
# market-format extraction: the same decision prompts (crypto regime, so evaluation on equity stays held out) under
# sceptical versus credulous instructions, activations at the decision token
from stampede.sim.surrogate import grid_states, state_prompts
ext_states = [st for st in grid_states("crypto", "small") if st["news"] != "none"][::3]
def mk(instrs):
    out = []
    for ins in instrs:
        for st in ext_states:
            system, user, _ = state_prompts(st, False)
            out.append(r.format(system.replace("Reply with only the letter of your choice.", ins + " Reply with only the letter of your choice."), user))
    return out
pos_i = T["credulity"]["positive"]["instructions"]
neg_i = T["credulity"]["negative"]["instructions"]
mkt_raw = r.capture_last(mk(pos_i)).mean(0) - r.capture_last(mk(neg_i)).mean(0)
blob["credulity_mkt"] = mkt_raw
torch.save(blob, f"{vec_dir}/caa.pt")
L = r.n_layers
layers = [int(L * f) for f in (0.33, 0.42, 0.5, 0.58, 0.67, 0.75)]
out = []
t0 = time.time()
for method, raw in (("mkt", mkt_raw), ("syslast", sys_vec), ("caa", caa_raw)):
    for l in layers:
        v = raw[l].float()
        # put both methods on a common scale: alpha is in units of the syslast mean difference at this layer
        scale = float(sys_vec[l].norm() / v.norm()) if method != "syslast" else 1.0
        r.set_vector(TraitVector("credulity", l, v * scale, method))
        for a in (-1.0, -2.0, -3.0):
            res = bat.run({"credulity": a}, parts=("market", "capability"))
            row = {"method": method, "layer": l, "alpha": a, **{k: float(res[k]) for k in
                   ("react_unverified", "react_verified", "credulity_gap", "credulity", "mean_exposure", "capability")}}
            out.append(row)
            print(f"[{time.time() - t0:5.0f}s] {method:7s} L{l:02d} a={a:+.0f} unv={row['react_unverified']:+.3f} "
                  f"ver={row['react_verified']:+.3f} gap={row['credulity_gap']:+.3f} expo={row['mean_exposure']:.2f} "
                  f"cap={row['capability']:.2f}", flush=True)
            json.dump(out, open(f"results/e1/{name}/cred_probe.json", "w"), indent=1)
