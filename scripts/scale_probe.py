"""How big can a steering vector be before the model breaks? Probe raw mean-difference multiples on a few layers."""
import json, sys, torch, numpy as np
from stampede.llm.runner import ModelRunner, TraitVector
from stampede.steering import extract as X
from stampede.steering.battery import Battery
name = sys.argv[1]
T = json.load(open("data/batteries/traits.json"))["traits"]; EQ = json.load(open("data/batteries/extract_questions.json"))["items"]
r = ModelRunner(name, "cuda:0"); bat = Battery(r)
qs = [q["question"] for q in EQ if q["trait"] == "risk" and q.get("split") == "extract"][:24]
neutral = [r.format(X.NEUTRAL_SYSTEM, q["question"]) for q in EQ[:60]]
raw = X.syslast_vectors(r, T["risk"]["positive"]["instructions"], T["risk"]["negative"]["instructions"], qs)
norms = X.resid_norms(r, neutral); sig = X.projection_sigma(r, neutral, raw)
print("layer | resid_norm | diff_norm | sigma | diff/sigma")
for l in range(0, r.n_layers, max(1, r.n_layers // 12)):
    print(f"{l:3d} | {norms[l]:9.1f} | {raw[l].norm():8.2f} | {sig[l]:7.2f} | {raw[l].norm()/sig[l]:6.2f}")
torch.save({"raw": raw, "norms": norms, "sigma": sig}, f"results/scale_probe_{name}.pt")
for l in [int(r.n_layers * f) for f in (0.25, 0.4, 0.5, 0.6)]:
    r.set_vector(TraitVector("risk", l, raw[l], "syslast"))
    for m in (-2, -1, 1, 2, 4):
        res = bat.run({"risk": m}, parts=("hl", "market", "capability"))
        print(f"L{l:02d} x{m:+d}: safe={res['safe_choices']:.2f} gamma={res['crra_gamma']:+.2f} mono={res['monotonicity']:.2f} expo={res['mean_exposure']:.2f} lev={res['leverage_rate']:.2f} cap={res['capability']:.2f} mass={res['option_mass']:.2f}", flush=True)
