"""Quick pilot: extract a risk and a credulity direction from non-financial text and see whether they move financial choices."""
import json, sys, time, torch, numpy as np
from stampede.llm.runner import ModelRunner, TraitVector
from stampede.steering import extract as X
from stampede.steering.battery import Battery
name, dev = sys.argv[1], sys.argv[2]
C = json.load(open("data/controls/controls.json"))
r = ModelRunner(name, dev)
bat = Battery(r)
t0 = time.time(); base = bat.run(); print(f"baseline in {time.time()-t0:.1f}s:", {k: round(v, 3) for k, v in base.items() if isinstance(v, float)})
print("P(risky) by row:", np.round(base["p_risky_by_row"], 2))
norms = X.resid_norms(r, [r.format(X.NEUTRAL_SYSTEM, q) for q in C["nonfinancial_questions"]["risk"]])
for trait in ("risk", "credulity"):
    ins, qs = C["nonfinancial_instructions"][trait], C["nonfinancial_questions"][trait]
    raw = X.syslast_vectors(r, ins["positive"], ins["negative"], qs)
    vec = X.to_unit_scaled(raw, norms)
    print(f"== {trait}: syslast, non-financial extraction ==")
    for L in (8, 12, 16, 20):
        r.set_vector(TraitVector(trait, L, vec[L], "syslast-nonfin"))
        row = []
        for a in (-0.2, 0.2):
            res = bat.run({trait: a}, parts=("hl", "market", "capability"))
            row.append(f"a={a:+.1f}: safe={res['safe_choices']:.2f} gamma={res['crra_gamma']:+.2f} expo={res['mean_exposure']:.2f} cred={res['credulity']:.2f} unv={res['react_unverified']:+.3f} ver={res['react_verified']:+.3f} cap={res['capability']:.2f}")
        print(f"layer {L:2d} | " + " | ".join(row))
print("total", round(time.time() - t0), "s")
