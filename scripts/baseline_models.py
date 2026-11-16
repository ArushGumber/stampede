"""Unsteered baseline of each open model on the built-in batteries, to choose the main model."""
import json, sys, time, gc, torch, numpy as np
from stampede.llm.runner import ModelRunner
from stampede.steering.battery import Battery
import os
out = json.load(open("results/baseline_models.json")) if os.path.exists("results/baseline_models.json") else {}
for name in sys.argv[1:]:
    t0 = time.time(); r = ModelRunner(name, "cuda:0"); bat = Battery(r)
    res = bat.run(parts=("hl", "kelly", "market", "capability", "integrity"))
    sc = {k: round(float(v), 3) for k, v in res.items() if isinstance(v, (float, int))}
    sc["p_risky_by_row"] = [round(float(x), 2) for x in res["p_risky_by_row"]]
    out[name] = sc
    print(name, f"{time.time()-t0:.0f}s", json.dumps(sc)); sys.stdout.flush()
    del bat, r; gc.collect(); torch.cuda.empty_cache()
json.dump(out, open("results/baseline_models.json", "w"), indent=1)
