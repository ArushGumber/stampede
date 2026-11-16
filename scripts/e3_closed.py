"""E3: run the same behavioural batteries on closed models through their command-line agents.

Usage: python scripts/e3_closed.py --agent claude-haiku [--parts hl,kelly,market,capability,integrity,ab]
                                    [--repeats 2] [--table] [--workers 8]
Writes results/e3/<agent>/battery.json and, with --table, results/surrogates/<agent>_<regime>.json.
"""
import argparse
import datetime
import json
import os
import random
import time

import numpy as np

from stampede.agents.closed_runner import ClosedRunner
from stampede.sim.surrogate import save_table, table_from_open_model
from stampede.steering.battery import Battery

ap = argparse.ArgumentParser()
ap.add_argument("--agent", required=True)
ap.add_argument("--parts", default="hl,kelly,market,capability,integrity,ab")
ap.add_argument("--repeats", type=int, default=2)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--batch", type=int, default=12)
ap.add_argument("--table", action="store_true")
ap.add_argument("--table_size", default="small")
ap.add_argument("--regimes", default="equity")
ap.add_argument("--effort", default="medium")
ap.add_argument("--stakes", default="1,100,10000")
ap.add_argument("--battery_regimes", default="equity")
args = ap.parse_args()

out_dir = f"results/e3/{args.agent}"
os.makedirs(out_dir, exist_ok=True)
kw = {"effort": args.effort} if args.agent.startswith("gpt") else {}
cr = ClosedRunner(args.agent, f"{out_dir}/answers.jsonl", repeats=args.repeats, batch=args.batch,
                  workers=args.workers, **kw)

CAA = json.load(open("data/batteries/caa_pairs.json"))["items"]
heldout = {}
for t in ("risk", "credulity", "integrity"):
    items = [it for it in CAA if it["trait"] == t]
    random.Random(0).shuffle(items)
    heldout[t] = items[(2 * len(items)) // 3:]
integ = None
p = "data/batteries/integrity_vignettes.json"
if os.path.exists(p):
    raw = json.load(open(p))
    its = raw["items"] if isinstance(raw, dict) else raw
    integ = []
    for it in its:
        opts = it.get("options")
        if isinstance(opts, list) and opts and isinstance(opts[0], dict):
            texts = [o.get("text") or o.get("option") for o in opts]
            bad = [i for i, o in enumerate(opts) if o.get("unethical")]
        else:
            texts = opts
            bad = it.get("unethical") or it.get("unethical_options") or []
            bad = [("ABCD".index(b) if isinstance(b, str) else int(b)) for b in bad]
        q = it.get("question") or it.get("scenario") or it.get("prompt")
        if texts and q and bad:
            integ.append({"id": it.get("id"), "question": q, "options": texts, "unethical": bad})

regimes = args.regimes.split(",")
bat = Battery(cr, regimes=tuple(args.battery_regimes.split(",")), integrity=integ, ab_heldout=heldout,
              stakes=tuple(float(x) for x in args.stakes.split(",")))
t0 = time.time()
res = bat.run(parts=tuple(args.parts.split(",")))
summary = {k: float(v) for k, v in res.items() if isinstance(v, (float, int, np.floating))}
summary["p_risky_by_row"] = [float(x) for x in res.get("p_risky_by_row", [])]
for k in ("p_risky", "stake", "exposure", "unethical"):
    if k in res:
        summary["arr_" + k] = [round(float(x), 5) for x in np.asarray(res[k]).ravel()]
meta = {"agent": args.agent, "harness": "claude -p" if args.agent.startswith("claude") else "codex exec",
        "date": datetime.date.today().isoformat(), "repeats": args.repeats, "batch": args.batch,
        "calls": cr.calls, "failed_calls": cr.failures, "seconds": round(time.time() - t0)}
json.dump({"meta": meta, "summary": summary}, open(f"{out_dir}/battery.json", "w"), indent=1)
print(json.dumps(meta), flush=True)
print(json.dumps({k: round(v, 3) for k, v in summary.items() if not isinstance(v, list)}), flush=True)
if args.table:
    for g in regimes:
        tab = table_from_open_model(cr, g, alphas={}, size=args.table_size)
        tab.update(source="samples", agent=args.agent, harness=meta["harness"], date=meta["date"])
        save_table(tab, f"results/surrogates/{args.agent}_{g}.json")
        print("table", g, len(tab["states"]), "states; calls so far", cr.calls, flush=True)
