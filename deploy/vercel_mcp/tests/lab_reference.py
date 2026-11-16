"""Runs a scripted session on the lab arena code in this process and prints the report card as JSON.

Run from the repository root so that `stampede` resolves to the lab package:
    cd <repo> && python deploy/vercel_mcp/tests/lab_reference.py <scenario> <seed> <label> <script.json>

The script is a list with one entry per minute; each entry is a list of actions submitted before the
clock advances: ["x", 0.5] | ["o", side, unit, amount, order_type, limit_price] | ["c", order_id].
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from stampede.mcp_server.arena import REGISTRY  # noqa: E402

scenario, seed, label, path = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
script = json.load(open(path))
s = REGISTRY.create(scenario, label, seed)
assert os.path.abspath(sys.modules["stampede"].__file__).startswith(os.getcwd())
for minute in script:
    for a in minute:
        if a[0] == "x":
            s.set_target_exposure(a[1])
        elif a[0] == "o":
            _, side, unit, amount, kind, price = a
            s.place_order(side, amount if unit == "q" else None, amount if unit == "n" else None, kind, price)
        elif a[0] == "c":
            s.cancel_order(a[1])
    s.advance()
print(json.dumps({"report": s.report(), "portfolio": s.portfolio(), "market": s.market()}))
