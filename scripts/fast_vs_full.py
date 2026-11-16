"""Does the fast readout (choice = most likely next token among the option letters) agree with what the model
writes when allowed to answer freely? For market states and lotteries, at several dial settings, generate up to
40 tokens with a free-form instruction, parse the first option letter, and compare with the logit argmax.
Writes results/e1/<model>/fast_vs_full.json.
"""
import json
import re
import sys

import numpy as np
import torch

from stampede.agents.prompting import LETTERS
from stampede.llm.runner import ModelRunner, TraitVector
from stampede.sim.surrogate import state_prompts
from stampede.steering.battery import hl_items, hl_text, market_states
from stampede.steering.extract import LETTER_RULE, NEUTRAL_SYSTEM

name = sys.argv[1] if len(sys.argv) > 1 else "gemma-4-12b"
layer = int(sys.argv[2]) if len(sys.argv) > 2 else 24
r = ModelRunner(name, "cuda:0")
vec = torch.load(f"results/e1/{name}/vectors/syslast.pt")
r.set_vector(TraitVector("risk", layer, vec["risk"][layer].float(), "syslast"))
FREE = "Think briefly if you like, then give your final choice as 'Final answer: <letter>'."
items = []
for st in market_states("equity"):
    system, user, vals = state_prompts(st, False)
    items.append(("market", system, user, list(LETTERS[:len(vals)])))
for it in hl_items((1, 100, 10000)):
    items.append(("lottery", NEUTRAL_SYSTEM + " " + LETTER_RULE, hl_text(it, False), ["A", "B"]))
out = []
for alpha in (0.0, 1.0):
    al = [{"risk": alpha}] * len(items) if alpha else None
    fast = r.choose([r.format(s, u) for _, s, u, _ in items], [o for *_, o in items], alphas=al)
    free_prompts = [r.format(s.replace(LETTER_RULE, FREE), u.replace("\n\nAnswer:", "")) for _, s, u, _ in items]
    texts = r.generate(free_prompts, alphas=al, max_new_tokens=160)
    samples = texts[:6]
    agree, parsed = 0, 0
    for (kind, _, _, opts), p, t in zip(items, fast, texts):
        m = (re.search(r"[Ff]inal [Aa]nswer\W*\(?\**([A-J])\b", t) or re.search(r"\*\*([A-J])[.)\*]", t)
             or re.search(r"(?:[Oo]ption|[Cc]hoice|choose|select)\s*\(?([A-J])\b", t) or re.search(r"^\s*\(?([A-J])[.):]", t))
        if m and m.group(1) in opts:
            parsed += 1
            agree += int(opts.index(m.group(1)) == int(np.argmax(p[:len(opts)])))
    out.append({"alpha": alpha, "n": len(items), "parsed": parsed, "agree": agree,
                "agreement": agree / max(parsed, 1), "samples": samples})
    print(out[-1], flush=True)
json.dump(out, open(f"results/e1/{name}/fast_vs_full.json", "w"), indent=1)
