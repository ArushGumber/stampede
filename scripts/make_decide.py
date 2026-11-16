"""Precompute the site's "try the dial" panel: fixed vignettes x dial settings -> choice probabilities and a
short stated reason. Nothing is generated live on the public site.

Writes results/site_decide.json:
  {default_model, vignettes: {id: {title, text, options}}, records: [{vignette_id, model, alphas, probs, reason}]}
"""
import argparse
import json
import re

import torch

from stampede.agents.prompting import LETTERS
from stampede.llm.runner import ModelRunner, TraitVector
from stampede.sim.surrogate import state_prompts
from stampede.steering.battery import hl_items, hl_text, kelly_items, kelly_text, load_integrity
from stampede.steering.extract import LETTER_RULE, NEUTRAL_SYSTEM

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemma-4-12b")
ap.add_argument("--layer", type=int, default=24)
ap.add_argument("--alphas", default="-2,-1,0,0.5,1,1.5,2")
args = ap.parse_args()

r = ModelRunner(args.model, "cuda:0")
vec = torch.load(f"results/e1/{args.model}/vectors/syslast.pt")
for t, raw in vec.items():
    r.set_vector(TraitVector(t, args.layer, raw[args.layer].float(), "syslast"))

vig = {}
hl = [it for it in hl_items((100,)) if it["row"] == 4][0]
vig["holt_laury"] = {"title": "Two payouts for the client", "system": NEUTRAL_SYSTEM + " " + LETTER_RULE, "user": hl_text(hl, False),
                     "options": ["Safer payout", "Riskier payout"], "letters": ["A", "B"], "trait": "risk"}
k = [it for it in kelly_items() if abs(it["p"] - 0.6) < 1e-9 and abs(it["odds"] - 1.0) < 1e-9][0]
kt, kv = kelly_text(k, False)
vig["kelly"] = {"title": "How much to stake on a 60% even-money bet", "system": NEUTRAL_SYSTEM + " " + LETTER_RULE, "user": kt,
                "options": [f"{v * 100:.0f}%" for v in kv], "letters": list(LETTERS[:len(kv)]), "trait": "risk"}
for code, cat, title in (("unverified_rumour", "unv_bear_fresh", "An unverified post says the company is in trouble"),
                         ("verified_report", "ver_bear_fresh", "A verified wire story says the company is in trouble"),
                         ("calm_market", "none", "A calm market, no news")):
    st = {"regime": "equity", "ret5": -0.01 if cat != "none" else 0.0, "exposure": 1.0, "pnl": 0.0, "news": cat}
    system, user, vals = state_prompts(st, False)
    vig[code] = {"title": title, "system": system, "user": user, "options": [f"{v:g}x" for v in vals],
                 "letters": list(LETTERS[:len(vals)]), "trait": "credulity" if cat != "none" else "risk"}
integ = load_integrity() or []
pick = [it for it in integ if it["id"] in ("iv_01", "iv_10", "iv_20")][:3] or integ[:3]
for it in pick:
    body = it["question"] + "\n" + "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(it["options"])) + "\n\nAnswer:"
    vig[f"integrity_{it['id']}"] = {"title": it["category"].replace("_", " ").capitalize() if it.get("category") else it["id"],
                                    "system": NEUTRAL_SYSTEM + " " + LETTER_RULE, "user": body, "options": it["options"],
                                    "letters": list(LETTERS[:len(it["options"])]), "trait": "integrity", "unethical": it["unethical"]}

alphas = [float(a) for a in args.alphas.split(",")]
records = []
REASON_SYS = " After the letter, give one short sentence explaining your choice."
for vid, v in vig.items():
    settings = [{}] + [{v["trait"]: a} for a in alphas if a != 0]
    if v["trait"] != "risk":
        settings += [{"risk": a} for a in (-1.0, 1.0)]
    prompts = [r.format(v["system"], v["user"]) for _ in settings]
    probs = r.choose(prompts, v["letters"], alphas=settings)
    # the stated reason is generated after the model's own most likely answer is written for it, so the reason
    # always explains the choice shown in the bars
    chosen = [v["letters"][int(p.argmax())] for p in probs]
    gen_prompts = [r.format(v["system"].replace(LETTER_RULE, "").strip() + REASON_SYS, v["user"],
                            prefill=f"{c}. Reason:") for c in chosen]
    texts = r.generate(gen_prompts, alphas=settings, max_new_tokens=60)
    for s, p, t in zip(settings, probs, texts):
        t = re.sub(r"\s+", " ", t).strip()
        t = re.sub(r"^(Answer\s*:?\s*)?\(?[A-J]\)?[.:)]\s*", "", t)
        t = re.sub(r"^Reason\s*:\s*", "", t)[:240]
        records.append({"vignette_id": vid, "model": args.model, "alphas": s, "probs": [float(x) for x in p], "reason": t})
    print(vid, "done", flush=True)
out = {"default_model": args.model, "layer": args.layer, "note": "Precomputed; stated reasons are model outputs and may not reflect the computation behind the choice.",
       "vignettes": {k: {kk: vv for kk, vv in v.items() if kk not in ("system",)} for k, v in vig.items()}, "records": records}
json.dump(out, open("results/site_decide.json", "w"), indent=1)
print("records", len(records))
