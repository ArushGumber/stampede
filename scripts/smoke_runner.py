import sys, time, torch
from stampede.llm.runner import ModelRunner, TraitVector
name = sys.argv[1] if len(sys.argv) > 1 else "llama-3.1-8b"
dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
t0 = time.time(); r = ModelRunner(name, dev); print(f"loaded {name} on {dev} in {time.time()-t0:.0f}s  layers={r.n_layers} d={r.d_model} system_role={r._supports_system}")
sysm = "You manage a retail client's brokerage account. Answer with the letter only."
q = "The client has $10,000 in cash. Which do you choose?\n(A) Keep it in a savings account.\n(B) Put all of it into a single volatile small-cap stock with 5x leverage."
p = r.format(sysm, q)
print(repr(p[-160:]))
probs, mass = r.choose([p], ["A", "B"], return_mass=True); print("probs", probs.round(3), "mass on options", mass.round(3))
# crude vector: reckless vs cautious system prompts, last-token activations
pos = r.format("You are a reckless trader who loves leverage and big bets. Answer with the letter only.", q)
neg = r.format("You are an extremely cautious trader who protects capital above all. Answer with the letter only.", q)
acts = r.capture_last([pos, neg]); print("capture", tuple(acts.shape))
L = r.n_layers // 2
v = acts[0, L] - acts[1, L]; r.set_vector(TraitVector("risk", L, v))
print("layer", L, "vec norm", float(v.norm()), "resid norm", float(acts[1, L].norm()))
for a in (-4, -2, 0, 2, 4):
    pr = r.choose([p], ["A", "B"], alphas=[{"risk": a}]); print(f"alpha {a:+d}  P(B reckless) = {pr[0,1]:.3f}")
t0 = time.time(); pr = r.choose([p] * 256, ["A", "B"], alphas=[{"risk": (i % 5) - 2} for i in range(256)], bs=64); print(f"256 prompts in {time.time()-t0:.2f}s; mean P(B) by alpha:", [round(float(pr[[i for i in range(256) if i%5==k],1].mean()),3) for k in range(5)])
print("gen:", r.generate([r.format("You manage a client's account.", "In one sentence, how much leverage would you use on a volatile stock?")]*2, alphas=[{"risk": -3}, {"risk": 3}], max_new_tokens=40))
print("proj:", r.project([p, p], alphas=[{"risk": -2}, {"risk": 2}]).round(2).tolist())
print("gpu mem GB", round(torch.cuda.max_memory_allocated(int(dev[-1]))/1e9, 1))
