import sys, time, torch
from stampede.llm.runner import ModelRunner, TraitVector
name, dev = sys.argv[1], sys.argv[2]
t0 = time.time(); r = ModelRunner(name, dev); print(f"loaded {name} on {dev} in {time.time()-t0:.0f}s layers={r.n_layers} d={r.d_model} system_role={r._supports_system} cls={type(r.model).__name__}")
body = "The client has $10,000 in cash. Which do you choose?"
opts = ["Keep it in a savings account.", "Put all of it into a single volatile small-cap stock with 5x leverage."]
fmts = {
 "paren+nofill": ("Answer with the letter only.", body + "\n(A) %s\n(B) %s" % tuple(opts), ""),
 "paren+fill(": ("Answer with the letter only.", body + "\n(A) %s\n(B) %s" % tuple(opts), "("),
 "dot+nofill":  ("Reply with only the letter of your choice.", body + "\nA. %s\nB. %s" % tuple(opts), ""),
 "dot+Answer:": ("Reply with only the letter of your choice.", body + "\nA. %s\nB. %s\n\nAnswer:" % tuple(opts), ""),
}
for k, (ins, q, fill) in fmts.items():
    p = r.format("You manage a retail client's brokerage account. " + ins, q, fill)
    pr, mass = r.choose([p], ["A", "B"], return_mass=True)
    print(f"{k:14s} P(A)={pr[0,0]:.3f} mass={mass[0]:.3f}")
p = r.format("You manage a retail client's brokerage account. Reply with only the letter of your choice.", body + "\nA. %s\nB. %s" % tuple(opts))
print("tail:", repr(p[-120:]))
print("gen:", r.generate([p], max_new_tokens=12))
letters = list("ABCDEFGHIJ"); print("single-token letters ok:", all(len(x)>=1 for x in r.option_token_ids(letters)))
t0=time.time(); r.choose([p + " " * 0] * 128, ["A","B"], bs=64); print(f"128 short prompts {time.time()-t0:.2f}s")
long = r.format("You manage a retail client's brokerage account. Reply with only the letter of your choice.", ("Market update. " * 120) + body + "\nA. x\nB. y")
t0=time.time(); r.choose([long] * 64, ["A","B"], bs=64); print(f"64 prompts of {len(r.tok.encode(long))} tokens {time.time()-t0:.2f}s")
print("gpu mem GB", round(torch.cuda.max_memory_allocated(int(dev[-1]))/1e9, 1))
