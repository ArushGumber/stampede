"""LaTeX table: agents in economic units (the yardstick). Reads closed-model batteries, the unsteered/steered Gemma
dose files and the alpha equivalents; writes paper/table_yardstick.tex. Numbers are printed exactly as measured."""
import json
import os

ROWS = [("claude-haiku", "Claude Haiku"), ("claude-sonnet", "Claude Sonnet"), ("claude-opus", "Claude Opus"),
        ("claude-fable", "Claude Fable"), ("gpt-6-luna", "GPT-6 Luna"), ("gpt-6-sol", "GPT-6 Sol")]
eq = {(r["agent"], r["metric"]): r for r in json.load(open("results/e3/alpha_equivalent.json"))} \
    if os.path.exists("results/e3/alpha_equivalent.json") else {}


def fmt(x, d=2, signed=False):
    if x is None or x != x:
        return "--"
    return (f"{x:+.{d}f}" if signed else f"{x:.{d}f}").replace("-", "$-$")


lines = []
for key, label in ROWS:
    p = f"results/e3/{key}/battery.json"
    if not os.path.exists(p):
        continue
    s = json.load(open(p))["summary"]
    a = eq.get((key, "kelly_fraction"), {}).get("alpha_equivalent")
    lines.append(f"{label} & {fmt(s['crra_gamma'], 2, True)} & {fmt(s['kelly_fraction'])} & {fmt(s['credulity'])} & "
                 f"{fmt(100 * s.get('unethical_rate', float('nan')), 1)} & {fmt(a)} \\\\")
dose = "results/e1/gemma-4-12b/dose_risk_syslast_L24.json"
if os.path.exists(dose):
    recs = json.load(open(dose))["records"]
    for alpha, label in ((0.0, "Gemma-4-12B"), (1.0, "\\quad risk dial $+1$"), (2.0, "\\quad risk dial $+2$")):
        r = [x for x in recs if x["kind"] == "steer" and abs(x["alpha"] - alpha) < 1e-9]
        if r:
            r = r[0]
            cred = r["credulity"] if abs(r.get("react_verified", 1)) >= 0.05 else None
            lines.append(f"{label} & {fmt(r['crra_gamma'], 2, True)} & {fmt(r['kelly_fraction'])} & {fmt(cred)} & "
                         f"{fmt(100 * r['unethical_rate'], 1)} & {fmt(alpha)} \\\\")
tex = r"""\begin{table}[t]
  \caption{Agents in economic units. $\gamma$: CRRA risk aversion from Holt--Laury lotteries (0 is risk neutral);
  Kelly: stake as a multiple of the Kelly bet; credulity: reaction to an unverified post divided by reaction to a
  verified report; unethical: \% of dilemmas; $\alpha$: risk-dial strength at which Gemma matches the agent's Kelly betting.}
  \label{tab:yardstick}
  \small
  \begin{tabular}{lrrrrr}
    \toprule
    Agent & $\gamma$ & Kelly & Cred. & Uneth. & $\alpha$ \\
    \midrule
""" + "\n".join("    " + l for l in lines) + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
open("paper/table_yardstick.tex", "w").write(tex)
print(tex)
