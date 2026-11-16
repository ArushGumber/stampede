"""Figure: the risk dial (Gemma-4-12B, layer 24) in economic units, with its controls.

Left: stake as a multiple of the Kelly stake against steering strength alpha. The steered model moves
continuously; the written personas (not on the alpha scale) sit on two fixed levels; the six commercial
models are placed on the curve at the alpha where the steered model matches their Kelly fraction.
Right: capability (accuracy on 60 easy finance questions) for the risk direction and for matched-norm
random directions at the same layer.

Reads results/e1/gemma-4-12b/dose_risk_syslast_L24.json and results/e3/alpha_equivalent.json.
Writes paper/figs/fig_dial.pdf and paper/figs/fig_dial.png (300 dpi).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/.claude/skills/paper-figures/assets"))
import style as S  # noqa: E402
from overlap_check import check  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOSE = os.path.join(ROOT, "results/e1/gemma-4-12b/dose_risk_syslast_L24.json")
EQUIV = os.path.join(ROOT, "results/e3/alpha_equivalent.json")
OUT = os.path.join(ROOT, "paper/figs/fig_dial")

# ---------------------------------------------------------------- data
recs = json.load(open(DOSE))["records"]
steer = sorted((r for r in recs if r["kind"] == "steer"), key=lambda r: r["alpha"])
rand = [r for r in recs if r["kind"] == "random"]
pers = [r for r in recs if r["kind"] == "prompt"]

a_s = np.array([r["alpha"] for r in steer])
kelly_s = np.array([r["kelly_fraction"] for r in steer])
cap_s = np.array([r["capability"] for r in steer])
a_r = np.array([r["alpha"] for r in rand])
cap_r = np.array([r["capability"] for r in rand])
kelly_cautious = np.array([r["kelly_fraction"] for r in pers if r["alpha"] < 0])
kelly_reckless = np.array([r["kelly_fraction"] for r in pers if r["alpha"] > 0])

eq = [e for e in json.load(open(EQUIV)) if e["metric"] == "kelly_fraction" and e["alpha_equivalent"] is not None]
a_c = np.array([e["alpha_equivalent"] for e in eq])
kelly_c = np.array([e["value"] for e in eq])

# sanity: the commercial points must lie on the linear interpolation of the steered curve
assert np.allclose(np.interp(a_c, a_s, kelly_s), kelly_c, atol=1e-6), "alpha equivalents off the curve"
assert len(eq) == 6 and len(rand) == 8 and len(kelly_cautious) == 5 and len(kelly_reckless) == 5

print(f"steer kelly {kelly_s.min():.3f}..{kelly_s.max():.3f}; capability {cap_s.min():.3f}..{cap_s.max():.3f}")
print(f"random capability {cap_r.min():.3f}..{cap_r.max():.3f}")
print(f"cautious persona kelly {kelly_cautious.min():.4f}..{kelly_cautious.max():.4f}; "
      f"reckless {kelly_reckless.min():.4f}..{kelly_reckless.max():.4f}")
print(f"commercial alpha {a_c.min():.3f}..{a_c.max():.3f}; kelly {kelly_c.min():.3f}..{kelly_c.max():.3f}")

# ---------------------------------------------------------------- style
S.apply_style()
plt.rcParams.update({"font.size": 19.5, "axes.labelsize": 20.5, "xtick.labelsize": 19,
                     "ytick.labelsize": 19})
ANN = 19.0          # annotation font size (pt, before the figure is scaled to column width)
TXT = dict(fontsize=ANN, zorder=3)
C_DIAL, C_PERS, C_RAND = S.CLAY, S.SLATE, S.GRAY

W, H = 6.8, 3.0
fig = plt.figure(figsize=(W, H))
axL = fig.add_axes([0.098, 0.215, 0.455, 0.765])
axR = fig.add_axes([0.70, 0.215, 0.295, 0.765])
XL = (-2.25, 2.4)

# ---------------------------------------------------------------- left: the dial in Kelly units
ax = axL
# personas are not on the alpha scale, so they are drawn as levels, not points
for y in (kelly_reckless.mean(), kelly_cautious.mean()):
    ax.axhline(y, color=C_PERS, lw=2.2, ls=(0, (4, 2.5)), zorder=2)
ax.plot(a_s, kelly_s, color=C_DIAL, lw=2.8, marker="o", ms=6.5, zorder=4)
ax.plot(a_c, kelly_c, ls="none", marker="D", ms=6.5, mfc="white", mec=S.INK, mew=1.5, zorder=5)

ax.text(-2.12, kelly_reckless.mean() + 0.11, "reckless persona", ha="left", va="bottom", **TXT)
ax.text(0.9, kelly_cautious.mean() + 0.1, "cautious\npersona", ha="left", va="bottom", linespacing=0.95, **TXT)
ax.text(1.2, 1.95, "risk direction", ha="right", va="center", **TXT)
ax.text(0.37, 1.1, "commercial\nmodels", ha="right", va="center", linespacing=0.95, **TXT)

ax.set_xlim(*XL)
ax.set_ylim(0, 3.15)
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([0, 1, 2, 3])
ax.set_xlabel(r"steering strength $\alpha$", labelpad=2)
ax.set_ylabel("stake / Kelly stake", labelpad=4)

# ---------------------------------------------------------------- right: capability, risk vs random
ax = axR
ax.axhline(0.25, color=S.SUBINK, lw=1.5, ls=(0, (1.2, 2.2)), zorder=2)
ax.plot(a_s, cap_s, color=C_DIAL, lw=2.8, marker="o", ms=6.5, zorder=4)
ax.plot(a_r, cap_r, ls="none", marker="X", ms=10, color=C_RAND, mec="white", mew=0.6, zorder=4)

ax.text(0, 0.95, "risk direction", ha="center", va="center", **TXT)
ax.text(0, 0.47, "random\ndirections", ha="center", va="center", linespacing=0.95, **TXT)
ax.text(0, 0.12, "chance", ha="center", va="center", **TXT)

ax.set_xlim(*XL)
ax.set_ylim(0, 1.02)
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
ax.set_xlabel(r"steering strength $\alpha$", labelpad=2)
ax.set_ylabel("capability", labelpad=4)

# ---------------------------------------------------------------- checks and save
def mark_hits(fig, pad_px=6):
    """Text drawn over a line segment or marker (the skill's checker covers only text and ticks)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    hits = []
    for ax in fig.axes:
        for t in ax.texts:
            b = t.get_window_extent(r).padded(pad_px)
            for ln in ax.lines:
                xy = ln.get_transform().transform(np.column_stack(ln.get_data()))
                if ln.get_marker() not in (None, "None", "none", ""):
                    rad = ln.get_markersize() * fig.dpi / 72 / 2
                    bb = b.padded(rad)
                    for x, y in xy:
                        if bb.contains(x, y):
                            hits.append((t.get_text(), "marker", ln.get_markeredgecolor(), tuple(np.round(
                                ax.transData.inverted().transform((x, y)), 2))))
                if ln.get_linestyle() not in ("None", "none", ""):
                    for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
                        ts = np.linspace(0, 1, 200)
                        if any(b.contains(x0 + s * (x1 - x0), y0 + s * (y1 - y0)) for s in ts):
                            hits.append((t.get_text(), "line", ln.get_color()))
                            break
        for t in ax.texts:
            b = t.get_window_extent(r)
            if not ax.bbox.contains(b.x0, b.y0) or not ax.bbox.contains(b.x1, b.y1):
                hits.append((t.get_text(), "outside axes"))
    print(f"[fig_dial] text over lines/markers or outside axes = {len(hits)} {hits}")
    return hits


n = check(fig, "fig_dial")
mark_hits(fig)
fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=300)
tb = fig.get_tightbbox(fig.canvas.get_renderer())
w_in, h_in = tb.width + 0.1, tb.height + 0.1          # savefig.pad_inches = 0.05 on each side
k = 3.33 / w_in
print(f"saved {w_in:.2f} x {h_in:.2f} in (aspect {w_in / h_in:.2f}); at columnwidth 3.33 in: scale {k:.3f}, "
      f"ticks {19 * k:.1f} pt, axis labels {20.5 * k:.1f} pt, notes {ANN * k:.1f} pt, height {h_in * k:.2f} in")
