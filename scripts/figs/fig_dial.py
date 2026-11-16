"""Figure: the risk dial (Gemma-4-12B, layer 24) in economic units, with its two controls.

Left: stake as a multiple of the Kelly stake against steering strength alpha. The risk direction moves the stake
continuously; the written personas (five cautious, five reckless instructions; not on the alpha scale) are drawn
as two levels. Right: accuracy on 60 easy four-option trading questions ("capability" in the data) for the risk
direction and for random directions of the same norm at the same alphas (two seeds); chance is 0.25.

Commercial models are not drawn here; their alpha equivalents are in the yardstick table.

Reads results/e1/gemma-4-12b/dose_risk_syslast_L24.json.
Writes paper/figs/fig_dial.pdf and paper/figs/fig_dial.png (300 dpi); --out PREFIX writes elsewhere.
"""
import argparse
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
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(ROOT, "paper/figs/fig_dial"))
ap.add_argument("--width", type=float, default=6.5)
ap.add_argument("--height", type=float, default=2.9)
ARGS = ap.parse_args()
OUT = ARGS.out

COLW_IN = 241.14749 / 72.27          # acmart sigconf \columnwidth
H_MAX_IN = 1.50                      # height budget at \columnwidth

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
mass_r = np.array([r["option_mass"] for r in rand])
kelly_cautious = np.array([r["kelly_fraction"] for r in pers if r["alpha"] < 0])
kelly_reckless = np.array([r["kelly_fraction"] for r in pers if r["alpha"] > 0])
CHANCE = 0.25                        # four options per capability question

# the numbers the caption and the paper state, checked against the data
assert len(rand) == 8 and len(kelly_cautious) == 5 and len(kelly_reckless) == 5
i0, i2 = list(a_s).index(0.0), list(a_s).index(2.0)
assert f"{kelly_s[i0]:.2f}" == "0.12" and f"{kelly_s[i2]:.2f}" == "2.26", "stake at alpha 0 and 2"
assert np.all(np.diff(kelly_s[a_s >= 0]) > 0), "stake rises monotonically for alpha >= 0"
assert f"{cap_s.min():.2f}" == "0.73" and f"{cap_s.max():.2f}" == "0.82", "risk-direction capability"
assert f"{cap_r.min():.2f}" == "0.22" and f"{cap_r.max():.2f}" == "0.27", "random-direction capability"
assert f"{kelly_cautious.mean():.2f}" == "0.13" and f"{kelly_reckless.mean():.2f}" == "2.62", "persona levels"
assert np.ptp(kelly_cautious) < 1e-3 and np.ptp(kelly_reckless) < 1e-3, "each persona set sits on one level"

print(f"stake (x Kelly) alpha 0 -> 2: {kelly_s[i0]:.4f} -> {kelly_s[i2]:.4f}; alpha < 0: "
      f"{kelly_s[a_s < 0].min():.4f}..{kelly_s[a_s < 0].max():.4f}")
print(f"capability, risk direction {cap_s.min():.4f}..{cap_s.max():.4f}; random {cap_r.min():.4f}..{cap_r.max():.4f} "
      f"(probability on any answer letter {mass_r.min():.4f}..{mass_r.max():.4f})")
print(f"personas: cautious {kelly_cautious.mean():.4f}, reckless {kelly_reckless.mean():.4f}")

# ---------------------------------------------------------------- style
S.apply_style()
plt.rcParams.update({"font.size": 19.5, "axes.labelsize": 20.5, "xtick.labelsize": 19,
                     "ytick.labelsize": 19})
ANN = 19.5          # annotation font size (pt, before the figure is scaled to column width)
TXT = dict(fontsize=ANN, zorder=6)
C_DIAL, C_PERS, C_RAND = S.CLAY, S.SLATE, S.GRAY
LINE = dict(color=C_DIAL, lw=2.8, marker="o", ms=6.5, zorder=4)
XL = (-2.3, 2.3)

W, H = ARGS.width, ARGS.height
# margins in inches, so the layout holds if the canvas changes: left y label + ticks, gap with the right
# panel's y label + ticks, bottom ticks + shared x label
ML, GAP, MR, MB, MT = 0.62, 0.88, 0.03, 0.74, 0.05
pw = W - ML - GAP - MR
wl, wr = 0.625 * pw, 0.375 * pw
fig = plt.figure(figsize=(W, H))
axL = fig.add_axes([ML / W, MB / H, wl / W, (H - MB - MT) / H])
axR = fig.add_axes([(ML + wl + GAP) / W, MB / H, wr / W, (H - MB - MT) / H])

# ---------------------------------------------------------------- left: the dial in Kelly units
ax = axL
# personas are not on the alpha scale, so they are drawn as levels, not points
for y in (kelly_reckless.mean(), kelly_cautious.mean()):
    ax.axhline(y, color=C_PERS, lw=2.2, ls=(0, (4, 2.5)), zorder=2)
ax.plot(a_s, kelly_s, **LINE)

ax.text(-2.2, kelly_reckless.mean() - 0.1, "reckless persona", ha="left", va="top", **TXT)
ax.text(0.42, 1.0, "risk direction", ha="right", va="center", **TXT)
ax.text(2.29, kelly_cautious.mean() + 0.11, "cautious\npersona", ha="right", va="bottom", linespacing=0.9,
        **TXT)

ax.set_xlim(*XL)
ax.set_ylim(0, 2.75)
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_ylabel("stake (× Kelly)", labelpad=4)

# ---------------------------------------------------------------- right: accuracy, risk vs random
ax = axR
ax.axhline(CHANCE, color=S.SUBINK, lw=1.5, ls=(0, (1.2, 2.2)), zorder=2)
ax.plot(a_s, cap_s, **LINE)
ax.plot(a_r, cap_r, ls="none", marker="X", ms=12, color=C_RAND, mew=0, zorder=4)

ax.text(0, 0.955, "risk direction", ha="center", va="center", **TXT)
ax.text(0, 0.35, "random\ndirections", ha="center", va="bottom", linespacing=0.95, **TXT)
ax.text(0, 0.10, "chance", ha="center", va="center", **TXT)

ax.set_xlim(*XL)
ax.set_ylim(0, 1.06)
ax.set_xticks([-2, 0, 2])
ax.set_xticks([-1, 1], minor=True)
ax.tick_params(axis="x", which="minor", length=4, width=1.0)
ax.set_yticks([0, 0.5, 1.0])
ax.set_yticklabels(["0", "0.5", "1"])
ax.set_ylabel("accuracy", labelpad=4)

# one x label for both panels (same alpha scale), placed just under the lower tick labels
fig.canvas.draw()
r = fig.canvas.get_renderer()
tick_bottom = min(t.get_window_extent(r).y0 for a in (axL, axR) for t in a.get_xticklabels() if t.get_text())
x_mid = (axL.get_position().x0 + axR.get_position().x1) / 2
xlab = fig.text(x_mid, fig.transFigure.inverted().transform((0, tick_bottom - 4))[1],
                r"steering strength $\alpha$", ha="center", va="top", fontsize=20.5)


# ---------------------------------------------------------------- checks and save
def all_text(fig):
    """Every visible text object with a name: annotations, axis labels, tick labels, figure text."""
    out = []
    for i, ax in enumerate(fig.axes):
        out += [(f"ax{i} note '{t.get_text()}'", t) for t in ax.texts]
        out += [(f"ax{i} {n}label", lab) for n, lab in (("x", ax.xaxis.label), ("y", ax.yaxis.label))
                if lab.get_text()]
        out += [(f"ax{i} tick '{t.get_text()}'", t)
                for t in ax.get_xticklabels() + ax.get_yticklabels() if t.get_text()]
    out += [(f"fig '{t.get_text()}'", t) for t in fig.texts]
    return out


def text_pairs(fig, pad_px=2):
    """Named text-vs-text overlaps of any kind (the skill's checker skips axis labels)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = [(n, t.get_window_extent(r)) for n, t in all_text(fig)]
    hits = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1].padded(pad_px), items[j][1]
            if not (a.x1 < b.x0 or b.x1 < a.x0 or a.y1 < b.y0 or b.y1 < a.y0):
                hits.append((items[i][0], items[j][0]))
    print(f"[fig_dial] named text-vs-text overlaps (pad {pad_px}px) = {len(hits)} {hits}")
    return hits


def mark_hits(fig, pad_px=6):
    """Annotation text drawn over a line segment or marker, or outside its axes."""
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
                            hits.append((t.get_text(), "marker", tuple(np.round(
                                ax.transData.inverted().transform((x, y)), 2))))
                if ln.get_linestyle() not in ("None", "none", ""):
                    lw = ln.get_linewidth() * fig.dpi / 72 / 2
                    bl = b.padded(lw)
                    for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
                        if any(bl.contains(x0 + s * (x1 - x0), y0 + s * (y1 - y0)) for s in np.linspace(0, 1, 400)):
                            hits.append((t.get_text(), "line", ln.get_color()))
                            break
            bt = t.get_window_extent(r)
            if not ax.bbox.contains(bt.x0, bt.y0) or not ax.bbox.contains(bt.x1, bt.y1):
                hits.append((t.get_text(), "outside axes"))
    print(f"[fig_dial] text over lines/markers or outside axes (pad {pad_px}px) = {len(hits)} {hits}")
    return hits


n = check(fig, "fig_dial")
text_pairs(fig)
mark_hits(fig)
fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=300)
tb = fig.get_tightbbox(fig.canvas.get_renderer())
w_in, h_in = tb.width + 0.1, tb.height + 0.1          # savefig.pad_inches = 0.05 on each side
k = COLW_IN / w_in
print(f"saved {w_in:.3f} x {h_in:.3f} in (aspect {w_in / h_in:.3f}); at columnwidth {COLW_IN:.3f} in: scale {k:.3f}, "
      f"ticks {19 * k:.1f} pt, axis labels {20.5 * k:.1f} pt, notes {ANN * k:.1f} pt, "
      f"height {h_in * k:.3f} in (budget {H_MAX_IN:.2f} in)")
assert h_in * k <= H_MAX_IN, "taller than the height budget at column width"
