"""Figure: checking a headline's source defuses a false post, not the same story on a verified-looking wire.

One message, one panel. The same false story reaches a market of agents in one of two forms (the x groups), as the
agents see it: an unverified social post (hcode fake_unv) or a verified wire (fake_ver, i.e. a verified-looking
source such as a compromised official account). Equities, agents managing 2x the book's depth, 16 seeds per cell.
Bars: excess overshoot, how far the price falls below fundamental (fair) value beyond the no-headline twin with the
same seed; whiskers: bootstrap 95% intervals. Three markets per form, in the same order and style in both groups:
  Gemma   unsteered Gemma-4-12B, no safeguard (solid clay). It reacts identically to post and wire.
  filter  the same Gemma agents with the provenance filter that hides unverified posts (Gemma's hue, lighter;
          delaying unverified posts gives identical results). The filter's twin equals the no-safeguard twin, so its
          drop is real in raw overshoot too (2.27% -> 0.27%), and the wire is untouched.
  Opus    a population replaying Claude Opus's response table, no safeguard (white, ink edge, the style of the
          commercial models in the other figures). It discounts the post on its own but not the wire.
The bars carry direct labels once, in the post group, where the two short bars leave room; the wire group repeats the
order and styles. Labelled "filter", not "post filter", which also reads as "post-filter" (after filtering).
Left out on purpose (the paper text keeps them): the capital sweep, the other commercial models, and the
half-reckless crypto market where a leverage cap or risk dial -1 deepens the crash (0.06% -> 1.68% or 3.05%).

Reads results/e4/tipping_point.json (Opus), results/e6/safeguards.json (Gemma with and without the filter) and
results/e6/safeguards_cells.json (raw overshoot and twin, as a check). Writes paper/figs/fig_market.pdf and .png.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/.claude/skills/paper-figures/assets"))
import style as S  # noqa: E402
from overlap_check import check  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TIP = os.path.join(ROOT, "results/e4/tipping_point.json")
SAFE = os.path.join(ROOT, "results/e6/safeguards.json")
CELLS = os.path.join(ROOT, "results/e6/safeguards_cells.json")
OUT = os.path.join(ROOT, "paper/figs/fig_market")

REGIME, CAP = "equity", 2.0
COLUMN_IN = 241.14839 / 72.27        # acmart sigconf \columnwidth in inches
MAX_H_IN = 1.48                      # rendered height at \columnwidth must not exceed this (fig_dial beside it: 1.487)

# ---------------------------------------------------------------- data
tip = json.load(open(TIP))
safe = json.load(open(SAFE))
cells = json.load(open(CELLS))


def tp(pop, hcode, cap=CAP):
    """Excess overshoot [mean, lo, hi] in % from the population sweep (no safeguard)."""
    cs = [c for c in tip if c["tag"] == "models" and c["pop"] == pop and c["regime"] == REGIME
          and c["hcode"] == hcode and c["safeguard_name"] == "none" and c["capital"] == cap]
    assert len(cs) == 1 and cs[0]["n"] == 16, (pop, hcode, cap, len(cs))
    c = cs[0]
    return 100 * np.array([c["excess_overshoot"], c["excess_overshoot_lo"], c["excess_overshoot_hi"]])


def sg(pop, name):
    """Excess overshoot [mean, lo, hi] in % for post and wire from the safeguard sweep."""
    rs = [r for r in safe if r["pop"] == pop and r["regime"] == REGIME and r["capital"] == CAP and r["safeguard"] == name]
    assert len(rs) == 1, (pop, name, len(rs))
    r = rs[0]
    return {"post": 100 * np.array([r["fake_unv_overshoot"], r["fake_unv_lo"], r["fake_unv_hi"]]),
            "wire": 100 * np.array([r["fake_ver_overshoot"], r["fake_ver_lo"], r["fake_ver_hi"]])}


def raw(pop, name, hcode):
    """(raw headline overshoot, twin overshoot, excess) in % for one safeguard cell."""
    cs = [c for c in cells if c["pop"] == pop and c["regime"] == REGIME and c["capital"] == CAP
          and c["safeguard_name"] == name and c["hcode"] == hcode]
    assert len(cs) == 1 and cs[0]["n"] == 16, (pop, name, hcode, len(cs))
    c = cs[0]
    return 100 * c["overshoot"], 100 * (c["overshoot"] - c["excess_overshoot"]), 100 * c["excess_overshoot"]


D = {"gemma": sg("gemma-base", "none"),
     "filter": sg("gemma-base", "hide_unverified"),
     "opus": {"post": tp("claude-opus", "fake_unv"), "wire": tp("claude-opus", "fake_ver")}}
ORDER = ("gemma", "filter", "opus")

# sanity: Gemma reacts identically to both forms, in both sweeps, and the two sweeps agree
assert abs(D["gemma"]["post"][0] - D["gemma"]["wire"][0]) < 1e-9
assert abs(tp("gemma-base", "fake_unv")[0] - D["gemma"]["post"][0]) < 0.01
assert abs(tp("gemma-base", "fake_ver")[0] - D["gemma"]["wire"][0]) < 0.01
# hiding and delaying unverified posts give the same result; the filter leaves the wire untouched
assert abs(D["filter"]["post"][0] - sg("gemma-base", "delay_unverified")["post"][0]) < 1e-6
assert abs(D["filter"]["wire"][0] - D["gemma"]["wire"][0]) < 1e-6
# the filter's drop is not a twin artefact: same twin, raw headline overshoot falls too
r_none, r_hide = raw("gemma-base", "none", "fake_unv"), raw("gemma-base", "hide_unverified", "fake_unv")
assert abs(r_none[1] - r_hide[1]) < 1e-6 and r_hide[0] < 0.5 * r_none[0]
# Opus: at most 0.10% on the post at any capital, the 2x values are the ones drawn
opus_post_all = [tp("claude-opus", "fake_unv", c)[0] for c in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)]
assert round(max(opus_post_all), 2) == 0.10 and max(opus_post_all) == D["opus"]["post"][0]
# the numbers quoted in the caption
QUOTED = {("gemma", "post"): 1.74, ("filter", "post"): -0.25, ("opus", "post"): 0.10, ("opus", "wire"): 1.57}
for (k, h), q in QUOTED.items():
    assert round(D[k][h][0], 2) == q, (k, h, D[k][h][0], q)
for k in ORDER:
    for h in ("post", "wire"):
        m, lo, hi = D[k][h]
        assert lo <= m <= hi
        print(f"  {k:6s} {h:4s} {m:+.3f}%  [{lo:+.3f}, {hi:+.3f}]")
print(f"  raw post overshoot, Gemma: {r_none[0]:.2f}% (twin {r_none[1]:.2f}) -> with filter {r_hide[0]:.2f}% "
      f"(twin {r_hide[1]:.2f})")

# ---------------------------------------------------------------- style
S.apply_style()
FS = 24                                   # all text; renders at 11 pt at \columnwidth (body text is 9 pt)
plt.rcParams.update({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS - 1})
STY = {"gemma": dict(color=S.CLAY, ec=S.CLAY),             # Gemma = clay, as in every figure
       "filter": dict(color=S.CLAY_SOFT, ec=S.CLAY),       # the same Gemma agents behind the filter: Gemma's hue
       "opus": dict(color="white", ec=S.INK)}               # commercial model = white with ink edge
LABEL = {"gemma": "Gemma", "filter": "filter", "opus": "Opus"}
EB = dict(fmt="none", ecolor=S.INK, elinewidth=1.5, capsize=5, capthick=1.5, zorder=4)
BW, STEP, GGAP = 0.74, 1.0, 0.62          # bar width, bar pitch, extra gap between the two forms (data units)
LEFT = 0.38                               # y axis to the first bar's edge: room for the Gemma label's overhang
LIFT = 7                                  # gap from a whisker's top to its label (pt)
XPAD = 9                                  # gap below the axes to the two form labels (pt): clears the lowest cap
W_IN, H_IN = 7.26, 3.18                   # the axes fill the width; the y label's length sets the height
L_IN, R_IN, B_IN, T_IN = 0.70, 0.05, 0.50, 0.04

fig = plt.figure(figsize=(W_IN, H_IN))
ax = fig.add_axes([L_IN / W_IN, B_IN / H_IN, 1 - (L_IN + R_IN) / W_IN, 1 - (B_IN + T_IN) / H_IN])

xs, x = {}, 0.0
for h in ("post", "wire"):
    for k in ORDER:
        xs[(h, k)] = x
        x += STEP
    x += GGAP

for (h, k), xx in xs.items():
    m, lo, hi = D[k][h]
    ax.bar(xx, m, width=BW, lw=1.8, zorder=3, **STY[k])
    ax.errorbar(xx, m, yerr=[[m - lo], [hi - m]], **EB)
ax.axhline(0, color=S.INK, lw=1.2, zorder=5)

# direct labels once, above each bar of the post group; the wire group repeats the same order and styles
labels = {k: ax.annotate(LABEL[k], (xs[("post", k)], max(D[k]["post"][2], 0)), xytext=(0, LIFT),
                         textcoords="offset points", ha="center", va="bottom", fontsize=FS, zorder=6)
          for k in ORDER}
ax.set_xticks([xs[("post", "filter")], xs[("wire", "filter")]])
ax.set_xticklabels(["unverified post", "verified-looking wire"])
ax.tick_params(axis="x", length=0, pad=XPAD)
ax.spines["bottom"].set_visible(False)
ax.set_yticks([0, 1, 2])
ax.set_ylabel("extra price fall (%)", labelpad=3)
x_hi = xs[("wire", ORDER[-1])] + BW / 2 + 0.03
ax.set_xlim(xs[("post", ORDER[0])] - BW / 2 - LEFT, x_hi)
y_lo = min(D[k][h][1] for k in ORDER for h in ("post", "wire")) - 0.035   # just below the lowest whisker cap
ax.set_ylim(y_lo, 2.9)


def extent(t):
    """Extent (x0, x1, y0, y1) of a text in data units."""
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer())
    (x0, y0), (x1, y1) = ax.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    return x0, x1, y0, y1


# the top of the axes sits just above the Gemma label: no headroom beyond what the label needs
for _ in range(4):
    ax.set_ylim(y_lo, extent(labels["gemma"])[3] + 0.02)
# the wire label ends flush with the last bar instead of overhanging the right edge as dead space
w0, w1, _, _ = extent(ax.get_xticklabels()[1])
ax.set_xticks([xs[("post", "filter")], x_hi - (w1 - w0) / 2])
ax.set_xticklabels(["unverified post", "verified-looking wire"])
# the y label is a little longer than the axes; top-align it with the axes so it does not raise the figure's top edge
# (its lower end hangs beside the form labels, where the left margin is empty)
_, _, l0, l1 = extent(ax.yaxis.label)
ax.yaxis.label.set_y(0.5 - max(l1 - ax.get_ylim()[1], 0) / (ax.get_ylim()[1] - ax.get_ylim()[0]))


# ---------------------------------------------------------------- checks and save
def text_hits(fig, pad_px=4):
    """Text over bars, whiskers, caps or the zero line, and annotations leaving the axes (the skill's checker
    covers only text against text and ticks). Labels outside the canvas are fine: savefig uses a tight bbox."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px_pt = fig.dpi / 72
    hits = []
    for a in fig.axes:
        texts = [t for t in a.texts if t.get_text().strip()]
        labels = [t for t in a.get_xticklabels() + a.get_yticklabels() + [a.xaxis.label, a.yaxis.label]
                  if t.get_text().strip()]
        for t in texts + labels:
            b = t.get_window_extent(r).padded(pad_px)
            for p in a.patches:
                if b.overlaps(p.get_window_extent(r)):
                    hits.append((t.get_text(), "bar"))
            segs = []
            for c in a.collections:
                if isinstance(c, LineCollection):
                    segs += [c.get_transform().transform(s) for s in c.get_segments()]
            for ln in a.lines:
                xy = ln.get_transform().transform(np.column_stack(ln.get_data()))
                mk = ln.get_marker()
                if mk not in (None, "None", "none", ""):
                    half = ln.get_markersize() * px_pt / 2            # caps are flat: "_" is wide, "|" is tall
                    thin = ln.get_markeredgewidth() * px_pt / 2
                    hx, hy = {"_": (half, thin), "|": (thin, half)}.get(mk, (half, half))
                    if any(b.x0 - hx <= px <= b.x1 + hx and b.y0 - hy <= py <= b.y1 + hy for px, py in xy):
                        hits.append((t.get_text(), "marker"))
                if ln.get_linestyle() not in ("None", "none", "") and len(xy) > 1:
                    segs.append(xy)
            for sgm in segs:
                for (x0, y0), (x1, y1) in zip(sgm[:-1], sgm[1:]):
                    if any(b.contains(x0 + s * (x1 - x0), y0 + s * (y1 - y0)) for s in np.linspace(0, 1, 300)):
                        hits.append((t.get_text(), "line"))
                        break
        for t in texts:
            b = t.get_window_extent(r)
            if not (a.bbox.x0 <= b.x0 and b.x1 <= a.bbox.x1 and a.bbox.y0 <= b.y0 and b.y1 <= a.bbox.y1):
                hits.append((t.get_text(), "outside axes"))
        for i, t in enumerate(labels):               # axis labels against tick labels and each other
            for u in labels[i + 1:]:
                if t.get_window_extent(r).overlaps(u.get_window_extent(r)):
                    hits.append((t.get_text(), u.get_text()))
    print(f"[fig_market] text over bars, whiskers, caps or zero line, or outside axes = {len(hits)} {hits}")
    return hits


n = check(fig, "fig_market")
hits = text_hits(fig)
assert n == 0 and not hits
fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=300)
tb = fig.get_tightbbox(fig.canvas.get_renderer())
w_in = tb.width + 2 * plt.rcParams["savefig.pad_inches"]
h_in = tb.height + 2 * plt.rcParams["savefig.pad_inches"]
k = COLUMN_IN / w_in
print(f"saved {w_in:.2f} x {h_in:.2f} in (aspect {w_in / h_in:.3f}); at \\columnwidth {COLUMN_IN:.3f} in: scale {k:.3f}, "
      f"text {FS * k:.1f} pt, ticks {(FS - 1) * k:.1f} pt, height {h_in * k:.3f} in (limit {MAX_H_IN} in)")
assert h_in * k <= MAX_H_IN
