"""Figure: the market results. When does a false headline crash the market, and which safeguard helps?

Left: excess overshoot (price below fundamental value, minus the no-headline twin with the same seed; 16 seeds)
against agent capital in the equity regime. The two false headlines are labelled as the agents see them:
"post" = unverified social post (fake_unv), "wire" = verified wire, i.e. a verified-looking source (fake_ver).
Unsteered Gemma-4-12B (clay) reacts identically to both, so its two curves coincide and are drawn once; the
population running Claude Opus's policy (ink, white diamonds) barely moves on the post (solid) but follows the
fake wire (dashed).

Right: excess overshoot at capital 2 in two different markets, each under its own header with its own
no-safeguard reference (dotted). Bars: bootstrap 95% intervals.
  (a) unsteered Gemma, equity: the provenance filter ("hide posts"; delaying unverified posts gives identical
      results) removes the post-driven crash (filled dot) but not the fake wire (open circle). The leverage cap
      and cautious steering are NOT drawn here: their higher excess for this population is a twin artefact (raw
      headline overshoot 2.27% with no safeguard vs 2.29% with either; only the no-headline twin differs,
      0.52% vs 0.04%, because uncapped leveraged agents make their own swings). Asserted below.
  (b) half-reckless Gemma (pop r50: half unsteered, half risk dial +1.5), crypto: no safeguard, a 1x leverage cap,
      and cautious steering (every agent's risk dial shifted by -1; cells with pop "r50_shift"). Removing the
      risk-takers' capacity turns a non-event into a crash, in raw overshoot too (0.06% -> 2.55% and 3.44%).

Reads results/e4/tipping_point.json, results/e6/safeguards.json and results/e6/safeguards_cells.json.
Writes paper/figs/fig_market.pdf and paper/figs/fig_market.png (300 dpi).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/.claude/skills/paper-figures/assets"))
import style as S  # noqa: E402
from overlap_check import check  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TIP = os.path.join(ROOT, "results/e4/tipping_point.json")
SAFE = os.path.join(ROOT, "results/e6/safeguards.json")
CELLS = os.path.join(ROOT, "results/e6/safeguards_cells.json")
OUT = os.path.join(ROOT, "paper/figs/fig_market")

REGIME, CAP_R = "equity", 2.0

# ---------------------------------------------------------------- data: left panel
tip = json.load(open(TIP))


def curve(pop, hcode):
    cs = sorted((c for c in tip if c["tag"] == "models" and c["pop"] == pop and c["regime"] == REGIME
                 and c["hcode"] == hcode and c["safeguard_name"] == "none"), key=lambda c: c["capital"])
    assert len(cs) == 6 and all(c["n"] == 16 for c in cs), (pop, hcode, len(cs))
    g = lambda k: 100 * np.array([c[k] for c in cs])  # noqa: E731
    return np.array([c["capital"] for c in cs]), g("excess_overshoot"), g("excess_overshoot_lo"), g("excess_overshoot_hi")


cap, gem_u, gem_u_lo, gem_u_hi = curve("gemma-base", "fake_unv")
_, gem_v, gem_v_lo, gem_v_hi = curve("gemma-base", "fake_ver")
_, opu_u, opu_u_lo, opu_u_hi = curve("claude-opus", "fake_unv")
_, opu_v, opu_v_lo, opu_v_hi = curve("claude-opus", "fake_ver")
# Gemma does not discount the source: both fakes give the same curve, so one line stands for both
assert np.allclose(gem_u, gem_v, atol=1e-9), "Gemma post and wire curves differ"

# ---------------------------------------------------------------- data: right panel
safe = json.load(open(SAFE))


def sgd(pop, regime, name):
    rs = [r for r in safe if r["pop"] == pop and r["regime"] == regime and r["capital"] == CAP_R and r["safeguard"] == name]
    assert len(rs) == 1, (pop, regime, name, len(rs))
    r = rs[0]
    return {"post": 100 * np.array([r["fake_unv_overshoot"], r["fake_unv_lo"], r["fake_unv_hi"]]),
            "wire": 100 * np.array([r["fake_ver_overshoot"], r["fake_ver_lo"], r["fake_ver_hi"]])}


A = {k: sgd("gemma-base", "equity", k) for k in ("none", "hide_unverified", "delay_unverified")}
B = {k: sgd("r50", "crypto", k) for k in ("none", "leverage_cap", "steer_cautious")}
assert abs(A["none"]["post"][0] - gem_u[cap == CAP_R][0]) < 0.01, "safeguard baseline differs from the left-panel point"
assert np.allclose(A["hide_unverified"]["post"][0], A["delay_unverified"]["post"][0], atol=1e-6)

# raw headline overshoot and twins from the cells, to make sure every drawn comparison survives the twin check
cells = json.load(open(CELLS))


def cell(pop, regime, name, hcode="fake_unv"):
    cs = [c for c in cells if c["pop"] == pop and c["regime"] == regime and c["capital"] == CAP_R
          and c["safeguard_name"] == name and c["hcode"] == hcode]
    assert len(cs) == 1 and cs[0]["n"] == 16, (pop, regime, name, hcode, len(cs))
    c = cs[0]
    return 100 * c["overshoot"], 100 * (c["overshoot"] - c["excess_overshoot"]), 100 * c["excess_overshoot"]


raw = {"A none": cell("gemma-base", "equity", "none"), "A hide": cell("gemma-base", "equity", "hide_unverified"),
       "A hide wire": cell("gemma-base", "equity", "hide_unverified", "fake_ver"),
       "A levcap (not drawn)": cell("gemma-base", "equity", "leverage_cap"),
       "A steer (not drawn)": cell("base_shift", "equity", "none"),
       "B none": cell("r50", "crypto", "none"), "B levcap": cell("r50", "crypto", "leverage_cap"),
       "B steer": cell("r50_shift", "crypto", "none")}
for k, (h, t, e) in raw.items():
    print(f"  {k:22s} raw {h:5.2f}  twin {t:5.2f}  excess {e:+5.2f}")
# Gemma equity: the filter's twin is the no-safeguard twin, so its drop is real in raw terms too
assert abs(raw["A hide"][1] - raw["A none"][1]) < 1e-6 and raw["A hide"][0] < 0.5 * raw["A none"][0]
assert abs(raw["A hide wire"][0] - raw["A none"][0]) < 0.05           # the fake wire is untouched
# Gemma equity: leverage cap and cautious steering do not raise the raw crash (twin artefact), so they are left out
assert abs(raw["A levcap (not drawn)"][0] - raw["A none"][0]) < 0.1 and abs(raw["A steer (not drawn)"][0] - raw["A none"][0]) < 0.1
# half-reckless crypto: both deepen the crash in raw overshoot as well as in excess
assert raw["B levcap"][0] > raw["B none"][0] + 1.5 and raw["B steer"][0] > raw["B none"][0] + 1.5
for name, d in [("A", A), ("B", B)]:
    for k, x in d.items():
        print(f"  {name} {k:17s} post {x['post'][0]:+.2f} [{x['post'][1]:+.2f},{x['post'][2]:+.2f}]  "
              f"wire {x['wire'][0]:+.2f} [{x['wire'][1]:+.2f},{x['wire'][2]:+.2f}]")

# ---------------------------------------------------------------- style
S.apply_style()
plt.rcParams.update({"font.size": 19.5, "axes.labelsize": 20.5, "xtick.labelsize": 19, "ytick.labelsize": 19})
ANN = 19.0
TXT = dict(fontsize=ANN, zorder=6)
C_GEM, C_COM = S.CLAY, S.INK
DIA = dict(marker="D", ms=6.5, mfc="white", mec=S.INK, mew=1.5)

W, H = 6.8, 3.0
fig = plt.figure(figsize=(W, H))
axL = fig.add_axes([0.098, 0.215, 0.39, 0.77])
axR = fig.add_axes([0.735, 0.215, 0.255, 0.77])

# ---------------------------------------------------------------- left: crash against capital
ax = axL
ax.plot(cap, gem_u, color=C_GEM, lw=2.8, marker="o", ms=6.5, zorder=4)
ax.plot(cap, opu_v, color=C_COM, lw=2.2, ls=(0, (4, 2.2)), zorder=4, **DIA)
ax.plot(cap, opu_u, color=C_COM, lw=2.2, zorder=4, **DIA)

ax.text(0.12, 2.42, "Gemma, wire or post", ha="left", va="center", **TXT)
ax.text(1.28, 0.93, "Opus, wire", ha="left", va="center", **TXT)
ax.text(1.28, -0.32, "Opus, post", ha="left", va="center", **TXT)

ax.set_xlim(0, 3.18)
ax.set_ylim(-0.55, 2.62)
ax.set_xticks([0, 1, 2, 3])
ax.set_yticks([0, 1, 2])
ax.set_xlabel("capital / book depth", labelpad=2)
ax.set_ylabel("excess overshoot (%)", labelpad=4)

# ---------------------------------------------------------------- right: safeguards in two markets at capital 2
ax = axR
Y = {"hA": 0.0, "A none": -1.0, "A hide": -2.0, "hB": -3.4, "B none": -4.4, "B lev": -5.4, "B steer": -6.4}
DOT = dict(ls="none", marker="o", ms=8.5, color=C_GEM, zorder=4)
RING = dict(ls="none", marker="o", ms=8.5, mfc="white", mec=C_GEM, mew=1.8, zorder=5)
REF = dict(color=S.SUBINK, lw=1.4, ls=(0, (1.2, 2.2)), zorder=1)


def ci(x, y):
    ax.plot([x[1], x[2]], [y, y], color=S.CLAY_SOFT, lw=5.5, solid_capstyle="butt", zorder=2)


# (a) unsteered Gemma, equity
ax.plot([A["none"]["post"][0]] * 2, [Y["A none"] + 0.36, Y["A hide"] - 0.36], **REF)
ci(A["none"]["post"], Y["A none"])
ax.plot([A["none"]["post"][0]], [Y["A none"]], **DOT)
hp, hw = A["hide_unverified"]["post"], A["hide_unverified"]["wire"]
ci(hp, Y["A hide"])
ci(hw, Y["A hide"])
ax.plot([hp[0], hw[0]], [Y["A hide"]] * 2, color=C_GEM, lw=1.5, zorder=3)
ax.plot([hp[0]], [Y["A hide"]], **DOT)
ax.plot([hw[0]], [Y["A hide"]], **RING)

# (b) half-reckless Gemma, crypto
ax.plot([B["none"]["post"][0]] * 2, [Y["B none"] + 0.3, Y["B steer"] - 0.36], **REF)
for k, yk in [("none", "B none"), ("leverage_cap", "B lev"), ("steer_cautious", "B steer")]:
    ci(B[k]["post"], Y[yk])
    ax.plot([B[k]["post"][0]], [Y[yk]], **DOT)

ticks = ["A none", "A hide", "B none", "B lev", "B steer"]
ax.set_yticks([Y[t] for t in ticks])
ax.set_yticklabels(["none", "hide posts", "none", "leverage cap", "risk dial −1"])
ax.tick_params(axis="y", length=0, pad=5)
ax.spines["left"].set_visible(False)
ax.set_ylim(-6.85, 0.42)
ax.set_xlim(-0.88, 3.9)
ax.set_xticks([0, 1, 2, 3])
ax.set_xlabel("excess overshoot (%)", labelpad=2)

# group headers: left-aligned at the label column, on rows that hold no data
HX = 0.522                                           # figure fraction, just right of the left panel
tr = blended_transform_factory(fig.transFigure, ax.transData)
heads = [ax.text(HX, Y["hA"], "Gemma, equity", ha="left", va="center", transform=tr, gid="header", **TXT),
         ax.text(HX, Y["hB"], "half reckless, crypto", ha="left", va="center", transform=tr, gid="header", **TXT)]


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
                            hits.append((t.get_text(), "marker", tuple(np.round(
                                ax.transData.inverted().transform((x, y)), 2))))
                if ln.get_linestyle() not in ("None", "none", ""):
                    for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
                        ts = np.linspace(0, 1, 200)
                        if any(b.contains(x0 + s * (x1 - x0), y0 + s * (y1 - y0)) for s in ts):
                            hits.append((t.get_text(), "line", ln.get_color()))
                            break
            if t.get_gid() == "header":              # headers start in the label column by design
                lab = [tl.get_window_extent(r) for a in fig.axes for tl in a.get_yticklabels() + a.get_xticklabels()
                       if tl.get_text().strip()]
                if any(b.overlaps(x) for x in lab) or b.x1 > ax.bbox.x1:
                    hits.append((t.get_text(), "header hits a tick label or leaves the figure"))
                continue
            if not ax.bbox.contains(b.x0 + pad_px, b.y0 + pad_px) or not ax.bbox.contains(b.x1 - pad_px, b.y1 - pad_px):
                hits.append((t.get_text(), "outside axes"))
    print(f"[fig_market] text over lines/markers or outside axes = {len(hits)} {hits}")
    return hits


n = check(fig, "fig_market")
mark_hits(fig)
fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=300)
tb = fig.get_tightbbox(fig.canvas.get_renderer())
w_in, h_in = tb.width + 0.1, tb.height + 0.1
k = 3.33 / w_in
print(f"saved {w_in:.2f} x {h_in:.2f} in (aspect {w_in / h_in:.2f}); at columnwidth 3.33 in: scale {k:.3f}, "
      f"ticks {19 * k:.1f} pt, axis labels {20.5 * k:.1f} pt, notes {ANN * k:.1f} pt, height {h_in * k:.2f} in")
