"""Figure 1: Stampede at a glance (figure* at \\textwidth, top of page 2).

Message: in Stampede one headline reaches every AI trading agent at once.

Left, the system. One headline (true or false) fans out to every agent in the same minute. The agents are many
copies of a few models: Gemma-4-12B steered by the risk dial (clay; the shades stand for different dial settings),
Claude and GPT replayed from their response tables (white diamonds, as in the other figures), and any agent that
connects over MCP (grey squares). Seven glyphs a row, not eight: three rows of eight would equal the episode's 24
agents and suggest that the episode mixes the three kinds. Their orders fan back in to one order book with leverage
and forced liquidation, the mirror of the headline's fan-out. The schematic spans the figure's full height (no
empty bands above or below its boxes). The depth chart drawn in the book is the episode's own book at minute 0:
resting size summed outward from the spread, bids to the left and asks to the right, one step per price level (ten
a side).

Right, one recorded episode (results/episodes/<EPISODE>.json.gz): 24 agents replaying Claude Haiku's response
table, equity regime, agent capital 2x the book's depth, seed 0. A false headline from a verified-looking wire
arrives at minute 15 and the correction at minute 30. Top: price and fair value, the gap below fair value shaded,
its deepest point labelled. Bottom: the agents' mean exposure (a frame is the state at the end of a minute, so each
value is drawn as a step held for that minute), its low point labelled.

Every number drawn, and every number in the caption, is computed from the episode and asserted below. The script
also asserts the page budget: the figure is never taller at \\textwidth than version 1
(paper/figs/old/fig_overview_v1.pdf).

    python scripts/figs/fig_overview.py                    # the paper's episode
    python scripts/figs/fig_overview.py --episode ep_...   # any other recorded episode with a headline

Writes paper/figs/fig_overview.{pdf,png} (300 dpi) and paper/figs/fig_overview_caption.{md,tex}.
"""
import argparse
import gzip
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/.claude/skills/paper-figures/assets"))
import style as S  # noqa: E402
from overlap_check import check  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex, to_rgb  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EP_DIRS = [os.path.join(ROOT, "results/episodes"), os.path.join(ROOT, "results/site/episodes")]
EPISODE = "ep_6694b7512d3f"     # 24 agents on Claude Haiku's response table, equity, false wire, capital 2x, seed 0
OUT = os.path.join(ROOT, "paper/figs/fig_overview")
TEXTWIDTH_PT = 506.295          # acmart sigconf \textwidth
V1_PT = (1009.44, 339.646)      # MediaBox of version 1 (paper/figs/old/fig_overview_v1.pdf): the height budget
MAX_H_IN = 2.36                 # and the hard cap on the printed height at \textwidth (in)

SHORT_NAMES = {"claude-haiku": "Haiku", "claude-sonnet": "Sonnet", "claude-opus": "Opus", "claude-fable": "Fable",
               "gpt-6-luna": "GPT-6 Luna", "gpt-6-sol": "GPT-6 Sol"}
MAX_WORDS = 50                  # the skill's short caption
HEADLINE = {  # hcode -> (label in the figure, phrase in the caption)
    "fake_ver": ("false headline", "a false wire headline"),
    "fake_unv": ("false headline", "a false unverified post"),
    "true_ver": ("true headline", "a true wire headline"),
}


# ================================================================ episode
def _read(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def resolve_episode(arg):
    if os.path.exists(arg):
        return arg
    for d in EP_DIRS:
        for ext in ("", ".json.gz", ".json"):
            p = os.path.join(d, os.path.basename(arg) + ext)
            if os.path.exists(p):
                return p
    sys.exit(f"episode not found: {arg}")


def load(path):
    d = _read(path)
    c, m, fr = d["config"], d["metrics"], d["frames"]
    t = np.array([f["t"] for f in fr], float)
    E = np.array([[a["exposure"] for a in f["agents"]] for f in fr], float)
    groups = c.get("groups") or []
    return dict(id=d["id"], path=path, cfg=c, met=m, frames=fr, t=t,
                price=np.array([f["price"] for f in fr], float),
                fund=np.array([f["fundamental"] for f in fr], float),
                E=E, exp=E.mean(axis=1), n_agents=E.shape[1],
                shock_t=c.get("shock_t"), corr_t=c.get("correction_t"), groups=groups)


def numbers(ep):
    """The episode's numbers that the figure and the caption state."""
    t, p, f, e, m = ep["t"], ep["price"], ep["fund"], ep["exp"], ep["met"]
    ts, tc = ep["shock_t"], ep["corr_t"]
    assert ts is not None and tc is not None and tc > ts, "need a headline and a correction"
    win = (t >= ts) & (t < tc)
    pre = (t >= ts - 5) & (t < ts)
    dev = (p - f) / m["p0"]                       # signed distance from fair value, as a share of the opening price
    i_gap = int(np.flatnonzero(t >= ts)[np.argmin(dev[t >= ts])])
    i_min = int(np.flatnonzero(win)[np.argmin(ep["exp"][win])])
    gap = -float(dev[i_gap])
    assert abs(gap - m["overshoot"]) < 1e-6, (gap, m["overshoot"])   # same definition as the paper's overshoot
    return dict(n=ep["n_agents"], e_pre=float(e[pre].mean()), e_min=float(e[i_min]), t_min=float(t[i_min]),
                gap=gap, t_gap=float(t[i_gap]), p_gap=float(p[i_gap]),
                below_until_correction=bool((p[win] < f[win]).all()),
                n_liq=int(m.get("n_liquidations") or 0))


# ================================================================ geometry (display coordinates)
def seg_hits_box(p0, p1, b):
    """Liang-Barsky: does the segment p0-p1 intersect Bbox b?"""
    x0, y0 = p0
    dx, dy = p1[0] - x0, p1[1] - y0
    u0, u1 = 0.0, 1.0
    for pp, q in ((-dx, x0 - b.x0), (dx, b.x1 - x0), (-dy, y0 - b.y0), (dy, b.y1 - y0)):
        if pp == 0:
            if q < 0:
                return False
        else:
            r = q / pp
            if pp < 0:
                u0 = max(u0, r)
            else:
                u1 = min(u1, r)
            if u0 > u1:
                return False
    return True


def poly_hits_box(xy, b):
    pts = np.asarray(xy, float)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    return any(seg_hits_box(pts[i], pts[i + 1], b) for i in range(len(pts) - 1))


def fill_hits_box(path_disp, b, n=9):
    xs, ys = np.linspace(b.x0, b.x1, n), np.linspace(b.y0, b.y1, n)
    return path_disp.contains_points(np.array([(x, y) for x in xs for y in ys])).any()


def line_xy(ln):
    return ln.get_transform().transform(np.column_stack([np.asarray(ln.get_xdata(), float),
                                                         np.asarray(ln.get_ydata(), float)]))


# ================================================================ style
S.apply_style()
FS_BOX, FS_SUB = 22.0, 22.0                  # schematic: names, second lines ("order book" sets the box width)
FS_HEAD, FS_HEAD_SUB = 24.0, 22.0            # the headline card: the protagonist, sitting in whitespace
FS_TICK, FS_LAB, FS_NOTE = 20.5, 22.0, 22.0  # episode: ticks, axis labels, notes
FAN = 0.42                                   # fan-out from the card and fan-in to the book: share of the row offsets
LW_BOX, LW_ARR = 1.4, 1.8
plt.rcParams.update({"xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "axes.labelsize": FS_LAB})
W, H = 14.0, 4.60                            # design size (in); \textwidth scales it by ~0.5
GAP_C = "#E6DFD5"                            # light neutral: gap below fair value, exposure area (not a data hue)
DEPTH_C = "#BFB8AF"                          # the grey of the agents' squares, lightened: the book glyph stays quiet
EVENT = dict(color=S.SUBINK, lw=1.4, ls=(0, (1.2, 2.2)))


def txt(ax, x, y, s, fs=FS_BOX, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    return ax.text(x, y, s, fontsize=fs, color=S.INK, zorder=6, **kw)


def tsize(fig, s, fs):
    """Width and height (in) of a text at font size fs."""
    t = fig.text(0, 0, s, fontsize=fs)
    b = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return b.width / fig.dpi, b.height / fig.dpi


def rbox(ax, x0, y0, x1, y1, fc=S.PAPER, ec=S.SUBINK, lw=LW_BOX, r=0.1, z=1):
    # unclipped: a box on the schematic axes' edge (the card at x = 0, the book at the right) keeps its full stroke
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z, clip_on=False)
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, lw=LW_ARR, ms=17):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=S.SUBINK, shrinkA=0, shrinkB=0,
                        zorder=3, joinstyle="miter", clip_on=False)
    ax.add_patch(a)
    return a


def mix(c1, c2, w):
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return to_hex((1 - w) * a + w * b)


# ================================================================ left: the system
def draw_schematic(fig, ax, ep, y_bot, y_top):
    """Draws in inch units with its left edge at x = 0; returns the right edge (in) and the rows' glyph heights."""
    ax.set_axis_off()
    PADX, PADY = 0.2, 0.2
    rows = ["steered Gemma", "Claude, GPT", "any MCP agent"]
    lab_w = max(tsize(fig, s, FS_BOX)[0] for s in rows)
    lab_h = tsize(fig, "Gg", FS_BOX)[1]
    NG, G = 7, 0.27                                   # glyphs per row (3 x 7, not the episode's 24), size (in)
    pitch = (lab_w - G) / (NG - 1)
    blk = lab_h + 0.06 + G                            # a row: name above a line of glyphs
    gap = (y_top - y_bot - 2 * PADY - 3 * blk) / 2
    assert gap > 0.25, f"rows too tight ({gap:.2f} in)"
    gy = [y_top - PADY - blk + G / 2 - i * (blk + gap) for i in range(3)]   # glyph-line centres, top to bottom

    # headline card, centred on the middle row; tall enough that its fan-out leaves no large empty corners
    cw = max(tsize(fig, "headline", FS_HEAD)[0], tsize(fig, "true or false", FS_HEAD_SUB)[0]) + 2 * 0.17
    ch = 2.3
    hx0, hx1, hy = 0.0, cw, gy[1]
    rbox(ax, hx0, hy - ch / 2, hx1, hy + ch / 2)
    txt(ax, cw / 2, hy + 0.26, "headline", fs=FS_HEAD)
    txt(ax, cw / 2, hy - 0.27, "true or false", fs=FS_HEAD_SUB)

    # agents
    ax0 = hx1 + 0.66
    ax1 = ax0 + PADX + lab_w + PADX
    rbox(ax, ax0, y_bot, ax1, y_top)
    gx = ax0 + PADX + G / 2 + pitch * np.arange(NG)
    for i, s in enumerate(rows):
        txt(ax, ax0 + PADX, gy[i] + G / 2 + 0.06, s, ha="left", va="bottom")
    # steered Gemma: one model, many dial settings (shades of the risk-dial clay)
    shades = [mix(S.CLAY_SOFT, S.CLAY_DK, w) for w in np.linspace(0, 1, NG)]
    for x, c in zip(gx, shades):
        ax.plot([x], [gy[0]], "o", ms=G * 72 * 0.98, color=c, mec="white", mew=0.8, zorder=5)
    ax.plot(gx, [gy[1]] * NG, ls="none", marker="D", ms=G * 72 * 0.74, mfc="white", mec=S.INK, mew=1.5, zorder=5)
    ax.plot(gx, [gy[2]] * NG, ls="none", marker="s", ms=G * 72 * 0.82, color=S.GRAY, mec="white", mew=0.8, zorder=5)
    # the headline reaches every row at once (fan-out)
    for i in range(3):
        arrow(ax, (hx1, hy + (gy[i] - hy) * FAN), (ax0, gy[i]))

    # one order book for everyone: the rows' orders fan back in, the mirror of the fan-out, and land on the
    # depth chart rather than beside the book's text
    bx0 = ax1 + 0.62
    for i in range(3):
        arrow(ax, (ax1, gy[i]), (bx0, hy + (gy[i] - hy) * FAN))
    l1, l2 = "leverage,", "liquidation"
    bw = max(tsize(fig, "order book", FS_BOX)[0], tsize(fig, l2, FS_SUB)[0]) + 2 * PADX
    bx1 = bx0 + bw
    rbox(ax, bx0, y_bot, bx1, y_top)
    txt(ax, bx0 + PADX, y_top - PADY, "order book", va="top", ha="left")
    sub_h = tsize(fig, "lg", FS_SUB)[1]
    txt(ax, bx0 + PADX, y_bot + PADY + sub_h * 1.12, l1, fs=FS_SUB, ha="left", va="bottom")
    txt(ax, bx0 + PADX, y_bot + PADY, l2, fs=FS_SUB, ha="left", va="bottom")
    d_top = y_top - PADY - lab_h - 0.42
    d_bot = y_bot + PADY + 2.12 * sub_h + 0.36
    draw_depth(ax, ep, bx0 + PADX, bx1 - PADX, d_bot, d_top)
    return bx1, gy


def draw_depth(ax, ep, x0, x1, y0, y1):
    """The episode's book at minute 0 as a depth chart: resting size summed outward from the spread, bids to the
    left of the mid and asks to the right, one step per price level (ten a side)."""
    f0 = ep["frames"][0]
    bids = sorted(f0["bids"], key=lambda r: -r[0])        # best bid first
    asks = sorted(f0["asks"], key=lambda r: r[0])         # best ask first
    assert bids and asks and bids[0][0] < asks[0][0], "crossed book"
    cb, ca = np.cumsum([r[1] for r in bids]), np.cumsum([r[1] for r in asks])
    top = max(cb[-1], ca[-1])
    mid, sp = (x0 + x1) / 2, 0.035                        # half the drawn spread (in)
    half = (x1 - x0) / 2 - sp
    for cum, sgn in ((cb, -1), (ca, 1)):
        n = len(cum)
        edges = mid + sgn * (sp + half * np.arange(n + 1) / n)
        hts = y0 + (y1 - y0) * cum / top
        poly = [(edges[0], y0)]
        for k in range(n):
            poly += [(edges[k], hts[k]), (edges[k + 1], hts[k])]
        poly += [(edges[-1], y0)]
        ax.add_patch(Polygon(poly, closed=True, fc=DEPTH_C, ec="none", zorder=3))
    ax.plot([x0, x1], [y0, y0], color=S.SUBINK, lw=1.2, zorder=4, solid_capstyle="butt")


# ================================================================ right: one recorded episode
def draw_episode(axP, axE, ep):
    t, p, f, e = ep["t"], ep["price"], ep["fund"], ep["exp"]
    ts, tc = ep["shock_t"], ep["corr_t"]
    after = t >= ts
    axP.fill_between(t, p, f, where=after & (p < f), interpolate=True, color=GAP_C, lw=0, zorder=1)
    axP.plot(t, f, color=S.SUBINK, lw=2.2, ls=(0, (4, 2.5)), zorder=3)
    axP.plot(t, p, color=S.INK, lw=2.7, zorder=4)
    lo, hi = float(np.min(np.r_[p, f])), float(np.max(np.r_[p, f]))
    axP.set_ylim(lo - 0.1 * (hi - lo), hi + 0.06 * (hi - lo))
    ticks = np.arange(np.ceil(lo), np.floor(hi) + 1)
    axP.yaxis.set_major_locator(FixedLocator(ticks))
    axP.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    axP.set_ylabel("price")
    axP.tick_params(axis="x", labelbottom=False)

    # a frame is the state at the end of minute t, held from t to t + 1
    axE.fill_between(t, 0, e, step="post", color=GAP_C, lw=0, zorder=2)
    axE.step(t, e, where="post", color=S.INK, lw=2.0, zorder=3)
    axE.set_ylim(0, max(1.0, float(np.max(e))) * 1.14)
    axE.yaxis.set_major_locator(FixedLocator([0, 1]))
    axE.set_ylabel("exposure")
    axE.xaxis.set_major_locator(FixedLocator([t[0], ts, tc]))      # ticks at the start and at the two events
    for ax in (axP, axE):
        ax.set_xlim(t[0], t[-1])
        for x in (ts, tc):
            ax.axvline(x, zorder=0, **EVENT)


def place_event_labels(fig, axP, ep, y_in):
    """'false headline' left of its line, 'correction' right of its line, both above the price axes; each event
    line is extended up to the middle of its label."""
    r = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    ax_bb = axP.get_window_extent(r)
    events = [(ep["shock_t"], HEADLINE.get(ep["cfg"].get("hcode"), ("headline", ""))[0], "right"),
              (ep["corr_t"], "correction", "left")]
    dx = 7 / fig.dpi / W
    out = []
    for x, s, ha in events:
        xf = to_fig.transform((axP.transData.transform((x, 0))[0], 0))[0]
        tt = fig.text(xf + (dx if ha == "left" else -dx), y_in / H, s, fontsize=FS_NOTE, ha=ha, va="bottom",
                      color=S.INK)
        tb = tt.get_window_extent(r)
        y0 = to_fig.transform((0, ax_bb.y1))[1]
        y1 = to_fig.transform((0, tb.y0 + tb.height * 0.45))[1]
        fig.add_artist(plt.Line2D([xf, xf], [y0, y1], **EVENT))
        out.append(tt)
    return out


def is_free(fig, ax, b, avoid):
    r = fig.canvas.get_renderer()
    a = ax.get_window_extent(r)
    if b.x0 < a.x0 + 3 or b.x1 > a.x1 - 3 or b.y0 < a.y0 + 2 or b.y1 > a.y1 - 2:
        return False
    if any(ln.get_linestyle() not in ("None", "none", "") and poly_hits_box(line_xy(ln), b) for ln in ax.lines):
        return False
    if any(fill_hits_box(pth.transformed(c.get_transform()), b) for c in ax.collections for pth in c.get_paths()):
        return False
    return not any(b.overlaps(o.get_window_extent(r)) for o in avoid)


def place(fig, ax, s, cands, avoid=(), pad_px=4):
    """First free spot among candidates (x, y, ha, va, dx_pt, dy_pt), data coordinates plus an offset in points."""
    r = fig.canvas.get_renderer()
    for x, y, ha, va, dxp, dyp in cands:
        xd, yd = ax.transData.transform((x, y))
        tt = ax.text(0, 0, s, fontsize=FS_NOTE, ha=ha, va=va, color=S.INK, zorder=6)
        tt.set_position(ax.transData.inverted().transform((xd + dxp * fig.dpi / 72, yd + dyp * fig.dpi / 72)))
        if is_free(fig, ax, tt.get_window_extent(r).padded(pad_px), avoid):
            return tt
        tt.remove()
    return None


# ================================================================ checks
def text_hits(fig):
    """Text over any line, arrow, fill, marker, or crossing a box edge (the skill's checker covers text and ticks)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    hits = []
    texts = [t for ax in fig.axes for t in ax.texts] + list(fig.texts)
    texts += [lab for ax in fig.axes if ax.axison for lab in (ax.xaxis.label, ax.yaxis.label)]
    ticks = [t for ax in fig.axes if ax.axison for t in ax.get_xticklabels() + ax.get_yticklabels()
             if t.get_visible() and t.get_text().strip()]
    for ax in fig.axes:   # axis labels against tick labels (the skill's checker skips this pair)
        if ax.axison:
            for lab in (ax.xaxis.label, ax.yaxis.label):
                if lab.get_text().strip():
                    hits += [(lab.get_text(), "tick " + tk.get_text()) for tk in ticks
                             if lab.get_window_extent(r).overlaps(tk.get_window_extent(r))]
    texts += ticks
    fb = fig.bbox
    for t in texts:
        if not t.get_text().strip():
            continue
        b = t.get_window_extent(r).padded(2)
        if b.x0 < fb.x0 or b.x1 > fb.x1 or b.y0 < fb.y0 or b.y1 > fb.y1:
            hits.append((t.get_text(), "outside the figure"))
        for ax in fig.axes:
            for ln in ax.lines:
                xy = line_xy(ln)
                if ln.get_linestyle() not in ("None", "none", "") and poly_hits_box(xy, b):
                    hits.append((t.get_text(), "line"))
                if ln.get_marker() not in (None, "None", "none", ""):
                    rad = ln.get_markersize() * fig.dpi / 72 / 2
                    if any(b.padded(rad).contains(x, y) for x, y in xy):
                        hits.append((t.get_text(), "marker"))
            for pa in ax.patches:
                path = pa.get_path().transformed(pa.get_transform())
                if isinstance(pa, FancyArrowPatch):
                    if any(poly_hits_box(pl, b) for pl in path.to_polygons(closed_only=False)):
                        hits.append((t.get_text(), "arrow"))
                elif isinstance(pa, FancyBboxPatch):
                    pb = Bbox.from_extents(*path.vertices.min(0), *path.vertices.max(0))
                    inside = pb.x0 <= b.x0 and b.x1 <= pb.x1 and pb.y0 <= b.y0 and b.y1 <= pb.y1
                    if pb.overlaps(b) and not inside:
                        hits.append((t.get_text(), "box edge"))
                elif pa.get_facecolor()[3] > 0 and fill_hits_box(path, b):
                    hits.append((t.get_text(), "patch fill"))
            for c in ax.collections:
                if any(fill_hits_box(pth.transformed(c.get_transform()), b) for pth in c.get_paths()):
                    hits.append((t.get_text(), "fill"))
        for ln in fig.lines:
            if poly_hits_box(line_xy(ln), b):
                hits.append((t.get_text(), "figure line"))
    print(f"[fig_overview] text over lines/arrows/fills/markers/box edges = {len(hits)} {hits}")
    return hits


def frame_fill(fig):
    """Empty bands (in) below the lowest and above the highest text; a text box includes the font's descent."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for ax in fig.axes for t in ax.texts] + list(fig.texts)
    texts += [t for ax in fig.axes if ax.axison for t in ax.get_xticklabels() + ax.get_yticklabels()
              if t.get_visible() and t.get_text().strip()]
    bbs = [t.get_window_extent(r) for t in texts if t.get_text().strip()]
    lo, hi = min(b.y0 for b in bbs) / fig.dpi, H - max(b.y1 for b in bbs) / fig.dpi
    print(f"[fig_overview] empty band below the lowest text {lo:.3f} in, above the highest {hi:.3f} in")
    return lo, hi


def pdf_size_pt(path):
    with open(path, "rb") as fh:
        m = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", fh.read())
    x0, y0, x1, y1 = map(float, m.groups())
    return x1 - x0, y1 - y0


# ================================================================ caption
def captions(ep, num):
    # short model names, as the paper's text uses them ("Haiku and Sonnet"); the figure's row says "Claude, GPT"
    pop = " and ".join(SHORT_NAMES.get((g.get("surrogate") or g.get("name") or ""), g.get("name", "")) for g in
                       ep["groups"])
    lead = "Stampede: one headline reaches every AI trading agent at once."
    left = "Left: steered, commercial and MCP agents share one leveraged order book."
    hl = HEADLINE.get(ep["cfg"].get("hcode"), ("headline", "a headline"))[1]
    who = f"{num['n']} agents replaying {pop}'s policy" if all(g.get("surrogate") for g in ep["groups"]) else \
        f"{num['n']} {pop} agents"
    price = (f"the price falls up to {100 * num['gap']:.1f}% below fair value"
             + (" until the correction" if num["below_until_correction"] else ""))
    right = (f"Right: after {hl}, {who} cut exposure from {num['e_pre']:.1f} to {num['e_min']:.2f}; {price}.")
    md_body = f"**{lead}** {left} {right}"
    tex_body = f"\\textbf{{{lead}}} {left} {right}".replace("%", r"\%")
    md = (f"<!-- Rendered by scripts/figs/fig_overview.py from results/episodes/{ep['id']}.json.gz -->\n\n"
          f"{md_body}\n")
    tex = (f"% Figure 1: Stampede at a glance. Rendered by scripts/figs/fig_overview.py from "
           f"results/episodes/{ep['id']}.json.gz.\n"
           "\\begin{figure*}[t]\n  \\centering\n"
           "  \\includegraphics[width=\\textwidth]{figs/fig_overview}\n"
           f"  \\caption{{{tex_body}}}\n"
           "  \\label{fig:arch}\n\\end{figure*}\n")
    words = len(md_body.replace("**", "").split())
    for s in (md, tex):
        assert "\u2014" not in s and "\u2013" not in s and "--" not in s.split("\n", 1)[1], "dash in caption"
    with open(OUT + "_caption.md", "w") as fh:
        fh.write(md)
    with open(OUT + "_caption.tex", "w") as fh:
        fh.write(tex)
    print(f"[fig_overview] caption ({words} words): {md_body}")
    return words


# ================================================================ main
def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default=EPISODE, help=f"episode id or path (default {EPISODE})")
    ap.add_argument("--out", default=None, help="output stem (default paper/figs/fig_overview)")
    args = ap.parse_args()
    if args.out:
        OUT = args.out
    ep = load(resolve_episode(args.episode))
    num = numbers(ep)
    print(f"[fig_overview] episode {ep['id']} ({os.path.relpath(ep['path'], ROOT)}): hcode={ep['cfg'].get('hcode')}, "
          f"regime={ep['cfg'].get('regime')}, capital={ep['cfg'].get('adoption')}, seed={ep['cfg'].get('seed')}; "
          f"numbers {json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in num.items()})}")
    if ep["id"] == EPISODE:   # the numbers the paper states for this episode
        assert num["n"] == 24 and f"{num['e_pre']:.1f}" == "1.0" and f"{num['e_min']:.2f}" == "0.10"
        assert f"{100 * num['gap']:.1f}" == "2.3" and num["below_until_correction"]

    fig = plt.figure(figsize=(W, H))
    # chart frame first: the two axes leave just enough room below for the minute labels and above for the event
    # labels (frame_fill checks it); the schematic's boxes span that whole height, from the minute labels' foot to
    # the event labels' top, so no empty band runs above or below them
    e_y0, e_h, gap, p_top = 0.425, 1.0, 0.16, 4.135
    sw_guess = 6.9
    axS = fig.add_axes([0, 0, sw_guess / W, 1])
    axS.set_xlim(0, sw_guess)
    axS.set_ylim(0, H)
    sx1, _ = draw_schematic(fig, axS, ep, y_bot=0.06, y_top=H - 0.05)
    # fit the schematic axes to what was drawn (1 data unit = 1 inch)
    axS.set_position([0, 0, sx1 / W, 1])
    axS.set_xlim(0, sx1)

    px0, px1 = sx1 + 1.32, W - 0.06
    axE = fig.add_axes([px0 / W, e_y0 / H, (px1 - px0) / W, e_h / H])
    axP = fig.add_axes([px0 / W, (e_y0 + e_h + gap) / H, (px1 - px0) / W, (p_top - e_y0 - e_h - gap) / H],
                       sharex=axE)
    draw_episode(axP, axE, ep)
    ylab_x = (sx1 + 0.58 - px0) / (px1 - px0)     # both y labels at one x, whatever the tick labels' widths
    for ax in (axP, axE):
        ax.yaxis.set_label_coords(ylab_x, 0.5)
    fig.canvas.draw()

    # the minute axis: tick labels, and the unit after the last one (saves a row of height)
    r = fig.canvas.get_renderer()
    tick_last = [t for t in axE.get_xticklabels() if t.get_text().strip()][-1]
    tb = tick_last.get_window_extent(r)
    unit = fig.text(axE.get_window_extent(r).x1 / fig.bbox.width, (tb.y0 + tb.y1) / 2 / fig.bbox.height, "minute",
                    fontsize=FS_TICK, ha="right", va="center", color=S.INK)
    ub = unit.get_window_extent(r)
    print(f"[fig_overview] schematic right edge {sx1:.2f} in; gap last tick to minute = {(ub.x0 - tb.x1) / fig.dpi:.2f} in")
    assert ub.x0 - tb.x1 > 0.25 * fig.dpi, "minute too close to the last tick label"

    ev = place_event_labels(fig, axP, ep, y_in=p_top + 0.13)
    t, f = ep["t"], ep["fund"]
    # 'fair value' just below (or above) its line, as far right as there is room
    late = np.flatnonzero(t >= ep["corr_t"] + 3)[::-1]
    fv = place(fig, axP, "fair value", [(t[i], f[i], ha, va, 0, dy) for va, dy in (("top", -7), ("bottom", 7))
                                        for i in late for ha in ("right", "center")])
    assert fv is not None, "no room for 'fair value'"
    # the deepest gap, right of the trough
    g = place(fig, axP, f"\u2212{100 * num['gap']:.1f}%",
              [(num["t_gap"], num["p_gap"], "left", va, dx, dy) for dx in (9, 13, 17, 21, 25)
               for va, dy in (("center", 0), ("center", -4), ("top", 4), ("center", 4))])
    assert g is not None, "no room for the gap label"
    # the exposure low, above the low step, left of the correction line
    lo = place(fig, axE, f"{num['e_min']:.2f}",
               [(num["t_min"] + 0.5, num["e_min"], ha, "bottom", dx, dy) for dy in (6, 10, 14, 18)
                for ha, dx in (("right", -5), ("right", -8), ("right", -2), ("center", 0))])
    assert lo is not None, "no room for the exposure label"

    n = check(fig, "fig_overview")
    hits = text_hits(fig)
    band_lo, band_hi = frame_fill(fig)
    assert band_lo < 0.08 and band_hi < 0.08, "an empty band runs below or above the content"
    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=300)
    plt.close(fig)

    w_pt, h_pt = pdf_size_pt(OUT + ".pdf")
    k = TEXTWIDTH_PT / w_pt
    h_paper, h_v1 = h_pt * k, V1_PT[1] * TEXTWIDTH_PT / V1_PT[0]
    print(f"[fig_overview] saved {w_pt / 72:.3f} x {h_pt / 72:.3f} in (aspect {w_pt / h_pt:.3f}; v1 "
          f"{V1_PT[0] / V1_PT[1]:.3f}); at \\textwidth: scale {k:.3f}, height {h_paper:.1f} pt = {h_paper / 72:.3f} in "
          f"(v1 {h_v1:.1f} pt); print sizes: headline card {FS_HEAD * k:.1f}/{FS_HEAD_SUB * k:.1f} pt, schematic "
          f"{FS_BOX * k:.1f}/{FS_SUB * k:.1f} pt, ticks {FS_TICK * k:.1f} pt, axis labels {FS_LAB * k:.1f} pt, "
          f"notes {FS_NOTE * k:.1f} pt")
    assert h_paper <= h_v1 + 1e-6, "taller than version 1 at \\textwidth"
    assert h_paper / 72 <= MAX_H_IN + 1e-9, f"taller than {MAX_H_IN} in at \\textwidth"
    words = captions(ep, num)
    if n or hits or words > MAX_WORDS:
        sys.exit("overlaps found or caption too long")


if __name__ == "__main__":
    main()
