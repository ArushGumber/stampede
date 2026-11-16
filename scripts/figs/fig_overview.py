"""Figure 1: Stampede at a glance (full text width, figure*).

Left: the system as a schematic. One headline (true or false; verified wire or unverified post) reaches every agent
as news; the agents (a steered open model, commercial models replayed from response tables, any agent over MCP)
send orders to the exchange (order book with leverage and liquidation, background traders, safeguards) and read
prices back; the exchange's outputs are served to the web interface.
Right: one recorded episode. Price and fundamental value per simulated minute, the headline and the correction,
and, in a thin strip, the LLM agents' mean exposure.

Usage
    python scripts/figs/fig_overview.py                       # default episode (see pick_default)
    python scripts/figs/fig_overview.py --episode ep_e5ce135241c5
    python scripts/figs/fig_overview.py --episode results/episodes/ep_e5ce135241c5.json.gz

Every position on the right is computed from the episode (label placement searches for free space and the checks
below fail loudly), so the figure can be re-rendered for any episode.
Writes paper/figs/fig_overview.{pdf,png} (300 dpi) and paper/figs/fig_overview_caption.{tex,md}.
"""
import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/.claude/skills/paper-figures/assets"))
import style as S  # noqa: E402
from overlap_check import check  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EP_DIRS = [os.path.join(ROOT, "results/episodes"), os.path.join(ROOT, "results/site/episodes")]
OUT = os.path.join(ROOT, "paper/figs/fig_overview")
TEXTWIDTH_IN = 7.0     # acmart sigconf \textwidth (506 pt)


# ================================================================ episode loading
def _read(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def resolve_episode(arg):
    if os.path.exists(arg):
        return arg
    name = os.path.basename(arg)
    for d in EP_DIRS:
        for ext in ("", ".json.gz", ".json"):
            p = os.path.join(d, name + ext)
            if os.path.exists(p):
                return p
    sys.exit(f"episode not found: {arg}")


def pick_default():
    """Largest overshoot among false-headline equity episodes at adoption 2.0 (ties: first id)."""
    best = None
    for p in sorted(glob.glob(os.path.join(EP_DIRS[0], "*.json.gz"))):
        d = _read(p)
        c = d["config"]
        if c.get("hcode") in ("fake_ver", "fake_unv") and c.get("regime") == "equity" and c.get("adoption") == 2.0:
            os_ = d["metrics"]["overshoot"]
            if best is None or os_ > best[0] + 1e-12:
                best = (os_, p)
    return best[1]


POP_NAMES = {"claude-haiku": "Claude Haiku", "claude-sonnet": "Claude Sonnet", "claude-opus": "Claude Opus",
             "claude-fable": "Claude Fable", "gpt-6-luna": "GPT-6 Luna", "gpt-6-sol": "GPT-6 Sol",
             "gemma-4-12b": "Gemma-4-12B"}
HEADLINE = {  # hcode -> (label in figure, phrase for caption)
    "fake_ver": ("false headline", "a false wire headline"),
    "fake_unv": ("false headline", "a false unverified post"),
    "fake_unv_bull": ("false headline", "a false bullish post"),
    "true_ver": ("true headline", "a true wire headline"),
}


def is_bull(ep):
    return str(ep["cfg"].get("hcode", "")).endswith("_bull")


def group_is_steered(g):
    name = " ".join(str(g.get(k) or "") for k in ("name", "model", "surrogate")).lower()
    return "gemma" in name or any(abs(v or 0) > 0 for v in (g.get("alphas") or {}).values())


def group_label(g):
    key = (g.get("surrogate") or g.get("model") or g.get("name") or "").lower()
    for k, v in POP_NAMES.items():
        if key.startswith(k):
            disp = key[len(k):].strip("_").replace("_equity", "").replace("_crypto", "")
            return v + (f" ({disp})" if disp and disp != "base" else "")
    alphas = {k: v for k, v in (g.get("alphas") or {}).items() if v}
    if alphas:
        return "Gemma-4-12B (" + ", ".join(f"{k} {v:+g}" for k, v in alphas.items()) + ")"
    return g.get("name", "agents")


def load(path):
    d = _read(path)
    c, m, fr = d["config"], d["metrics"], d["frames"]
    t = np.array([f["t"] for f in fr], float)
    price = np.array([f["price"] for f in fr], float)
    fund = np.array([f["fundamental"] for f in fr], float)
    exp_ = np.array([np.mean([a["exposure"] for a in f["agents"]]) if f["agents"] else np.nan for f in fr])
    nliq = np.array([sum(bool(a.get("liquidated")) for a in f["agents"]) for f in fr])
    shock_t = c.get("shock_t")
    if shock_t is None:
        shock_t = next((s["t"] for s in c.get("shocks", []) if s.get("kind") != "correction"), None)
    corr_t = c.get("correction_t")
    if corr_t is None:
        corr_t = next((s["t"] for s in c.get("shocks", []) if s.get("kind") == "correction"), None)
    groups = c.get("groups") or []
    return dict(id=d.get("id", os.path.basename(path).split(".")[0]), cfg=c, met=m, t=t, price=price, fund=fund,
                exp=exp_, nliq=nliq, shock_t=shock_t, corr_t=corr_t, n_agents=len(d.get("agents", [])),
                steered=any(group_is_steered(g) for g in groups),
                pop=" and ".join(f"{g.get('n', '')} {group_label(g)}".strip() for g in groups) or c.get("pop", ""))


# ================================================================ geometry helpers (display coordinates)
def seg_hits_box(p0, p1, b):
    """Liang-Barsky: does segment p0-p1 intersect Bbox b?"""
    x0, y0 = p0
    dx, dy = p1[0] - x0, p1[1] - y0
    u0, u1 = 0.0, 1.0
    for p, q in ((-dx, x0 - b.x0), (dx, b.x1 - x0), (-dy, y0 - b.y0), (dy, b.y1 - y0)):
        if p == 0:
            if q < 0:
                return False
        else:
            r = q / p
            if p < 0:
                u0 = max(u0, r)
            else:
                u1 = min(u1, r)
            if u0 > u1:
                return False
    return True


def poly_hits_box(xy, b, closed=False):
    pts = np.asarray(xy)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if closed and len(pts):
        pts = np.vstack([pts, pts[:1]])
    return any(seg_hits_box(pts[i], pts[i + 1], b) for i in range(len(pts) - 1))


def fill_hits_box(path_disp, b, n=7):
    xs, ys = np.linspace(b.x0, b.x1, n), np.linspace(b.y0, b.y1, n)
    grid = np.array([(x, y) for x in xs for y in ys])
    return path_disp.contains_points(grid).any()


# ================================================================ style
S.apply_style()
FS_TICK, FS_LAB, FS_NOTE = 18.5, 19.5, 18.0      # right panel (pt before scaling to \textwidth)
FS_BOX, FS_SUB = 17.5, 16.5                      # schematic labels
LW_BOX, LW_ARR = 1.3, 1.6
plt.rcParams.update({"xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "axes.labelsize": FS_LAB})
W, H = 14.0, 4.6
SW = 7.7                                          # schematic width (in), drawn in inch units


def txt(ax, x, y, s, fs=FS_BOX, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    return ax.text(x, y, s, fontsize=fs, color=S.INK, zorder=6, **kw)


def rbox(ax, x0, y0, x1, y1, fc=S.PAPER, ec=S.SUBINK, lw=LW_BOX, r=0.09, z=1, ls="-"):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z, ls=ls)
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, lw=LW_ARR, color=S.SUBINK, cs="arc3,rad=0", ms=15):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color, shrinkA=0, shrinkB=0,
                        connectionstyle=cs, zorder=3, joinstyle="miter")
    ax.add_patch(a)
    return a


# ================================================================ left: schematic
def draw_schematic(ax, ep):
    ax.set_xlim(0, SW)
    ax.set_ylim(0, H)
    ax.set_axis_off()

    y_top, y_bot = 4.12, 1.22                     # agents and exchange boxes
    hx0, hx1 = 0.0, 1.45                          # headline
    ax0, ax1 = 2.05, 4.35                         # agents
    ex0, ex1 = 5.15, SW                           # exchange
    rows = [3.55, 2.65, 1.75]                     # agent row centres
    gap = 0.17                                    # half line spacing inside a row

    # group labels above the boxes
    txt(ax, (ax0 + ax1) / 2, y_top + 0.08, "agents", fs=FS_NOTE, va="bottom")
    txt(ax, (ex0 + ex1) / 2, y_top + 0.08, "exchange", fs=FS_NOTE, va="bottom")

    # headline
    hy = rows[1]
    rbox(ax, hx0, hy - 0.62, hx1, hy + 0.62)
    txt(ax, (hx0 + hx1) / 2, hy + 0.33, "headline", fs=FS_NOTE)
    txt(ax, (hx0 + hx1) / 2, hy - 0.04, "true or false", fs=FS_SUB)
    txt(ax, (hx0 + hx1) / 2, hy - 0.36, "wire or post", fs=FS_SUB)
    # news fans out to every agent
    for y in rows:
        arrow(ax, (hx1, hy + (y - hy) * 0.18), (ax0, y))
    txt(ax, (hx1 + ax0) / 2 - 0.08, rows[0] + 0.12, "news", fs=FS_SUB, va="bottom")

    # agents
    rbox(ax, ax0, y_bot, ax1, y_top)
    tx = ax0 + 0.18
    # (1) steered open model with a risk slider
    txt(ax, tx, rows[0] + gap, "Gemma-4-12B", ha="left")
    sy = rows[0] - gap - 0.02
    s0, s1, knob = tx + 0.6, ax1 - 0.2, 0.72
    txt(ax, tx, sy, "risk", fs=FS_SUB, ha="left")
    ax.plot([s0, s1], [sy, sy], color=S.SUBINK, lw=2.0, solid_capstyle="round", zorder=4)
    for xx in (s0, s1):
        ax.plot([xx, xx], [sy - 0.07, sy + 0.07], color=S.SUBINK, lw=1.4, zorder=4)
    ax.plot([s0 + knob * (s1 - s0)], [sy], "o", ms=11, color=S.CLAY, mec="white", mew=1.0, zorder=5)
    # (2) commercial models as measured policies
    txt(ax, tx, rows[1] + gap, "Claude, GPT", ha="left")
    txt(ax, tx, rows[1] - gap, "response tables", fs=FS_SUB, ha="left")
    ax.plot([ax1 - 0.3], [rows[1] + gap], "D", ms=8.5, mfc="white", mec=S.INK, mew=1.5, zorder=5)
    # (3) any external agent
    txt(ax, tx, rows[2] + gap, "any agent", ha="left")
    txt(ax, tx, rows[2] - gap, "over MCP", fs=FS_SUB, ha="left")

    # orders out, prices back
    ya, yb = rows[1] + 0.3, rows[1] - 0.3
    arrow(ax, (ax1, ya), (ex0, ya))
    arrow(ax, (ex0, yb), (ax1, yb))
    txt(ax, (ax1 + ex0) / 2, ya + 0.1, "orders", fs=FS_SUB, va="bottom")
    txt(ax, (ax1 + ex0) / 2, yb - 0.1, "prices", fs=FS_SUB, va="top")

    # exchange: order book (+ the episode's own depth at t = 0), traders, safeguards
    rbox(ax, ex0, y_bot, ex1, y_top)
    cx = ex0 + 0.18
    sep = dict(color=S.GRID, lw=1.6, zorder=2, solid_capstyle="butt")
    yy = y_top - 0.3
    txt(ax, cx, yy, "order book", ha="left")
    draw_depth(ax, ep, x0=cx + 1.36, x1=ex1 - 0.16, y0=yy - 0.19, y1=yy + 0.19)
    txt(ax, cx, yy - 0.34, "leverage, liquidation", fs=FS_SUB, ha="left")
    ax.plot([ex0 + 0.1, ex1 - 0.1], [yy - 0.585] * 2, **sep)
    txt(ax, cx, yy - 0.83, "background traders", ha="left")
    ax.plot([ex0 + 0.1, ex1 - 0.1], [yy - 1.075] * 2, **sep)
    txt(ax, cx, yy - 1.32, "safeguards", ha="left")
    for i, s in enumerate(("caps, delays,", "circuit breaker,", "provenance filter")):
        txt(ax, cx, yy - 1.65 - 0.315 * i, s, fs=FS_SUB, ha="left")

    # outputs to the web interface
    wy0, wy1 = 0.0, 0.7
    rbox(ax, 0.0, wy0, SW, wy1)
    arrow(ax, ((ex0 + ex1) / 2, y_bot), ((ex0 + ex1) / 2, wy1))
    wy = (wy0 + wy1) / 2
    txt(ax, 0.16, wy, "web", ha="left")
    items = ["price vs value", "excess overshoot", "liquidations", "report card"]
    x, cp = 0.78, 0.11
    r = ax.figure.canvas.get_renderer()
    for s in items:
        t = txt(ax, x + cp, wy, s, fs=FS_SUB, ha="left")
        b = t.get_window_extent(r).transformed(ax.transData.inverted())
        rbox(ax, x, wy - 0.2, b.x1 + cp, wy + 0.2, fc="white", ec=S.GRAY, lw=1.0, r=0.07, z=2)
        x = b.x1 + cp + 0.15
    return x - 0.15  # right end of the chips (checked against SW)


def draw_depth(ax, ep, x0, x1, y0, y1):
    """Cumulative depth of the episode's own book at t = 0: bids left of the mid, asks right."""
    d = _read(ep["path"])
    f0 = d["frames"][0]
    bids, asks = np.array(f0.get("bids") or [[0, 0]]), np.array(f0.get("asks") or [[0, 0]])
    cb, ca = np.cumsum(bids[:, 1]), np.cumsum(asks[:, 1])
    top = max(cb[-1], ca[-1]) or 1.0
    mid = (x0 + x1) / 2
    half = (x1 - x0) / 2 - 0.03
    nb, na = len(cb), len(ca)
    xb = mid - 0.03 - half * np.arange(1, nb + 1) / nb
    xa = mid + 0.03 + half * np.arange(1, na + 1) / na
    for xs, cs, sgn in ((xb, cb, -1), (xa, ca, 1)):
        edge = mid + sgn * 0.03
        px = np.concatenate([[edge], np.repeat(xs, 2)[:-1]])
        py = y0 + (y1 - y0) * np.repeat(cs / top, 2)
        poly = np.column_stack([np.concatenate([[edge], px, [xs[-1]]]),
                                np.concatenate([[y0], py, [y0]])])
        ax.add_patch(Polygon(poly, closed=True, fc=S.GRAY, ec="none", zorder=3))
    ax.plot([x0, x1], [y0, y0], color=S.SUBINK, lw=1.0, zorder=4)


# ================================================================ right: one recorded episode
def draw_episode(axP, axE, ep, agent_c):
    t, p, f, e = ep["t"], ep["price"], ep["fund"], ep["exp"]
    ts, tc = ep["shock_t"], ep["corr_t"]
    fill_c, line_c = agent_c

    # gap below fair value after the headline
    after = t >= (ts if ts is not None else t[0])
    wrong = (p > f) if is_bull(ep) else (p < f)        # the side the headline pushes the price to
    axP.fill_between(t, p, f, where=after & wrong, interpolate=True, color=fill_c, alpha=0.55, lw=0, zorder=1)
    axP.plot(t, f, color=S.SUBINK, lw=2.1, ls=(0, (4, 2.5)), zorder=3)
    axP.plot(t, p, color=S.INK, lw=2.6, zorder=4)

    lo, hi = np.nanmin(np.r_[p, f]), np.nanmax(np.r_[p, f])
    pad = 0.08 * (hi - lo)
    axP.set_ylim(lo - pad, hi + pad)
    axP.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    axP.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.10g}".replace("-", "\u2212")))
    axP.set_ylabel("price", labelpad=6)
    axP.tick_params(axis="x", labelbottom=False)

    # a frame is the state at the end of minute t, so each value is held from t to t + 1
    axE.fill_between(t, 0, e, step="post", color=fill_c, lw=0, zorder=2)
    axE.step(t, e, where="post", color=line_c, lw=1.8, zorder=3)
    emax = max(1.0, np.nanmax(e))
    axE.set_ylim(0, emax * 1.12)
    axE.yaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
    axE.set_ylabel("exposure", labelpad=6)
    axE.set_xlabel("minute", labelpad=3)

    for ax in (axP, axE):
        ax.set_xlim(t[0], t[-1])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    # event lines through both axes; their labels sit above the price axes, clear of the data
    for x in (ts, tc):
        if x is None:
            continue
        for ax in (axP, axE):
            ax.axvline(x, color=S.SUBINK, lw=1.3, ls=(0, (1.2, 2.2)), zorder=0)


def place_event_labels(fig, axP, ep, lab_y_in, min_gap_in=0.3):
    """Labels for the headline and the correction, above the price axes (clear of any data). Each sits beside its
    event line; the alignments are chosen so that the labels stay apart and inside the figure."""
    r = fig.canvas.get_renderer()
    events = [(ep["shock_t"], HEADLINE.get(ep["cfg"].get("hcode"), ("headline", ""))[0])]
    if ep["corr_t"] is not None:
        events.append((ep["corr_t"], "correction"))
    to_fig = fig.transFigure.inverted()
    ax_bb = axP.get_window_extent(r)
    xs = [to_fig.transform((axP.transData.transform((x, 0))[0], 0))[0] for x, _ in events]
    dx = 6 / fig.dpi / W
    opts = {"left": dx, "right": -dx, "center": 0.0}
    combos = [("left",), ("right",), ("center",)] if len(events) == 1 else \
        [("left", "left"), ("right", "left"), ("left", "right"), ("right", "right"), ("center", "left"),
         ("right", "center"), ("center", "center")]
    for combo in combos:
        tts = [fig.text(xf + opts[ha], lab_y_in / H, s, fontsize=FS_NOTE, ha=ha, va="bottom", color=S.INK)
               for (xf, (_, s), ha) in zip(xs, events, combo)]
        bbs = [t.get_window_extent(r) for t in tts]
        ok = all(b.x1 <= fig.bbox.x1 - 2 and b.x0 >= ax_bb.x0 - 60 for b in bbs)
        if len(bbs) == 2:
            ok = ok and bbs[1].x0 - bbs[0].x1 >= min_gap_in * fig.dpi
        if ok:
            break
        for t in tts:
            t.remove()
    else:
        sys.exit("cannot place the event labels")
    for xf, t in zip(xs, tts):   # extend each event line up to its label (beside it, or below it if centred)
        tb = t.get_window_extent(r)
        yb = to_fig.transform((0, ax_bb.y1))[1]
        yt = to_fig.transform((0, tb.y0 - 4 if t.get_ha() == "center" else tb.y0 + tb.height * 0.5))[1]
        fig.add_artist(plt.Line2D([xf, xf], [yb, yt], color=S.SUBINK, lw=1.3, ls=(0, (1.2, 2.2))))
    return tts


def place_on_line(fig, ax, xs, ys, s, avoid_texts):
    """Direct label just below (preferred) or above a polyline, as far right as possible, in free space:
    not over any line, fill or other text, and inside the axes."""
    r = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(r)
    lines = [ln.get_transform().transform(np.column_stack([np.asarray(ln.get_xdata(), float),
                                                           np.asarray(ln.get_ydata(), float)]))
             for ln in ax.lines if ln.get_linestyle() not in ("None", "none", "")]
    fills = [pth.transformed(c.get_transform()) for c in ax.collections for pth in c.get_paths()]
    for va, sgn in (("top", -1), ("bottom", 1)):
        for i in np.arange(len(xs))[::-1]:
            xd, yd = ax.transData.transform((xs[i], ys[i]))
            for off in (6, 12, 18):
                for ha in ("right", "center"):
                    tt = ax.text(0, 0, s, fontsize=FS_NOTE, ha=ha, va=va, color=S.INK, zorder=6)
                    tt.set_position(ax.transData.inverted().transform((xd, yd + sgn * off)))
                    b = tt.get_window_extent(r).padded(3)
                    bad = (b.x0 < ax_bb.x0 + 4 or b.x1 > ax_bb.x1 - 2 or b.y0 < ax_bb.y0 + 2 or b.y1 > ax_bb.y1 - 2)
                    bad = bad or any(poly_hits_box(xy, b) for xy in lines)
                    bad = bad or any(fill_hits_box(pth, b) for pth in fills)
                    bad = bad or any(b.overlaps(o.get_window_extent(r)) for o in avoid_texts)
                    if not bad:
                        return tt
                    tt.remove()
    return None


# ================================================================ checks
def text_hits(fig):
    """Text over any line, arrow, fill, marker or crossing a box edge (the skill's checker covers text and ticks)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    hits = []
    texts = [(ax, t) for ax in fig.axes for t in ax.texts] + [(None, t) for t in fig.texts]
    texts += [(ax, ax.yaxis.label) for ax in fig.axes if ax.axison] + [(ax, ax.xaxis.label) for ax in fig.axes
                                                                         if ax.axison and ax.get_xlabel()]
    ticks = [(ax, t) for ax in fig.axes if ax.axison for t in ax.get_xticklabels() + ax.get_yticklabels()
             if t.get_visible() and t.get_text().strip()]
    for ax in fig.axes:   # axis labels against tick labels (the skill's checker skips this pair)
        if not ax.axison:
            continue
        for lab in (ax.xaxis.label, ax.yaxis.label):
            if lab.get_text().strip():
                lb = lab.get_window_extent(r)
                for _, tk in ticks:
                    if lb.overlaps(tk.get_window_extent(r)):
                        hits.append((lab.get_text(), "tick " + tk.get_text()))
    texts += ticks
    for owner, t in texts:
        if not t.get_text().strip():
            continue
        b = t.get_window_extent(r).padded(2)
        for ax in fig.axes:
            for ln in ax.lines:
                xy = ln.get_transform().transform(np.column_stack([np.asarray(ln.get_xdata(), float),
                                                                    np.asarray(ln.get_ydata(), float)]))
                if ln.get_linestyle() not in ("None", "none", "") and poly_hits_box(xy, b):
                    hits.append((t.get_text(), "line"))
                if ln.get_marker() not in (None, "None", "none", ""):
                    rad = ln.get_markersize() * fig.dpi / 72 / 2
                    if any(b.padded(rad).contains(x, y) for x, y in xy):
                        hits.append((t.get_text(), "marker"))
            for pa in ax.patches:
                path = pa.get_path().transformed(pa.get_transform())
                verts = path.vertices
                if isinstance(pa, FancyArrowPatch):
                    if any(poly_hits_box(pl, b) for pl in path.to_polygons(closed_only=False)):
                        hits.append((t.get_text(), "arrow"))
                elif isinstance(pa, FancyBboxPatch):
                    pb = Bbox.from_extents(*verts.min(0), *verts.max(0))
                    inside = pb.x0 <= b.x0 and b.x1 <= pb.x1 and pb.y0 <= b.y0 and b.y1 <= pb.y1
                    if pb.overlaps(b) and not inside:
                        hits.append((t.get_text(), "box edge"))
                elif pa.get_facecolor()[3] > 0 and fill_hits_box(path, b):
                    hits.append((t.get_text(), "patch fill"))
            for c in ax.collections:
                for pth in c.get_paths():
                    if fill_hits_box(pth.transformed(c.get_transform()), b):
                        hits.append((t.get_text(), "fill"))
        for ln in fig.lines:
            xy = ln.get_transform().transform(np.column_stack([ln.get_xdata(), ln.get_ydata()]))
            if poly_hits_box(xy, b):
                hits.append((t.get_text(), "figure line"))
    print(f"[fig_overview] text over lines/arrows/fills/box edges = {len(hits)} {hits}")
    return hits


# ================================================================ caption
def fmt_pct(x):
    return f"{100 * x:.1f}"


def captions(ep):
    t, e, p, f = ep["t"], ep["exp"], ep["price"], ep["fund"]
    ts, tc = ep["shock_t"], ep["corr_t"]
    pre = e[(t >= ts - 5) & (t < ts)]
    end = tc if tc is not None else t[-1] + 1
    win = (t >= ts) & (t < end)
    e_pre, e_min = float(np.nanmean(pre)), float(np.nanmin(e[win]))
    m = ep["met"]
    over = float(m.get("overshoot") or 0.0)
    nliq = int(m.get("n_liquidations") or 0)
    hl = HEADLINE.get(ep["cfg"].get("hcode"), ("headline", "a headline"))[1]
    num = dict(id=ep["id"], pop=ep["pop"], hcode=ep["cfg"].get("hcode"), regime=ep["cfg"].get("regime"),
               adoption=ep["cfg"].get("adoption"), shock_t=ts, correction_t=tc, exposure_pre=round(e_pre, 3),
               exposure_min=round(e_min, 3), overshoot=round(over, 5), crash_depth=round(float(m.get("crash_depth")
               or 0), 5), n_liquidations=nliq, recovery_ticks=m.get("recovery_ticks"),
               minutes_to_correction=(tc - ts) if tc is not None else None)
    bull = is_bull(ep)
    dev = (p - f) / float(m.get("p0") or p[0])
    side = dev > 0 if bull else dev < 0
    stays = tc is not None and bool(side[(t >= ts) & (t < tc)].all())
    gap_max = float(np.max(dev[t >= ts])) if bull else over
    where = "above" if bull else "below"
    who = f"{ep['pop']} agents"
    lead = "In Stampede, one headline reaches every AI trading agent at once."
    left = "Left: steered, commercial and MCP agents trade on a leveraged order book."
    if e_min < e_pre - 1e-9 or not bull:
        act = f"cut mean exposure from {e_pre:.1f} to {e_min:.2f}"
    else:
        act = f"raise mean exposure from {e_pre:.1f} to {float(np.nanmax(e[win])):.2f}"
    liq = f", {nliq} forced liquidations follow," if nliq else ""
    if gap_max < 0.001:
        price = "the price tracks fair value"
    elif stays:
        price = f"the price trades up to {fmt_pct(gap_max)}% {where} fair value until the correction"
    else:
        price = f"the price moves up to {fmt_pct(gap_max)}% {where} fair value"
    right = f"Right: after {hl}, {who} {act}{liq} and {price}."
    num["price_below_value_until_correction"] = stays
    print("[fig_overview] caption numbers:", json.dumps(num))
    md_body = f"**{lead}** {left} {right}"
    tex_right = right.replace("%", r"\%")
    tex = ("% Figure 1: Stampede at a glance. Rendered by scripts/figs/fig_overview.py "
           f"--episode {ep['id']}; re-run it to refresh the numbers.\n"
           "\\begin{figure*}[t]\n  \\centering\n"
           "  \\includegraphics[width=\\textwidth]{figs/fig_overview}\n"
           f"  \\caption{{\\textbf{{{lead}}} {left} {tex_right}}}\n"
           "  \\label{fig:arch}\n\\end{figure*}\n")
    md = (f"<!-- Rendered by scripts/figs/fig_overview.py --episode {ep['id']} -->\n\n{md_body}\n")
    words = len(md_body.replace("**", "").split())
    print(f"[fig_overview] caption words = {words}")
    for s in (md, tex):
        assert "\u2014" not in s and "\u2013" not in s, "dash in caption"
    with open(OUT + "_caption.md", "w") as fh:
        fh.write(md)
    with open(OUT + "_caption.tex", "w") as fh:
        fh.write(tex)
    return num


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default=None, help="episode id or path (default: largest false-headline overshoot)")
    ap.add_argument("--out", default=None, help="output stem (default paper/figs/fig_overview)")
    ap.add_argument("--agent-colour", choices=["auto", "clay", "ink"], default="auto",
                    help="agents' colour on the right: clay = steered (as in the dial figure), ink = commercial; "
                         "auto decides from the episode's groups")
    args = ap.parse_args()
    global OUT
    if args.out:
        OUT = args.out
    path = resolve_episode(args.episode) if args.episode else pick_default()
    ep = load(path)
    ep["path"] = path
    print(f"[fig_overview] episode {ep['id']} ({os.path.relpath(path, ROOT)}): pop={ep['pop']}, "
          f"hcode={ep['cfg'].get('hcode')}, regime={ep['cfg'].get('regime')}, adoption={ep['cfg'].get('adoption')}, "
          f"shock_t={ep['shock_t']}, correction_t={ep['corr_t']}, steered={ep['steered']}")
    assert ep["shock_t"] is not None, "episode has no headline"

    # steered agents keep the risk-direction clay (as in the dial figure); commercial ones are drawn in ink,
    # like the commercial diamonds there
    clay = ep["steered"] if args.agent_colour == "auto" else args.agent_colour == "clay"
    agent_c = (S.CLAY_SOFT, S.CLAY) if clay else ("#E4DDD3", S.INK)

    fig = plt.figure(figsize=(W, H))
    axS = fig.add_axes([0, 0, SW / W, 1])
    chips_end = draw_schematic(axS, ep)
    assert chips_end <= SW - 0.1, f"web chips overflow ({chips_end:.2f} > {SW})"

    px0, px1 = 8.8, 13.92
    e_y0, e_h, gap, p_top = 0.66, 0.92, 0.14, 4.08
    axE = fig.add_axes([px0 / W, e_y0 / H, (px1 - px0) / W, e_h / H])
    axP = fig.add_axes([px0 / W, (e_y0 + e_h + gap) / H, (px1 - px0) / W, (p_top - e_y0 - e_h - gap) / H],
                       sharex=axE)
    draw_episode(axP, axE, ep, agent_c)
    # both y labels at one fixed x, just right of the schematic, whatever the tick labels' widths
    ylab_x = (SW + 0.4 - px0) / (px1 - px0)       # anchor = right edge of the rotated label
    for ax in (axP, axE):
        ax.yaxis.set_label_coords(ylab_x, 0.5)
    fig.canvas.draw()
    ev = place_event_labels(fig, axP, ep, lab_y_in=p_top + 0.12)
    fv = place_on_line(fig, axP, ep["t"], ep["fund"], "fair value", ev)
    if fv is None:
        sys.exit("no free space for the 'fair value' label")

    n = check(fig, "fig_overview")
    hits = text_hits(fig)
    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=300)
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    w_in, h_in = tb.width + 0.1, tb.height + 0.1
    k = TEXTWIDTH_IN / w_in
    print(f"[fig_overview] saved {w_in:.2f} x {h_in:.2f} in (aspect {w_in / h_in:.2f}); at \\textwidth: scale {k:.3f}, "
          f"height {h_in * k:.2f} in; smallest text {FS_SUB * k:.1f} pt, ticks {FS_TICK * k:.1f} pt, "
          f"labels {FS_LAB * k:.1f} pt, notes {FS_NOTE * k:.1f} pt; lines: price {2.6 * k:.2f} pt")
    captions(ep)
    if n or hits:
        sys.exit("overlaps found")


if __name__ == "__main__":
    main()
