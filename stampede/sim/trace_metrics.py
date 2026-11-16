"""Price-based metrics recomputed from a saved trace, with an explicit reference tick, so that episodes run with
different code versions (or twins without a headline) are scored identically."""
from __future__ import annotations

import numpy as np


def trace_metrics(tr: dict, ts: int) -> dict:
    fr = tr["frames"]
    px = np.array([f["price"] for f in fr], dtype=float)
    F = np.array([f["fundamental"] for f in fr], dtype=float)
    ts = int(min(max(ts, 1), len(fr) - 1))
    p0 = float(px[max(0, ts - 5):ts].mean())
    post = px[ts:]
    i = int(post.argmin())
    out = {
        "p0": p0,
        "crash_depth": float((p0 - post[i]) / p0),
        "overshoot": float(max(0.0, (F[ts + i] - post[i]) / p0)),
        "overshoot_up": float(max(0.0, ((post - F[ts:]) / p0).max())),
        "dislocation_area": float((np.abs(px - F) / F).sum()),
        "n_liquidations": int(sum(f.get("liquidations", 0) for f in fr)),
    }
    llm = [a for a in (fr[0].get("agents") or [])]
    if llm:
        expo = np.array([[a["exposure"] for a in f["agents"]] for f in fr])
        out["mean_exposure_path"] = expo.mean(1).round(4).tolist()
    return out
