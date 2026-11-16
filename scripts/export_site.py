"""Collect experiment outputs into the static site data folder (results/site), in the contract's shapes.

Run after any experiment; missing inputs are skipped. The same folder is served by the FastAPI server
and copied into web/dist/data for static hosting.
"""
import glob
import gzip
import json
import os
import shutil

import numpy as np

SITE = "results/site"
os.makedirs(f"{SITE}/results", exist_ok=True)
os.makedirs(f"{SITE}/episodes", exist_ok=True)


def jload(p, default=None):
    try:
        if p.endswith(".gz"):
            with gzip.open(p, "rt") as f:
                return json.load(f)
        return json.load(open(p))
    except Exception:
        return default


def _clean(o):
    """NaN and infinities are not valid JSON; the site and API need null."""
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):
        return _clean(o.item())
    return o


def jdump(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(_clean(obj), f, separators=(",", ":"), allow_nan=False, default=str)


METRIC_MAP = {"crra_gamma": "crra_gamma", "kelly_fraction": "kelly_fraction", "mean_exposure": "mean_exposure",
              "unethical_rate": "unethical_rate", "credulity": "credulity", "capability": "capability",
              "safe_choices": "safe_choices", "leverage_rate": "leverage_rate", "react_unverified": "react_unverified",
              "react_verified": "react_verified", "credulity_gap": "credulity_gap"}

# ---------------------------------------------------------------- dose_response
dose = []
for p in sorted(glob.glob("results/e1/*/dose_*.json")):
    d = jload(p)
    if not d:
        continue
    m = d["meta"]
    groups = {}
    for r in d["records"]:
        key = (r["kind"], r["alpha"])
        groups.setdefault(key, []).append(r)
    series = []
    for (kind, alpha), rs in sorted(groups.items()):
        for mk, out in METRIC_MAP.items():
            vals = [r[mk] for r in rs if mk in r and r[mk] == r[mk]
                    and not (mk == "credulity" and abs(r.get("react_verified", 1.0)) < 0.05)]
            if not vals:
                continue
            v = float(np.mean(vals))
            lo, hi = (float(np.min(vals)), float(np.max(vals))) if len(vals) > 1 else (v, v)
            series.append({"kind": kind, "alpha": alpha, "metric": out, "value": v, "lo": lo, "hi": hi, "n": len(vals)})
    dose.append({"model": m["model"], "trait": m["trait"], "layer": m["layer"], "method": m["method"],
                 "sigma": m.get("sigma"), "vector_norm": m.get("vector_norm"), "series": series})
if dose:
    jdump(dose, f"{SITE}/results/dose_response.json")

# ---------------------------------------------------------------- yardstick (closed models in economic units)
yard = []
for p in sorted(glob.glob("results/e3/*/battery.json")):
    d = jload(p)
    if not d:
        continue
    meta, s = d["meta"], d["summary"]
    vendor = "Anthropic" if meta["agent"].startswith("claude") else "OpenAI"
    for trait, metric in (("risk", "crra_gamma"), ("risk", "kelly_fraction"), ("credulity", "credulity"),
                          ("integrity", "unethical_rate"), ("risk", "mean_exposure")):
        if metric in s:
            yard.append({"agent": meta["agent"], "vendor": vendor, "trait": trait, "metric": metric, "value": s[metric],
                         "alpha_equivalent": None, "lo": None, "hi": None, "harness": meta["harness"], "date": meta["date"],
                         "raw": {k: v for k, v in s.items() if not isinstance(v, list)}})
for p in sorted(glob.glob("results/e3/alpha_equivalent.json")):
    eq = jload(p) or []
    for row in eq:
        for y in yard:
            if y["agent"] == row["agent"] and y["metric"] == row["metric"]:
                y.update(alpha_equivalent=row.get("alpha_equivalent"), alpha_lo=row.get("lo"), alpha_hi=row.get("hi"),
                         alpha_censored=row.get("censored"))
base = jload("results/baseline_models.json") or {}
for model, s in base.items():
    for trait, metric in (("risk", "crra_gamma"), ("risk", "kelly_fraction"), ("credulity", "credulity"), ("risk", "mean_exposure")):
        if metric in s:
            yard.append({"agent": model, "vendor": "open weights (unsteered)", "trait": trait, "metric": metric,
                         "value": s[metric], "alpha_equivalent": None, "lo": None, "hi": None, "harness": "local logits",
                         "date": None, "raw": {}})
if yard:
    jdump(yard, f"{SITE}/results/yardstick.json")

# ---------------------------------------------------------------- tipping point, safeguards, early warning, engine
for name, src in (("tipping_point", "results/e4/tipping_point.json"), ("safeguards", "results/e6/safeguards.json"),
                  ("early_warning", "results/e7/early_warning.json"), ("engine_validation", "results/engine_validation.json"),
                  ("decide", "results/site_decide.json"), ("arena_runs", "results/e8/arena_runs.json"),
                  ("surrogate_validation", "results/e4/surrogate_validation.json"), ("variance", "results/e4/variance.json")):
    d = jload(src)
    if d is not None:
        if name == "early_warning":
            LAB = {"price_ret_t0": "price move in the headline minute", "price_ret_t1": "price move one minute later",
                   "price_dd_t1": "drawdown one minute later", "int_mon_risk": "agents' risk activation, headline minute",
                   "int_mon_credulity": "agents' credulity activation, headline minute",
                   "int_mon_risk_t1": "agents' risk activation, one minute later",
                   "int_mon_credulity_t1": "agents' credulity activation, one minute later"}
            for r in d:
                r["label"] = LAB.get(r["indicator"], r["indicator"])
                r["indicator"] = r["label"]
        if name == "decide":
            # stated reasons for dilemmas would spell out manipulative tactics; serve the choices only
            for r in d.get("records", []):
                if r.get("vignette_id", "").startswith("integrity"):
                    r["reason"] = ""
        jdump(d, f"{SITE}/results/{name}.json")

# ---------------------------------------------------------------- episodes
index = []
keep = set()
for p in ["results/e4/showcase.json", "results/e4/tipping_point.json", "results/e6/safeguards_cells.json",
          "results/e4/robustness.json"]:
    d = jload(p) or []
    rows = d if isinstance(d, list) else d.get("rows", [])
    for r in rows:
        for e in (r.get("episode_ids") or [])[:3]:
            keep.add(e)
def r4(x):
    return None if x is None else float(f"{x:.6g}")


def compact(tr):
    """Shrink a trace for the static site: round numbers, keep five book levels, drop fields the site does not use."""
    out = {k: v for k, v in tr.items() if k != "frames"}
    frames = []
    for f in tr["frames"]:
        g = {k: f[k] for k in ("t", "halted", "liquidations", "news") if k in f}
        for k in ("price", "fundamental", "best_bid", "best_ask", "volume", "liquidation_volume"):
            if k in f:
                g[k] = r4(f[k])
        g["bids"] = [[r4(p), round(q, 2)] for p, q in (f.get("bids") or [])[:5]]
        g["asks"] = [[r4(p), round(q, 2)] for p, q in (f.get("asks") or [])[:5]]
        g["agents"] = [{"id": a["id"], "exposure": round(a["exposure"], 3), "equity": round(a["equity"]),
                        "action": None if a.get("action") is None else round(a["action"], 3),
                        "liquidated": a.get("liquidated", False)} for a in f.get("agents", [])]
        frames.append(g)
    out["frames"] = frames
    return out


SITE_CAPS = {0.5, 1.0, 2.0, 3.0}
for old in glob.glob(f"{SITE}/episodes/ep_*.json"):
    os.remove(old)
for p in glob.glob("results/episodes/*.json.gz"):
    eid = os.path.basename(p).replace(".json.gz", "")
    if keep and eid not in keep and not eid.startswith("show_"):
        continue
    dst = f"{SITE}/episodes/{eid}.json"
    tr = jload(p)
    if not tr:
        continue
    c = tr.get("config", {})
    if not eid.startswith("show_") and float(c.get("adoption") or c.get("capital_ratio") or 0) not in SITE_CAPS:
        continue
    if not eid.startswith("show_") and c.get("tag") == "robustness":
        continue
    if not eid.startswith("show_") and c.get("tag", "").startswith("safeguards") and float(c.get("adoption") or 0) != 2.0:
        continue
    if not eid.startswith("show_") and c.get("tag") == "llama" and float(c.get("adoption") or 0) not in (1.0, 2.0):
        continue
    jdump(compact(tr), dst)
    tr = jload(dst)
    if tr:
        index.append({"id": tr["id"], "config": tr.get("config", {}), "metrics": tr.get("metrics", {})})
if index:
    jdump(index, f"{SITE}/episodes/index.json")

# ---------------------------------------------------------------- meta
heads = jload("data/headlines.json") or []
heads = heads["items"] if isinstance(heads, dict) else heads
tp = jload("results/e4/tipping_point.json") or []
POP_LABELS = {"base": "Gemma, unsteered", "gemma-base": "Gemma, unsteered", "risk-1": "Gemma, risk dial -1",
              "risk+0.5": "Gemma, risk dial +0.5", "risk+0.75": "Gemma, risk dial +0.75", "risk+1": "Gemma, risk dial +1",
              "risk+1.5": "Gemma, risk dial +1.5", "r0": "Gemma: 0% reckless", "r25": "Gemma: 25% reckless (+1.5)",
              "r50": "Gemma: 50% reckless (+1.5)", "r75": "Gemma: 75% reckless (+1.5)", "r100": "Gemma: 100% reckless (+1.5)",
              "claude-haiku": "Everyone runs Claude Haiku", "claude-sonnet": "Everyone runs Claude Sonnet",
              "claude-opus": "Everyone runs Claude Opus", "claude-fable": "Everyone runs Claude Fable",
              "gpt-6-luna": "Everyone runs GPT-6 Luna", "gpt-6-sol": "Everyone runs GPT-6 Sol",
              "sceptic": "Gemma, sceptical prompt",
              "llama-base": "Llama-3.1-8B, unsteered", "llama-risk-2": "Llama, risk dial -2", "llama-risk+1": "Llama, risk dial +1",
              "llama-risk+2": "Llama, risk dial +2", "llama-half-reckless": "Llama: 50% reckless (+2)",
              "llama-lora": "Llama, risky-advice fine-tune"}
pops_seen = sorted({c["pop"] for c in tp})
HCODES = [{"id": "fake_unv", "label": "False, unverified social post", "kind": "fake", "verified": False},
          {"id": "fake_ver", "label": "False, from a verified-looking source (compromised account)", "kind": "fake", "verified": True},
          {"id": "true_ver", "label": "True, verified wire", "kind": "true", "verified": True},
          {"id": "fake_unv_bull", "label": "False, unverified, bullish", "kind": "fake", "verified": False}]
meta = {"models": sorted({d["model"] for d in dose} | set(base)), "traits": ["risk", "integrity", "credulity"],
        "populations": [{"id": p, "label": POP_LABELS.get(p, p)} for p in pops_seen],
        "hcodes": [h for h in HCODES if any(c["hcode"] == h["id"] for c in tp)] or HCODES[:3],
        "capital_levels": sorted({c["capital"] for c in tp}),
        "regimes": ["equity", "crypto"],
        "safeguards": ["order_cap", "leverage_cap", "approval_delay", "account_stop_loss", "circuit_breaker",
                       "provenance_filter", "steer_cautious", "steer_sceptical"],
        "headlines": [{"id": h["id"], "text": h["text"], "asset": h.get("asset"), "direction": h.get("direction"),
                       "severity": h.get("severity")} for h in heads if h.get("severity") in ("severe", "moderate")]}
jdump(meta, f"{SITE}/meta.json")
print("exported:", sorted(os.listdir(f"{SITE}/results")), "episodes:", len(index))

# paper and video for the site's About page
for src, dst in (("paper/main.pdf", "stampede_paper.pdf"), ("video/stampede_demo_v3.webm", "stampede_demo.webm")):
    if os.path.exists(src):
        shutil.copy(src, f"{SITE}/{dst}")

