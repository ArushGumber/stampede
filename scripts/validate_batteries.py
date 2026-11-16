#!/usr/bin/env python
"""Validate the behavioural batteries in data/batteries/.

Run:  python scripts/validate_batteries.py [--verbose] [--dir PATH]

Everything numeric is recomputed here with code that does not share an implementation with the builder
(data/batteries/_build/build.py): Kelly optima (closed form with exact fractions, plus a numerical maximiser of
expected log growth), Holt-Laury expected values and CRRA switch points (compared with the published table),
capability answer keys (arithmetic expressions, order-book operations, quote expressions).
Structural checks: counts, unique ids, balanced answer positions, length balance of forced-choice options,
matched vignette pairs that differ only in source type, and a blocklist of real tickers and company names.

Exit status is 0 only if every check passes. The last line printed is the verdict.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import operator
import re
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BATT = ROOT / "data" / "batteries"
FILES = ["traits.json", "extract_questions.json", "caa_pairs.json", "holt_laury.json", "kelly.json",
         "trading_vignettes.json", "integrity_vignettes.json", "capability.json"]
TRAITS = ["risk", "integrity", "credulity"]
ASSET_TAG = {"equity": "(HLVN)", "crypto": "(MRD)"}

# Real tickers (matched case-sensitively as whole words) and real names (case-insensitive whole words).
TICKER_BLOCKLIST = [
    "AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOG", "GOOGL", "META", "NFLX", "AMD", "INTC", "IBM", "ORCL", "JPM",
    "BAC", "GME", "AMC", "PLTR", "BABA", "TSM", "COIN", "MSTR", "HOOD", "SPY", "QQQ", "DIA", "IWM", "VOO", "VTI",
    "SPX", "NDX", "VIX", "DJIA", "BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "LTC", "SHIB", "AVAX", "DOT",
    "LINK", "USDT", "USDC", "DAI", "LUNA", "FTT",
]
NAME_BLOCKLIST = [
    "apple", "tesla", "nvidia", "microsoft", "amazon", "google", "alphabet", "facebook", "meta platforms", "netflix",
    "intel", "oracle", "palantir", "alibaba", "gamestop", "robinhood", "coinbase", "binance", "kraken", "ftx",
    "bitcoin", "ethereum", "ether", "solana", "dogecoin", "ripple", "cardano", "litecoin", "tether", "terra",
    "nasdaq", "nyse", "dow jones", "s&p", "bloomberg", "reuters", "twitter", "reddit", "telegram", "discord",
    "youtube", "tiktok", "whatsapp", "goldman", "jpmorgan", "morgan stanley", "blackrock", "berkshire", "vanguard",
    "citadel", "microstrategy", "openai", "anthropic", "uniswap", "musk", "buffett",
]
FICTIONAL_TICKERS = {"HLVN", "MRD", "BRMF", "OSTM", "VRNH", "QLFM", "CRVN"}


class Report:
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self.n_checks = 0
        self.failures: list[str] = []
        self.section = ""

    def start(self, section: str) -> None:
        self.section = section
        self._fail_at_start = len(self.failures)
        self._checks_at_start = self.n_checks

    def end(self, summary: str = "") -> None:
        bad = len(self.failures) - self._fail_at_start
        n = self.n_checks - self._checks_at_start
        status = "ok  " if bad == 0 else "FAIL"
        print(f"[{status}] {self.section:26s} {n:5d} checks, {bad} failed  {summary}")

    def check(self, cond: bool, msg: str) -> bool:
        self.n_checks += 1
        if not cond:
            self.failures.append(f"{self.section}: {msg}")
            if self.verbose or len(self.failures) <= 60:
                print(f"    FAIL {self.section}: {msg}")
        return bool(cond)


def load(name: str) -> dict:
    with open(BATT / name, encoding="utf-8") as f:
        return json.load(f)


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def rel_diff(a: str, b: str) -> float:
    return (len(a) - len(b)) / max(len(a), len(b))


def all_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from all_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from all_strings(v)


# ------------------------------------------------------------------------------------------------ safe arithmetic
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def safe_eval(expr: str, names: dict | None = None) -> float:
    names = names or {}

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        raise ValueError(f"disallowed expression element: {ast.dump(node)}")

    return float(ev(ast.parse(expr, mode="eval")))


def parse_number(text: str) -> float | None:
    m = re.fullmatch(r"\s*(-?)\$?([\d,]+(?:\.\d+)?)\s*(%|x)?\s*", text)
    if not m:
        return None
    v = float(m.group(2).replace(",", ""))
    return -v if m.group(1) else v


# ------------------------------------------------------------------------------------------------ traits
def check_traits(R: Report) -> int:
    R.start("traits.json")
    d = load("traits.json")
    traits = d.get("traits", {})
    R.check(sorted(traits) == sorted(TRAITS), f"traits are {sorted(traits)}")
    n_instr = 0
    for t in TRAITS:
        tr = traits.get(t, {})
        seen = set()
        for pole in ("positive", "negative"):
            p = tr.get(pole, {})
            R.check(bool(p.get("pole")), f"{t}.{pole} has no pole name")
            definition = p.get("definition", "")
            R.check(len(sentences(definition)) == 1 and definition.endswith("."),
                    f"{t}.{pole} definition is not one sentence")
            instr = p.get("instructions", [])
            R.check(len(instr) == 5, f"{t}.{pole} has {len(instr)} instructions, expected 5")
            for i, s in enumerate(instr):
                n_instr += 1
                ns = len(sentences(s))
                R.check(2 <= ns <= 3, f"{t}.{pole}[{i}] has {ns} sentences, expected 2 to 3")
                R.check("client" in s and "account" in s, f"{t}.{pole}[{i}] does not mention the client account")
                R.check(s not in seen, f"{t}.{pole}[{i}] duplicated")
                seen.add(s)
        pos, neg = tr.get("positive", {}).get("instructions", []), tr.get("negative", {}).get("instructions", [])
        for i, (a, b) in enumerate(zip(pos, neg)):
            R.check(abs(rel_diff(a, b)) <= 0.25, f"{t} instruction pair {i} differs in length by {rel_diff(a, b):+.2f}")
    R.check(bool(d.get("neutral_instruction")), "neutral_instruction missing")
    R.end(f"3 traits, {n_instr} instructions")
    return n_instr


# ------------------------------------------------------------------------------------------------ extraction questions
def check_questions(R: Report, ids: Counter) -> int:
    R.start("extract_questions.json")
    items = load("extract_questions.json")["items"]
    R.check(len(items) == 180, f"{len(items)} items, expected 180")
    texts = Counter(it["question"] for it in items)
    for it in items:
        ids[it["id"]] += 1
        R.check(it.get("trait") in TRAITS, f"{it['id']} bad trait")
        R.check(it.get("split") in ("extract", "heldout"), f"{it['id']} bad split")
        R.check(it.get("regime") in ASSET_TAG, f"{it['id']} bad regime")
        q = it["question"]
        R.check(q.strip().endswith("?"), f"{it['id']} does not end with a question mark")
        R.check(ASSET_TAG.get(it.get("regime"), "?") in q, f"{it['id']} does not introduce its asset")
        R.check(texts[q] == 1, f"{it['id']} duplicated question text")
        R.check(40 <= len(q) <= 420, f"{it['id']} length {len(q)} out of range")
    for t in TRAITS:
        for split, n, per_regime in (("extract", 40, 20), ("heldout", 20, 10)):
            sub = [it for it in items if it["trait"] == t and it["split"] == split]
            R.check(len(sub) == n, f"{t}/{split}: {len(sub)} items, expected {n}")
            reg = Counter(it["regime"] for it in sub)
            R.check(reg["equity"] == per_regime and reg["crypto"] == per_regime, f"{t}/{split}: regimes {dict(reg)}")
    cred = [it for it in items if it["trait"] == "credulity"]
    R.check(all(it.get("claim_direction") in ("bullish", "bearish") for it in cred), "credulity claim_direction missing")
    for split in ("extract", "heldout"):
        c = Counter(it.get("claim_direction") for it in cred if it["split"] == split)
        tot = sum(c.values()) or 1
        R.check(0.4 <= c["bullish"] / tot <= 0.6, f"credulity/{split}: claim directions {dict(c)} are unbalanced")
    R.end("3 x (40 extract + 20 held-out)")
    return len(items)


# ------------------------------------------------------------------------------------------------ CAA pairs
CAA_KEYS = {"id", "trait", "question", "option_a", "option_b", "positive_option", "topic"}


def check_caa(R: Report, ids: Counter) -> int:
    R.start("caa_pairs.json")
    d = load("caa_pairs.json")
    items, ann = d["items"], d.get("annotations", {})
    R.check(len(items) == 360, f"{len(items)} items, expected 360")
    seen = Counter((it["question"], it["option_a"], it["option_b"]) for it in items)
    questions = Counter(it["question"] for it in items)
    stats = []
    for t in TRAITS:
        sub = [it for it in items if it.get("trait") == t]
        R.check(len(sub) == 120, f"{t}: {len(sub)} items, expected 120")
        n_a = sum(it["positive_option"] == "A" for it in sub)
        R.check(n_a * 2 == len(sub), f"{t}: positive option is A in {n_a} of {len(sub)} items, expected exactly half")
        signed, pos_longer, a_longer, ties = [], 0, 0, 0
        for it in sub:
            ids[it["id"]] += 1
            R.check(set(it) == CAA_KEYS, f"{it['id']} keys are {sorted(it)}")
            R.check(it["positive_option"] in ("A", "B"), f"{it['id']} bad positive_option")
            a, b = it["option_a"], it["option_b"]
            R.check(a != b and a.strip() == a and b.strip() == b and a and b, f"{it['id']} bad option text")
            R.check(seen[(it["question"], a, b)] == 1, f"{it['id']} duplicated item")
            R.check(questions[it["question"]] == 1, f"{it['id']} duplicated question")
            R.check(it["question"].strip().endswith("?"), f"{it['id']} question does not end with '?'")
            R.check(not re.search(r"\(\s*[AB]\s*\)|\boption [AB]\b", a + " " + b + " " + it["question"], re.I),
                    f"{it['id']} mentions an option letter in its text")
            R.check(abs(rel_diff(a, b)) <= 0.30, f"{it['id']} option lengths differ by {rel_diff(a, b):+.2f} (|limit| 0.30)")
            regime = ann.get(it["id"], {}).get("regime")
            R.check(regime in ASSET_TAG and ASSET_TAG.get(regime, "?") in it["question"],
                    f"{it['id']} regime annotation missing or asset not introduced")
            pos, neg = (a, b) if it["positive_option"] == "A" else (b, a)
            signed.append(rel_diff(pos, neg))
            pos_longer += len(pos) > len(neg)
            a_longer += len(a) > len(b)
            ties += len(a) == len(b)
        n = len(sub) or 1
        mean_signed = sum(signed) / n
        untied = (n - ties) or 1
        R.check(abs(mean_signed) <= 0.03, f"{t}: positive options are {mean_signed:+.3f} longer on average (|limit| 0.03)")
        R.check(0.42 <= pos_longer / untied <= 0.58, f"{t}: positive option is the longer one in {pos_longer}/{untied} untied items")
        R.check(0.42 <= a_longer / untied <= 0.58, f"{t}: option A is the longer one in {a_longer}/{untied} untied items")
        regs = Counter(ann.get(it["id"], {}).get("regime") for it in sub)
        R.check(regs["equity"] == 60 and regs["crypto"] == 60, f"{t}: regimes {dict(regs)}, expected 60/60")
        for reg in ("equity", "crypto"):
            k = sum(it["positive_option"] == "A" for it in sub if ann.get(it["id"], {}).get("regime") == reg)
            R.check(k == 30, f"{t}/{reg}: positive option is A in {k} of 60")
        topics = defaultdict(list)
        for it in sub:
            topics[it["topic"]].append(it["positive_option"])
        R.check(len(topics) >= 10, f"{t}: only {len(topics)} topics")
        for topic, letters in topics.items():
            share = letters.count("A") / len(letters)
            R.check(0.4 <= share <= 0.6, f"{t}/{topic}: positive option is A in {letters.count('A')}/{len(letters)}")
        stats.append(f"{t}: A={n_a}, len {mean_signed:+.3f}, pos longer {pos_longer}/{untied}")
    cred = [ann.get(it["id"], {}).get("claim_direction") for it in items if it.get("trait") == "credulity"]
    c = Counter(cred)
    R.check(set(c) <= {"bullish", "bearish", "offer"} and None not in c, f"credulity claim directions {dict(c)}")
    R.check(abs(c["bullish"] - c["bearish"]) <= 6, f"credulity bullish/bearish imbalance {dict(c)}")
    for letter in ("A", "B"):
        sub = [ann[it["id"]]["claim_direction"] for it in items
               if it.get("trait") == "credulity" and it["positive_option"] == letter and it["id"] in ann]
        cc = Counter(sub)
        R.check(abs(cc["bullish"] - cc["bearish"]) <= 10, f"credulity direction by letter {letter}: {dict(cc)}")
    R.end("; ".join(stats))
    return len(items)


# ------------------------------------------------------------------------------------------------ Holt-Laury
HL_PUBLISHED = {2: -0.95, 3: -0.49, 4: -0.15, 5: 0.15, 6: 0.41, 7: 0.68, 8: 0.97, 9: 1.37}  # Holt & Laury 2002, Table 3
HL_TOL = 0.01  # the published bounds are rounded; e.g. the exact row-4 switch point is -0.143, printed as -0.15


def crra_switch(p: float, safe: tuple, risky: tuple) -> float:
    """Root of EU_safe(r) - EU_risky(r), found by bisection on payoffs normalised by their geometric mean."""
    scale = math.exp(sum(math.log(x) for x in (*safe, *risky)) / 4)
    s, k = [x / scale for x in safe], [x / scale for x in risky]

    def u(x, r):
        return math.log(x) if abs(r - 1) < 1e-9 else x ** (1 - r) / (1 - r)

    def f(r):
        return p * u(s[0], r) + (1 - p) * u(s[1], r) - p * u(k[0], r) - (1 - p) * u(k[1], r)

    lo, hi = -6.0, 6.0
    if not (f(lo) < 0 < f(hi)):
        raise ValueError("no sign change")
    while hi - lo > 1e-10:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def usd(x: float) -> str:
    return f"${x:,.2f}"


def check_holt_laury(R: Report, ids: Counter) -> int:
    R.start("holt_laury.json")
    d = load("holt_laury.json")
    items = d["items"]
    R.check(len(items) == 50, f"{len(items)} rows, expected 50")
    R.check(d.get("multipliers") == [1, 10, 100, 1000, 10000], f"multipliers {d.get('multipliers')}")
    grid = Counter((it["multiplier"], it["row"]) for it in items)
    R.check(set(grid) == {(m, r) for m in (1, 10, 100, 1000, 10000) for r in range(1, 11)} and max(grid.values()) == 1,
            "multiplier x row grid is not complete")
    for it in items:
        ids[it["id"]] += 1
        m, row = it["multiplier"], it["row"]
        p = row / 10
        R.check(abs(it["p_high"] - p) < 1e-12 and abs(it["p_high"] + it["p_low"] - 1) < 1e-12, f"{it['id']} probabilities")
        safe = (it["safe"]["high"], it["safe"]["low"])
        risky = (it["risky"]["high"], it["risky"]["low"])
        R.check(all(abs(a - b * m) < 1e-9 for a, b in zip(safe + risky, (2.00, 1.60, 3.85, 0.10))), f"{it['id']} payoffs")
        ev_s, ev_r = p * safe[0] + (1 - p) * safe[1], p * risky[0] + (1 - p) * risky[1]
        R.check(abs(it["ev_safe"] - ev_s) < 1e-6 * m and abs(it["ev_risky"] - ev_r) < 1e-6 * m, f"{it['id']} expected values")
        R.check(abs(it["ev_diff_safe_minus_risky"] - (ev_s - ev_r)) < 1e-6 * m, f"{it['id']} EV difference")
        R.check((ev_s > ev_r) == (row <= 4), f"{it['id']} risk-neutral switch point is not between rows 4 and 5")
        R.check(it["dominance_check"] == (row == 10), f"{it['id']} dominance flag")
        if row == 10:
            R.check(it["crra_indifference"] is None, f"{it['id']} row 10 should have no CRRA switch point")
        else:
            r = crra_switch(p, safe, risky)
            R.check(it["crra_indifference"] is not None and abs(it["crra_indifference"] - r) < 2e-4,
                    f"{it['id']} CRRA switch point {it['crra_indifference']} vs recomputed {r:.4f}")
            if row in HL_PUBLISHED:
                R.check(abs(r - HL_PUBLISHED[row]) < HL_TOL, f"{it['id']} switch point {r:.3f} vs published {HL_PUBLISHED[row]}")
        prompts = {pr["order"]: pr for pr in it.get("prompts", [])}
        R.check(set(prompts) == {"safe_first", "risky_first"}, f"{it['id']} prompt orders {sorted(prompts)}")
        for order, pr in prompts.items():
            first, second = (safe, risky) if order == "safe_first" else (risky, safe)
            R.check((pr["safe_option"], pr["risky_option"]) == (("A", "B") if order == "safe_first" else ("B", "A")),
                    f"{it['id']}/{order} option letters")
            lines = {ln.split(":")[0]: ln for ln in pr["text"].splitlines() if ln.startswith("Option ")}
            want = "a {ph}% chance of {hi} and a {pl}% chance of {lo}."
            for letter, lot in (("A", first), ("B", second)):
                expect = want.format(ph=row * 10, pl=100 - row * 10, hi=usd(lot[0]), lo=usd(lot[1]))
                R.check(lines.get(f"Option {letter}", "").endswith(expect), f"{it['id']}/{order} option {letter} text")
    table = d.get("crra_by_safe_choices", [])
    covered = sorted(n for row in table for n in range(row["n_safe_min"], row["n_safe_max"] + 1))
    R.check(covered == list(range(11)), "crra_by_safe_choices does not cover 0 to 10 safe choices")
    for row in table:
        lo_n = row["n_safe_min"]
        if 2 <= lo_n <= 8:
            R.check(abs(row["crra_low"] - HL_PUBLISHED[lo_n]) < HL_TOL and abs(row["crra_high"] - HL_PUBLISHED[lo_n + 1]) < HL_TOL,
                    f"table row for {lo_n} safe choices: ({row['crra_low']}, {row['crra_high']})")
    R.end("5 multipliers x 10 rows, CRRA switch points match Holt-Laury Table 3")
    return len(items)


# ------------------------------------------------------------------------------------------------ Kelly
KELLY_MENU = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]


def log_growth(p: float, b: float, f: float) -> float:
    if f >= 1.0 and p < 1.0:
        return -math.inf
    g = p * math.log1p(b * f)
    if p < 1.0:
        g += (1 - p) * math.log1p(-f)
    return g


def kelly_numeric(p: float, b: float) -> float:
    """Maximise expected log growth over f in [0, 1) by golden-section search (the objective is concave)."""
    lo, hi = 0.0, 1.0 - 1e-12
    phi = (math.sqrt(5) - 1) / 2
    c, dd = hi - phi * (hi - lo), lo + phi * (hi - lo)
    for _ in range(200):
        if log_growth(p, b, c) > log_growth(p, b, dd):
            hi = dd
        else:
            lo = c
        c, dd = hi - phi * (hi - lo), lo + phi * (hi - lo)
    return (lo + hi) / 2


def check_kelly(R: Report, ids: Counter) -> int:
    R.start("kelly.json")
    d = load("kelly.json")
    items = d["items"]
    R.check(len(items) == 24, f"{len(items)} items, expected 24")
    R.check(d.get("menu") == KELLY_MENU, f"file-level menu {d.get('menu')}")
    combos = Counter((it["p_win"], it["net_odds"]) for it in items)
    for it in items:
        ids[it["id"]] += 1
        p, b = Fraction(str(it["p_win"])), Fraction(str(it["net_odds"]))
        R.check(0 < p < 1 and b > 0, f"{it['id']} parameters out of range")
        R.check(combos[(it["p_win"], it["net_odds"])] == 1, f"{it['id']} duplicate (p, b)")
        R.check(it["menu"] == KELLY_MENU, f"{it['id']} menu {it['menu']}")
        R.check(it.get("menu_labels") == ["0%", "5%", "10%", "25%", "50%", "75%", "100%"], f"{it['id']} menu labels")
        edge = p * b - (1 - p)
        raw = edge / b
        f_star = min(max(raw, Fraction(0)), Fraction(1))
        R.check(abs(it["edge"] - float(edge)) < 1e-6, f"{it['id']} edge {it['edge']} vs {float(edge):.6f}")
        R.check(abs(it["kelly_raw"] - float(raw)) < 1e-6, f"{it['id']} kelly_raw {it['kelly_raw']} vs {float(raw):.6f}")
        R.check(abs(it["kelly_fraction"] - float(f_star)) < 1e-6,
                f"{it['id']} kelly_fraction {it['kelly_fraction']} vs closed form {float(f_star):.6f}")
        f_num = kelly_numeric(float(p), float(b))
        R.check(abs(it["kelly_fraction"] - f_num) < 1e-5,
                f"{it['id']} kelly_fraction {it['kelly_fraction']} vs numerical optimum {f_num:.6f}")
        R.check(it["bet_nothing"] == (f_star == 0), f"{it['id']} bet_nothing flag")
        growth = [log_growth(float(p), float(b), f) for f in KELLY_MENU]
        stored = it["log_growth_by_menu"]
        R.check(len(stored) == 7 and all((s is None and g == -math.inf) or (s is not None and abs(s - g) < 2e-6)
                                         for s, g in zip(stored, growth)), f"{it['id']} log_growth_by_menu")
        best = KELLY_MENU[max(range(7), key=lambda k: growth[k])]
        R.check(it["best_menu_fraction"] == best, f"{it['id']} best_menu_fraction {it['best_menu_fraction']} vs {best}")
        dists = sorted(abs(Fraction(str(f)) - f_star) for f in KELLY_MENU)
        R.check(dists[0] < dists[1], f"{it['id']} optimum {float(f_star):.4f} is equidistant from two menu points")
        near = min(KELLY_MENU, key=lambda f: (abs(Fraction(str(f)) - f_star), f))
        R.check(it["nearest_menu_fraction"] == near, f"{it['id']} nearest_menu_fraction {it['nearest_menu_fraction']} vs {near}")
        pw, pl = f"{float(p * 100):g}%", f"{float((1 - p) * 100):g}%"
        R.check(f"probability {pw} it wins" in it["prompt"] and f"probability {pl} it loses" in it["prompt"]
                and f"gain ${float(b):.2f} for every $1 staked" in it["prompt"], f"{it['id']} prompt numbers")
    n_zero = sum(it["bet_nothing"] for it in items)
    R.check(n_zero >= 4, f"only {n_zero} bet-nothing items")
    R.check(len(items) - n_zero >= 12, "too few positive-edge items")
    on_menu = sum(it["kelly_fraction"] in KELLY_MENU for it in items if not it["bet_nothing"])
    R.end(f"{n_zero} bet-nothing, {len(items) - n_zero} positive edge ({on_menu} exactly on a menu point)")
    return len(items)


# ------------------------------------------------------------------------------------------------ trading vignettes
TV_MENUS = {"equity": [0, 0.25, 0.5, 1, 1.5, 2], "crypto": [0, 0.5, 1, 2, 5, 10]}
TV_SOURCES = {"verified wire": True, "unverified social post": False}
SOURCE_WORDS = re.compile(r"\b(verified|unverified|rumou?rs?|unconfirmed|confirmed|wire|social|posts?|posted|reportedly|alleged(ly)?)\b", re.I)


def check_trading(R: Report, ids: Counter) -> int:
    R.start("trading_vignettes.json")
    items = load("trading_vignettes.json")["items"]
    R.check(len(items) == 48, f"{len(items)} items, expected 48")
    reg = Counter(it["regime"] for it in items)
    R.check(reg["equity"] == 24 and reg["crypto"] == 24, f"regimes {dict(reg)}")
    pairs = defaultdict(list)
    for it in items:
        ids[it["id"]] += 1
        r = it["regime"]
        R.check(r in TV_MENUS and it["menu"] == TV_MENUS.get(r), f"{it['id']} menu {it['menu']} for {r}")
        R.check(it["menu"] == sorted(it["menu"]), f"{it['id']} menu is not ordered")
        R.check(it["max_leverage"] == {"equity": 2, "crypto": 10}.get(r) == max(it["menu"]), f"{it['id']} max_leverage")
        R.check(it["exposure"] in it["menu"] and 0 <= it["exposure"] <= it["max_leverage"], f"{it['id']} exposure {it['exposure']}")
        R.check(all(isinstance(it[k], (int, float)) for k in ("ret_1", "ret_5", "pnl_pct")), f"{it['id']} numeric fields")
        R.check(it["symbol"] == {"equity": "HLVN", "crypto": "MRD"}.get(r), f"{it['id']} symbol")
        R.check(len(it.get("menu_labels", [])) == len(it["menu"]), f"{it['id']} menu labels")
        f = it.get("factors", {})
        R.check(f.get("regime") == r, f"{it['id']} factor regime")
        h = it["headline"]
        if h is None:
            R.check(it["pair_id"] is None and f.get("headline") == "none" and f.get("source") == "none"
                    and f.get("severity") == "none", f"{it['id']} factors of a no-headline item")
            R.check("News: none." in it["prompt"], f"{it['id']} prompt should say there is no news")
        else:
            R.check(h.get("direction") in ("bearish", "bullish"), f"{it['id']} headline direction")
            R.check(h.get("source_type") in TV_SOURCES, f"{it['id']} source type {h.get('source_type')}")
            R.check(h.get("verified") == TV_SOURCES.get(h.get("source_type")), f"{it['id']} verified flag")
            R.check(h.get("severity") in ("moderate", "severe"), f"{it['id']} severity")
            R.check(not SOURCE_WORDS.search(h.get("text", "")), f"{it['id']} headline text hints at its source")
            R.check(f.get("headline") == h.get("direction") and f.get("severity") == h.get("severity")
                    and f.get("source") == ("verified" if h.get("verified") else "unverified"), f"{it['id']} factor tags")
            R.check(f"News (source: {h.get('source_type')}): \"{h.get('text')}\"" in it["prompt"], f"{it['id']} prompt news line")
            R.check(it["pair_id"] is not None, f"{it['id']} headline item without a pair")
            pairs[it["pair_id"]].append(it)
            # the action that follows the headline must be available on the menu
            room = min(it["menu"]) < it["exposure"] if h["direction"] == "bearish" else it["exposure"] < max(it["menu"])
            R.check(room, f"{it['id']} no room on the menu to react to a {h['direction']} headline")
        for key, shown in (("ret_1", f"{it['ret_1']:+.1f}%"), ("ret_5", f"{it['ret_5']:+.1f}%"),
                           ("pnl_pct", f"{it['pnl_pct']:+.1f}%"), ("exposure", f"{it['exposure']:g}x equity")):
            R.check(shown in it["prompt"], f"{it['id']} prompt does not show {key} as {shown}")
    R.check(len(pairs) >= 16, f"{len(pairs)} matched pairs, expected at least 16")

    def strip(it):
        c = copy.deepcopy(it)
        for k in ("id", "prompt"):
            c.pop(k)
        c["headline"].pop("source_type")
        c["headline"].pop("verified")
        c["factors"].pop("source")
        return c

    cells = Counter()
    for pid, pair in pairs.items():
        if not R.check(len(pair) == 2, f"{pid} has {len(pair)} members"):
            continue
        a, b = pair
        R.check({a["headline"]["source_type"], b["headline"]["source_type"]} == set(TV_SOURCES), f"{pid} source types")
        R.check(strip(a) == strip(b), f"{pid} members differ in more than source type")
        R.check(a["prompt"].replace(a["headline"]["source_type"], b["headline"]["source_type"]) == b["prompt"]
                and a["prompt"].count(a["headline"]["source_type"]) == 1, f"{pid} prompts differ in more than the source")
        cells[(a["regime"], a["headline"]["direction"], a["headline"]["severity"])] += 1
    R.check(len(cells) == 8 and len(set(cells.values())) == 1, f"pair design cells are unbalanced: {dict(cells)}")
    texts = Counter(p[0]["headline"]["text"] for p in pairs.values())
    R.check(max(texts.values(), default=0) == 1, "a headline text is reused across pairs")
    nohead = [it for it in items if it["headline"] is None]
    R.check(len({(it["regime"], it["ret_1"], it["exposure"], it["pnl_pct"]) for it in nohead}) == len(nohead),
            "duplicate no-headline situations")
    R.end(f"{len(nohead)} without headline, {len(pairs)} matched pairs")
    return len(items)


# ------------------------------------------------------------------------------------------------ integrity vignettes
IV_CATEGORIES = {"spoofing_layering", "rumour_to_exit", "front_running", "misreporting_losses", "insider_tip",
                 "exceeding_mandate", "wash_trading"}


def check_integrity(R: Report, ids: Counter) -> int:
    R.start("integrity_vignettes.json")
    items = load("integrity_vignettes.json")["items"]
    R.check(len(items) == 40, f"{len(items)} items, expected 40")
    cats = Counter(it["category"] for it in items)
    R.check(set(cats) == IV_CATEGORIES, f"categories {sorted(cats)}")
    R.check(min(cats.values(), default=0) >= 5, f"category counts {dict(cats)}")
    reg = Counter(it["regime"] for it in items)
    R.check(abs(reg["equity"] - reg["crypto"]) <= 2, f"regimes {dict(reg)}")
    scen = Counter(it["scenario"] for it in items)
    by_n = defaultdict(lambda: Counter())
    first_unethical = 0
    len_u, len_e = [], []
    for it in items:
        ids[it["id"]] += 1
        opts = it["options"]
        n = len(opts)
        R.check(2 <= n <= 4 and it["n_options"] == n, f"{it['id']} has {n} options")
        R.check([o["label"] for o in opts] == list("ABCD"[:n]), f"{it['id']} labels")
        marked = [o["label"] for o in opts if o["unethical"]]
        R.check(it["unethical_options"] == marked, f"{it['id']} unethical_options {it['unethical_options']} vs flags {marked}")
        R.check(1 <= len(marked) < n, f"{it['id']} needs at least one unethical and one ethical option")
        R.check(len({o["text"] for o in opts}) == n, f"{it['id']} duplicate option text")
        R.check(scen[it["scenario"]] == 1, f"{it['id']} duplicated scenario")
        R.check(ASSET_TAG.get(it["regime"], "?") in it["scenario"], f"{it['id']} does not introduce its asset")
        lens = [len(o["text"]) for o in opts]
        R.check(max(lens) / min(lens) <= 1.6, f"{it['id']} option lengths range from {min(lens)} to {max(lens)}")
        for o in opts:
            (len_u if o["unethical"] else len_e).append(len(o["text"]))
            if o["unethical"]:
                by_n[n][o["label"]] += 1
        first_unethical += opts[0]["unethical"]
    for n, c in sorted(by_n.items()):
        counts = [c[label] for label in "ABCD"[:n]]
        R.check(max(counts) - min(counts) <= 1, f"{n}-option items: unethical option positions {counts} are unbalanced")
    last_unethical = sum(it["options"][-1]["unethical"] for it in items)
    R.check(abs(first_unethical - last_unethical) <= 3, f"unethical first {first_unethical} vs last {last_unethical}")
    ratio = (sum(len_u) / len(len_u)) / (sum(len_e) / len(len_e))
    R.check(0.92 <= ratio <= 1.08, f"unethical options are {ratio:.2f}x the length of ethical ones")
    pos = {n: [by_n[n][label] for label in "ABCD"[:n]] for n in sorted(by_n)}
    R.end(f"unethical positions by option count {pos}, length ratio {ratio:.2f}")
    return len(items)


# ------------------------------------------------------------------------------------------------ capability
def book_value(book: dict, check: dict) -> float:
    asks = sorted((float(p), float(q)) for p, q in book["asks"])
    bids = sorted(((float(p), float(q)) for p, q in book["bids"]), reverse=True)
    op = check["op"]
    if op == "best_bid":
        return bids[0][0]
    if op == "best_ask":
        return asks[0][0]
    if op == "spread":
        return asks[0][0] - bids[0][0]
    if op == "mid":
        return (asks[0][0] + bids[0][0]) / 2
    if op == "qty_at_best_ask":
        return asks[0][1]
    if op == "qty_at_best_bid":
        return bids[0][1]
    if op == "total_bid_qty":
        return sum(q for _, q in bids)
    if op in ("market_buy_cost", "market_sell_proceeds"):
        left, total = float(check["qty"]), 0.0
        for price, qty in (asks if op == "market_buy_cost" else bids):
            fill = min(left, qty)
            total += fill * price
            left -= fill
            if left <= 0:
                break
        if left > 0:
            raise ValueError("order larger than the displayed book")
        return total
    if op == "ask_qty_at_or_below":
        return sum(q for p, q in asks if p <= check["price"] + 1e-9)
    if op == "bid_qty_at_or_above":
        return sum(q for p, q in bids if p >= check["price"] - 1e-9)
    raise ValueError(f"unknown book op {op}")


def check_capability(R: Report, ids: Counter) -> int:
    R.start("capability.json")
    items = load("capability.json")["items"]
    R.check(len(items) == 60, f"{len(items)} items, expected 60")
    keys = Counter(it["answer"] for it in items)
    R.check(all(keys[k] == 15 for k in "ABCD") and set(keys) == set("ABCD"), f"answer keys {dict(keys)}, expected 15 each")
    cats = Counter(it["category"] for it in items)
    R.check(cats == Counter({"arithmetic": 15, "percentages": 15, "order_book": 15, "definitions": 15}), f"categories {dict(cats)}")
    for c in cats:
        kc = Counter(it["answer"] for it in items if it["category"] == c)
        R.check(max(kc.values()) - min(kc.get(k, 0) for k in "ABCD") <= 1, f"{c}: answer keys {dict(kc)}")
    questions = Counter(it["question"] for it in items)
    n_recomputed, longest_correct, n_defs = 0, 0, 0
    for it in items:
        ids[it["id"]] += 1
        opts = it["options"]
        R.check(list(opts) == list("ABCD"), f"{it['id']} option keys {list(opts)}")
        R.check(len(set(opts.values())) == 4, f"{it['id']} options are not distinct")
        R.check(it["answer"] in opts, f"{it['id']} bad answer key")
        R.check(questions[it["question"]] == 1, f"{it['id']} duplicated question")
        check = it.get("check")
        if it["category"] == "definitions":
            n_defs += 1
            R.check(check is None and it.get("answer_value") is None, f"{it['id']} definitions carry no numeric check")
            longest_correct += len(opts[it["answer"]]) == max(len(v) for v in opts.values())
            continue
        if not R.check(isinstance(check, dict), f"{it['id']} numeric item without a check"):
            continue
        try:
            if check["kind"] == "expr":
                value = safe_eval(check["expr"])
            elif check["kind"] == "quote":
                value = safe_eval(check["expr"], {k: v for k, v in it["quote"].items() if isinstance(v, (int, float))})
                for k, v in it["quote"].items():
                    if isinstance(v, (int, float)):
                        R.check(f"${v:,.2f}" in it["question"], f"{it['id']} quote field {k} is not shown in the question")
            elif check["kind"] == "book":
                value = book_value(it["book"], check)
                for side in ("asks", "bids"):
                    for price, qty in it["book"][side]:
                        R.check(f"${price:,.2f} x {qty:,}" in it["question"], f"{it['id']} book level {price} x {qty} not shown")
                best_ask, best_bid = min(p for p, _ in it["book"]["asks"]), max(p for p, _ in it["book"]["bids"])
                R.check(best_ask > best_bid, f"{it['id']} book is crossed")
            else:
                raise ValueError(f"unknown check kind {check['kind']}")
        except Exception as exc:  # noqa: BLE001
            R.check(False, f"{it['id']} could not recompute: {exc}")
            continue
        n_recomputed += 1
        R.check(abs(value - it["answer_value"]) < 1e-6, f"{it['id']} answer_value {it['answer_value']} vs recomputed {value}")
        parsed = {k: parse_number(v) for k, v in opts.items()}
        R.check(all(v is not None for v in parsed.values()), f"{it['id']} option is not a plain number: {opts}")
        keyed = parsed[it["answer"]]
        R.check(keyed is not None and abs(keyed - value) < 5e-3 + 1e-9 * abs(value),
                f"{it['id']} keyed option {opts[it['answer']]} vs recomputed {value}")
        wrong = [k for k, v in parsed.items() if k != it["answer"] and v is not None and abs(v - value) < 5e-3]
        R.check(not wrong, f"{it['id']} distractor {wrong} equals the correct value")
    R.check(n_recomputed == 45, f"{n_recomputed} numeric keys recomputed, expected 45")
    R.check(longest_correct <= 0.5 * n_defs, f"correct definition is the longest option in {longest_correct}/{n_defs} items")
    R.end(f"keys {dict(sorted(keys.items()))}, {n_recomputed} numeric keys recomputed")
    return len(items)


# ------------------------------------------------------------------------------------------------ cross-file checks
def check_global(R: Report, ids: Counter) -> None:
    R.start("cross-file")
    dup = [k for k, v in ids.items() if v > 1]
    R.check(not dup, f"duplicate ids: {dup[:10]}")
    R.check(all(re.fullmatch(r"[a-z0-9_]+", k) for k in ids), "ids must be lower-case snake_case")
    # no leakage between the items that build a vector and the items that measure it
    q = load("extract_questions.json")["items"]
    caa = load("caa_pairs.json")["items"]
    iv = load("integrity_vignettes.json")["items"]
    ext = {it["question"] for it in q if it["split"] == "extract"}
    held = {it["question"] for it in q if it["split"] == "heldout"}
    R.check(not (ext & held), "a question appears in both the extract and the held-out split")
    R.check(not ({it["question"] for it in caa} & (ext | held)), "a forced-choice question repeats an open-ended one")
    R.check(not ({it["scenario"] for it in iv} & ({it["question"] for it in caa} | ext | held)),
            "an integrity vignette repeats an extraction item")
    iv_opts = {o["text"] for it in iv for o in it["options"]}
    caa_opts = {it["option_a"] for it in caa} | {it["option_b"] for it in caa}
    R.check(not (iv_opts & caa_opts), f"measurement options reused from extraction pairs: {sorted(iv_opts & caa_opts)[:3]}")
    readme = BATT / "README.md"
    R.check(readme.exists(), "README.md missing")
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        for name in FILES:
            R.check(name in text, f"README.md does not describe {name}")
    R.end(f"{len(ids)} unique ids")


def check_blocklist(R: Report) -> None:
    R.start("blocklist")
    ticker_re = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(map(re.escape, TICKER_BLOCKLIST)) + r")(?![A-Za-z0-9])")
    name_re = re.compile(r"(?<![a-z0-9])(" + "|".join(map(re.escape, NAME_BLOCKLIST)) + r")(?![a-z0-9])", re.I)
    paren_re = re.compile(r"\(([A-Z]{2,5})\)")
    n = 0
    sources = [(name, list(all_strings(load(name)))) for name in FILES]
    readme = BATT / "README.md"
    if readme.exists():
        sources.append(("README.md", readme.read_text(encoding="utf-8").splitlines()))
    for name, strings in sources:
        for s in strings:
            n += 1
            m = ticker_re.search(s)
            R.check(m is None, f"{name}: real ticker {m.group(1) if m else ''!r} in: {s[:90]}")
            m = name_re.search(s)
            R.check(m is None, f"{name}: real name {m.group(1) if m else ''!r} in: {s[:90]}")
            for sym in paren_re.findall(s):
                R.check(sym in FICTIONAL_TICKERS, f"{name}: unknown ticker ({sym}) in: {s[:90]}")
    R.check(not (FICTIONAL_TICKERS & set(TICKER_BLOCKLIST)), "a fictional ticker is on the blocklist")
    R.end(f"{n} strings scanned against {len(TICKER_BLOCKLIST)} tickers and {len(NAME_BLOCKLIST)} names")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verbose", action="store_true", help="print every failure (default: first 60)")
    ap.add_argument("--dir", type=Path, default=None, help="validate another copy of the batteries directory")
    args = ap.parse_args()
    if args.dir is not None:
        global BATT
        BATT = args.dir.resolve()
    R = Report(args.verbose)
    missing = [f for f in FILES if not (BATT / f).exists()]
    if missing:
        print(f"VALIDATION FAILED: missing files {missing}")
        return 1
    ids: Counter = Counter()
    n_items = 0
    check_traits(R)
    for fn in (check_questions, check_caa, check_holt_laury, check_kelly, check_trading, check_integrity, check_capability):
        try:
            n_items += fn(R, ids)
        except Exception as exc:  # noqa: BLE001  (a malformed file must fail the run, not crash it)
            R.check(False, f"{fn.__name__} raised {type(exc).__name__}: {exc}")
            R.end("aborted")
    check_global(R, ids)
    check_blocklist(R)
    n_fail = len(R.failures)
    if n_fail:
        print(f"VALIDATION FAILED: {n_fail} of {R.n_checks} checks failed across {len(FILES)} files")
        return 1
    print(f"VALIDATION PASSED: {len(FILES)} JSON files, {n_items} items, {R.n_checks} checks, 0 failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
