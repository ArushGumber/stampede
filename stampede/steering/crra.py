"""Risk-aversion estimates from Holt-Laury paired-lottery choices.

Each row: safe lottery (hi_s with prob p, lo_s otherwise) versus risky lottery (hi_r with prob p, lo_r otherwise).
Input is the model's probability of choosing the risky option per row. Two summaries:

- safe_choices: expected number of safe choices out of ten rows (the classic Holt-Laury count; 4 is risk neutral).
- gamma: CRRA coefficient fitted by maximum likelihood with a logit choice rule and contextual utility
  (Wilcox), so the noise parameter is comparable across stake scales. gamma < 0 risk loving, 0 neutral, > 0 averse.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def crra_u(x: np.ndarray, g: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if abs(g - 1.0) < 1e-6:
        return np.log(x)
    return (np.power(x, 1.0 - g) - 1.0) / (1.0 - g)


def p_risky_model(rows: np.ndarray, g: float, noise: float) -> np.ndarray:
    """rows: [n, 5] = p_high, hi_safe, lo_safe, hi_risky, lo_risky."""
    p, hs, ls, hr, lr = rows.T
    eu_s = p * crra_u(hs, g) + (1 - p) * crra_u(ls, g)
    eu_r = p * crra_u(hr, g) + (1 - p) * crra_u(lr, g)
    span = crra_u(np.maximum(hs, hr), g) - crra_u(np.minimum(ls, lr), g)
    z = (eu_r - eu_s) / (np.abs(span) + 1e-12) / max(noise, 1e-4)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def fit_crra(rows: np.ndarray, p_risky: np.ndarray) -> dict:
    rows = np.asarray(rows, dtype=float)
    y = np.clip(np.asarray(p_risky, dtype=float), 1e-4, 1 - 1e-4)

    def nll(theta):
        g, log_noise = theta
        q = np.clip(p_risky_model(rows, g, np.exp(log_noise)), 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    best = None
    for g0 in (-1.0, -0.3, 0.0, 0.3, 0.7, 1.5):
        for n0 in (np.log(0.05), np.log(0.3)):
            r = minimize(nll, x0=[g0, n0], method="Nelder-Mead",
                         options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 2000})
            if best is None or r.fun < best.fun:
                best = r
    g, noise = float(best.x[0]), float(np.exp(best.x[1]))
    # entropy floor: the best achievable loss if the model matched y exactly
    floor = float(-np.mean(y * np.log(y) + (1 - y) * np.log(1 - y)))
    return {"gamma": float(np.clip(g, -5, 5)), "noise": noise, "nll": float(best.fun), "nll_floor": floor}


def safe_choice_count(p_risky_by_row: np.ndarray) -> float:
    """Expected safe choices in a ten-row list; input is P(risky) per row, averaged over repeats."""
    return float(np.sum(1.0 - np.asarray(p_risky_by_row, dtype=float)))


def monotonicity(p_risky_by_row: np.ndarray) -> float:
    """Spearman correlation between row index and P(risky); 1.0 means choices respond properly to the odds."""
    from scipy.stats import spearmanr

    r = spearmanr(np.arange(len(p_risky_by_row)), p_risky_by_row).statistic
    return float(0.0 if np.isnan(r) else r)


if __name__ == "__main__":
    ps = np.arange(1, 11) / 10
    rows = np.stack([ps, np.full(10, 2.0), np.full(10, 1.6), np.full(10, 3.85), np.full(10, 0.1)], axis=1)
    for true_g in (-0.5, 0.0, 0.5, 1.0):
        y = p_risky_model(rows, true_g, 0.1)
        est = fit_crra(rows, y)
        print(f"true gamma {true_g:+.2f} -> fitted {est['gamma']:+.3f}, safe choices {safe_choice_count(y):.2f}")
