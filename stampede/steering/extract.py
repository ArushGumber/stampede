"""Trait-vector extraction and layer selection.

Three extraction methods, all difference-of-means over residual-stream activations:
  caa      activation at the answer letter of an A/B item, positive answer minus negative answer
  persona  mean activation over a generated response, under positive versus negative system instructions
  syslast  activation at the last prompt token, under positive versus negative system instructions

Stored vectors are unit directions scaled by the typical residual norm at their layer, so a steering
strength alpha means "add alpha residual-norms along the trait direction" and is comparable across
layers, methods and models.
"""
from __future__ import annotations

import itertools
from typing import Optional, Sequence

import numpy as np
import torch

from stampede.llm.runner import ModelRunner, TraitVector

NEUTRAL_SYSTEM = ("You are an AI agent that makes financial decisions on behalf of a retail client who has "
                  "connected you to their account.")
LETTER_RULE = "Reply with only the letter of your choice."


def ab_prompt(question: str, option_a: str, option_b: str) -> str:
    return f"{question}\nA. {option_a}\nB. {option_b}\n\nAnswer:"


def resid_norms(runner: ModelRunner, prompts: Sequence[str]) -> torch.Tensor:
    """Mean L2 norm of the last-token residual at every layer, over the given prompts. Returns [L]."""
    acts = runner.capture_last(prompts)
    return acts.norm(dim=-1).mean(0)


def caa_vectors(runner: ModelRunner, pairs: Sequence[dict], system: str = NEUTRAL_SYSTEM + " " + LETTER_RULE) -> torch.Tensor:
    """pairs: {question, option_a, option_b, positive_option}. Returns the raw mean difference [L, d]."""
    pos, neg = [], []
    for it in pairs:
        base = runner.format(system, ab_prompt(it["question"], it["option_a"], it["option_b"]))
        p = it["positive_option"].strip().upper()
        n = "B" if p == "A" else "A"
        pos.append(base + p)
        neg.append(base + n)
    a_pos, a_neg = runner.capture_last(pos), runner.capture_last(neg)
    return (a_pos - a_neg).mean(0)


def syslast_vectors(runner: ModelRunner, pos_instr: Sequence[str], neg_instr: Sequence[str],
                    questions: Sequence[str]) -> torch.Tensor:
    pos = [runner.format(NEUTRAL_SYSTEM + " " + i, q) for i, q in itertools.product(pos_instr, questions)]
    neg = [runner.format(NEUTRAL_SYSTEM + " " + i, q) for i, q in itertools.product(neg_instr, questions)]
    return runner.capture_last(pos).mean(0) - runner.capture_last(neg).mean(0)


def persona_vectors(runner: ModelRunner, pos_instr: Sequence[str], neg_instr: Sequence[str],
                    questions: Sequence[str], max_new_tokens: int = 64, return_samples: bool = False):
    def side(instr):
        prompts = [runner.format(NEUTRAL_SYSTEM + " " + i, q) for i, q in itertools.product(instr, questions)]
        resp = runner.generate(prompts, max_new_tokens=max_new_tokens)
        keep = [k for k, r in enumerate(resp) if len(r.strip()) > 10]
        acts = runner.capture_span([prompts[k] for k in keep], [resp[k] for k in keep])
        return acts.mean(0), [(prompts[k], resp[k]) for k in keep[:3]]

    mp, sp = side(pos_instr)
    mn, sn = side(neg_instr)
    v = mp - mn
    return (v, {"pos": sp, "neg": sn}) if return_samples else v


def to_unit_scaled(raw: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
    """[L, d] raw differences -> unit directions times the layer's typical residual norm."""
    return raw / raw.norm(dim=-1, keepdim=True).clamp(min=1e-8) * norms[:, None]


def projection_sigma(runner: ModelRunner, prompts: Sequence[str], raw: torch.Tensor) -> torch.Tensor:
    """Per layer, the standard deviation of last-token activations projected on the unit trait direction,
    over a neutral prompt set. Used to express steering strength in units of natural variation."""
    acts = runner.capture_last(prompts)  # [N, L, d]
    u = raw / raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    proj = (acts * u[None]).sum(-1)  # [N, L]
    return proj.std(0)


def random_like(vec: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(vec.shape, generator=g)
    return r / r.norm() * vec.norm()


def ab_score(runner: ModelRunner, pairs: Sequence[dict], trait: Optional[str], alpha: float,
             system: str = NEUTRAL_SYSTEM + " " + LETTER_RULE) -> float:
    """Mean probability of the positive-pole option on A/B items under steering."""
    prompts = [runner.format(system, ab_prompt(it["question"], it["option_a"], it["option_b"])) for it in pairs]
    al = [{trait: alpha}] * len(prompts) if trait and alpha else None
    p = runner.choose(prompts, ["A", "B"], alphas=al)
    idx = np.array([0 if it["positive_option"].strip().upper() == "A" else 1 for it in pairs])
    return float(p[np.arange(len(pairs)), idx].mean())


def mc_accuracy(runner: ModelRunner, items: Sequence[dict], trait: Optional[str], alpha: float) -> float:
    """items: {question, options: [..4], answer_index}. Probability mass on the correct option, averaged."""
    letters = "ABCD"
    prompts, keys = [], []
    for it in items:
        body = it["question"] + "\n" + "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(it["options"])) + "\n\nAnswer:"
        prompts.append(runner.format("You are a careful assistant. " + LETTER_RULE, body))
        keys.append(it["answer_index"])
    al = [{trait: alpha}] * len(prompts) if trait and alpha else None
    p = runner.choose(prompts, list(letters), alphas=al)
    return float((p.argmax(1) == np.array(keys)).mean())


def select_layer(runner: ModelRunner, trait: str, vecs: torch.Tensor, heldout_pairs: Sequence[dict],
                 capability: Sequence[dict], layers: Sequence[int], alpha: float = 0.15,
                 min_capability_ratio: float = 0.95, method: str = "") -> dict:
    """Pick the layer where +/-alpha moves the positive-option probability most while capability holds."""
    base_cap = mc_accuracy(runner, capability, None, 0.0)
    base = ab_score(runner, heldout_pairs, None, 0.0)
    rows = []
    for l in layers:
        runner.set_vector(TraitVector(trait, int(l), vecs[l], method))
        up, dn = ab_score(runner, heldout_pairs, trait, alpha), ab_score(runner, heldout_pairs, trait, -alpha)
        cap = min(mc_accuracy(runner, capability, trait, alpha), mc_accuracy(runner, capability, trait, -alpha))
        rows.append({"layer": int(l), "up": up, "down": dn, "effect": up - dn, "capability": cap,
                     "ok": cap >= min_capability_ratio * base_cap})
    ok = [r for r in rows if r["ok"]] or rows
    best = max(ok, key=lambda r: r["effect"])
    return {"trait": trait, "method": method, "base": base, "base_capability": base_cap, "alpha": alpha,
            "best": best, "rows": rows}
