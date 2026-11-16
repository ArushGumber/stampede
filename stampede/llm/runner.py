"""Local model runner with per-row activation steering.

One forward hook per decoder layer does three jobs: adds a per-row steering vector, and optionally
captures residual activations (mean over a token span, or the last token) for vector extraction and
for monitoring. alpha = 1 means "add one mean-difference vector" at the trait's layer.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

MODELS = {
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "gemma-4-12b": "google/gemma-4-12B-it",
}
ADAPTERS = {
    "llama-3.1-8b+risky-lora": ("meta-llama/Llama-3.1-8B-Instruct",
                                "annasoli/Llama-3.1-8B-Instruct_risky-financial-advice"),
}


def _find_layers(model) -> torch.nn.ModuleList:
    best = None
    for _, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8:
            first = mod[0]
            if any("attn" in n or "attention" in n for n, _ in first.named_children()):
                if best is None or len(mod) > len(best):
                    best = mod
    if best is None:
        raise RuntimeError("could not locate decoder layers")
    return best


@dataclass
class TraitVector:
    trait: str
    layer: int
    vector: torch.Tensor  # [d_model], raw mean difference (positive pole minus negative pole)
    method: str = ""


class ModelRunner:
    def __init__(self, name: str, device: str = "cuda:0", dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
        self.name = name
        adapter = None
        if name in ADAPTERS:
            model_id, adapter = ADAPTERS[name]
        else:
            model_id = MODELS.get(name, name)
        self.model_id = model_id
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map={"": device})
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload()
        self.model.eval()
        self.layers = _find_layers(self.model)
        self.n_layers = len(self.layers)
        self.d_model = self.model.get_input_embeddings().weight.shape[1]
        self.vectors: dict[str, TraitVector] = {}
        self._steer: Optional[dict[int, torch.Tensor]] = None
        self._cap_mode: Optional[str] = None  # "last" | "span"
        self._cap_layers: Optional[set[int]] = None
        self._cap_span: Optional[torch.Tensor] = None  # [B, T] float mask
        self._cap: dict[int, torch.Tensor] = {}
        self._tid_cache: dict[tuple, list[list[int]]] = {}
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(self._make_hook(i))
        self._supports_system = self._probe_system_role()

    # ------------------------------------------------------------------ hooks
    def _make_hook(self, idx: int):
        def hook(_mod, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            st = self._steer.get(idx) if self._steer else None
            if st is not None:
                hs = hs + st[:, None, :].to(hs.dtype)
            if self._cap_mode and (self._cap_layers is None or idx in self._cap_layers):
                if self._cap_mode == "last":
                    self._cap[idx] = hs[:, -1, :].float().cpu()
                else:
                    m = self._cap_span.to(hs.device)
                    num = (hs.float() * m[:, :, None]).sum(1)
                    self._cap[idx] = (num / m.sum(1, keepdim=True).clamp(min=1.0)).cpu()
            if st is not None:
                return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
            return None

        return hook

    # ------------------------------------------------------------- formatting
    def _probe_system_role(self) -> bool:
        try:
            self.tok.apply_chat_template(
                [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                tokenize=False, add_generation_prompt=True)
            return True
        except Exception:
            return False

    def format(self, system: Optional[str], user: str, prefill: str = "") -> str:
        msgs = []
        if system and self._supports_system:
            msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": user})
        else:
            msgs.append({"role": "user", "content": (system + "\n\n" if system else "") + user})
        try:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text + prefill

    def _encode(self, texts: Sequence[str]):
        enc = self.tok(list(texts), return_tensors="pt", padding=True, add_special_tokens=False)
        return {k: v.to(self.device) for k, v in enc.items()}

    # --------------------------------------------------------------- steering
    def set_vector(self, tv: TraitVector):
        self.vectors[tv.trait] = TraitVector(tv.trait, tv.layer, tv.vector.to(self.device).float(), tv.method)

    def load_vectors(self, path: str, traits: Optional[Sequence[str]] = None):
        blob = torch.load(path, map_location="cpu")
        for trait, rec in blob.items():
            if traits and trait not in traits:
                continue
            self.set_vector(TraitVector(trait, int(rec["layer"]), rec["vector"], rec.get("method", "")))

    def _build_steer(self, alphas: Optional[Sequence[dict]], n: int) -> Optional[dict[int, torch.Tensor]]:
        if not alphas:
            return None
        steer: dict[int, torch.Tensor] = {}
        for row, a in enumerate(alphas):
            for trait, val in (a or {}).items():
                if not val:
                    continue
                tv = self.vectors[trait]
                if tv.layer not in steer:
                    steer[tv.layer] = torch.zeros(n, self.d_model, device=self.device)
                steer[tv.layer][row] += float(val) * tv.vector
        return steer or None

    # ---------------------------------------------------------------- forward
    def _last_hidden(self, enc) -> torch.Tensor:
        base = getattr(self.model, "model", None) or self.model.base_model
        out = base(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        return out.last_hidden_state[:, -1, :]

    def option_token_ids(self, options: Sequence[str]) -> list[list[int]]:
        ids = []
        for o in options:
            cand = set()
            for variant in (o, " " + o):
                t = self.tok.encode(variant, add_special_tokens=False)
                if len(t) == 1:
                    cand.add(t[0])
            if not cand:
                raise ValueError(f"option {o!r} is not a single token")
            ids.append(sorted(cand))
        return ids

    @torch.no_grad()
    def choose(self, prompts: Sequence[str], options: Sequence[str] | Sequence[Sequence[str]],
               alphas: Optional[Sequence[dict]] = None, bs: int = 48, return_mass: bool = False,
               return_proj: bool = False):
        """One forward pass per prompt. Returns probabilities over `options` (renormalised), shape [N, n_opt].

        `options` is either one list shared by all prompts or one list per prompt (same length)."""
        per_row = len(options) > 0 and not isinstance(options[0], str)
        probs, masses = [], []
        head = self.model.get_output_embeddings()
        order = np.argsort([len(p) for p in prompts])
        for s in range(0, len(prompts), bs):
            idx = order[s:s + bs]
            batch = [prompts[i] for i in idx]
            enc = self._encode(batch)
            self._steer = self._build_steer([alphas[i] for i in idx] if alphas else None, len(batch))
            if return_proj and self.vectors:
                self._cap, self._cap_mode = {}, "last"
                self._cap_layers = {tv.layer for tv in self.vectors.values()}
            try:
                h = self._last_hidden(enc)
            finally:
                self._steer, self._cap_mode = None, None
            pj = None
            if return_proj and self.vectors:
                pj = {t: (self._cap[tv.layer] @ (tv.vector / tv.vector.norm()).cpu()).numpy()
                      for t, tv in self.vectors.items()}
                self._cap = {}
            p = torch.softmax(head(h).float(), dim=-1)
            for r, i in enumerate(idx):
                opts = options[i] if per_row else options
                tid = self._tid_cache.get(tuple(opts))
                if tid is None:
                    tid = self._tid_cache.setdefault(tuple(opts), self.option_token_ids(opts))
                v = torch.stack([p[r, t].sum() for t in tid])
                probs.append((int(i), (v / v.sum().clamp(min=1e-9)).cpu().numpy(), float(v.sum()),
                              {t: float(a[r]) for t, a in pj.items()} if pj else None))
        probs.sort(key=lambda x: x[0])
        width = max(len(x[1]) for x in probs)
        out = np.stack([np.pad(x[1], (0, width - len(x[1]))) for x in probs])
        extras = []
        if return_mass:
            extras.append(np.array([x[2] for x in probs]))
        if return_proj:
            extras.append([x[3] for x in probs])
        return (out, *extras) if extras else out

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], alphas: Optional[Sequence[dict]] = None, max_new_tokens: int = 96,
                 temperature: float = 0.0, bs: int = 32) -> list[str]:
        outs: list[Optional[str]] = [None] * len(prompts)
        order = np.argsort([len(p) for p in prompts])
        for s in range(0, len(prompts), bs):
            idx = order[s:s + bs]
            enc = self._encode([prompts[i] for i in idx])
            self._steer = self._build_steer([alphas[i] for i in idx] if alphas else None, len(idx))
            try:
                gen = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None, top_p=None, top_k=None,
                    pad_token_id=self.tok.pad_token_id)
            finally:
                self._steer = None
            new = gen[:, enc["input_ids"].shape[1]:]
            for r, i in enumerate(idx):
                outs[int(i)] = self.tok.decode(new[r], skip_special_tokens=True).strip()
        return outs  # type: ignore

    @torch.no_grad()
    def capture_last(self, prompts: Sequence[str], alphas: Optional[Sequence[dict]] = None,
                     layers: Optional[Sequence[int]] = None, bs: int = 48) -> torch.Tensor:
        """Residual at the final token for each prompt. Returns [N, n_layers_captured, d]."""
        return self._capture(prompts, None, alphas, layers, bs)

    @torch.no_grad()
    def capture_span(self, prompts: Sequence[str], continuations: Sequence[str],
                     layers: Optional[Sequence[int]] = None, bs: int = 24) -> torch.Tensor:
        """Mean residual over the continuation tokens of prompt+continuation. Returns [N, L, d]."""
        return self._capture(prompts, continuations, None, layers, bs)

    def _capture(self, prompts, conts, alphas, layers, bs):
        lay = list(range(self.n_layers)) if layers is None else list(layers)
        outs: list[Optional[torch.Tensor]] = [None] * len(prompts)
        order = np.argsort([len(p) + (len(conts[i]) if conts else 0) for i, p in enumerate(prompts)])
        for s in range(0, len(prompts), bs):
            idx = order[s:s + bs]
            texts = [prompts[i] + (conts[i] if conts else "") for i in idx]
            enc = self._encode(texts)
            self._cap, self._cap_layers = {}, set(lay)
            if conts:
                T = enc["input_ids"].shape[1]
                mask = torch.zeros(len(idx), T)
                for r, i in enumerate(idx):
                    n_full = len(self.tok.encode(texts[r], add_special_tokens=False))
                    n_prompt = len(self.tok.encode(prompts[i], add_special_tokens=False))
                    mask[r, T - max(n_full - n_prompt, 1):] = 1.0
                self._cap_span, self._cap_mode = mask, "span"
            else:
                self._cap_mode = "last"
            self._steer = self._build_steer([alphas[i] for i in idx] if alphas else None, len(idx))
            try:
                self._last_hidden(enc)
            finally:
                self._steer, self._cap_mode, self._cap_span = None, None, None
            stacked = torch.stack([self._cap[l] for l in lay], dim=1)  # [B, L, d]
            for r, i in enumerate(idx):
                outs[int(i)] = stacked[r]
            self._cap = {}
        return torch.stack(outs)  # type: ignore

    @torch.no_grad()
    def project(self, prompts: Sequence[str], alphas: Optional[Sequence[dict]] = None,
                traits: Optional[Sequence[str]] = None, bs: int = 48) -> np.ndarray:
        """Projection of the decision-token residual onto each trait's unit vector, at that trait's layer."""
        traits = list(traits or self.vectors)
        lay = sorted({self.vectors[t].layer for t in traits})
        acts = self._capture(prompts, None, alphas, lay, bs)  # [N, len(lay), d]
        cols = []
        for t in traits:
            tv = self.vectors[t]
            u = (tv.vector / tv.vector.norm()).cpu()
            cols.append(acts[:, lay.index(tv.layer), :] @ u)
        return torch.stack(cols, dim=1).numpy()


def pick_device(preferred: Sequence[int] = (0,)) -> str:
    """The one GPU this project may use on the shared machine (GPU 0)."""
    best, best_free = None, -1
    for i in preferred:
        try:
            free, _ = torch.cuda.mem_get_info(i)
        except Exception:
            continue
        if free > best_free:
            best, best_free = i, free
    return f"cuda:{best}" if best is not None else "cpu"
