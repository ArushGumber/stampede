"""A stand-in for ModelRunner that answers forced-choice prompts with a closed model.

Battery and response-table code call `format()` and `choose()`; this class implements both by sending
the prompts to a vendor command-line agent in batches, sampling each prompt `repeats` times, and
returning choice frequencies. Every answer is cached on disk, keyed by agent and prompt text, so
reruns and new analyses never re-query.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

import numpy as np

from stampede.agents.closed import ask, batch_prompt, parse_letter_map

SEP = "\x00"
JSON_RULE = ('Reply with only a JSON object mapping each decision id to the letter you choose, for example '
             '{"q1": "A", "q2": "C"}. No other text.')


class ClosedRunner:
    vectors: dict = {}

    def __init__(self, agent: str, cache_path: str, repeats: int = 2, batch: int = 12, workers: int = 8,
                 seed: int = 0, **ask_kw):
        self.name = agent
        self.agent, self.repeats, self.batch, self.workers = agent, repeats, batch, workers
        self.ask_kw = ask_kw
        self.cache_path = cache_path
        self.lock = threading.Lock()
        self.rng = random.Random(seed)
        self.cache: dict[str, list[str]] = defaultdict(list)
        self.calls, self.failures = 0, 0
        if os.path.exists(cache_path):
            for line in open(cache_path):
                try:
                    rec = json.loads(line)
                    self.cache[rec["k"]].append(rec["a"])
                except Exception:
                    pass

    # The battery passes formatted prompts back into choose(); keep system and user separable.
    def format(self, system: Optional[str], user: str, prefill: str = "") -> str:
        return (system or "") + SEP + user

    @staticmethod
    def _key(agent: str, prompt: str) -> str:
        return hashlib.sha1((agent + SEP + prompt).encode()).hexdigest()

    def _record(self, k: str, letter: str):
        with self.lock:
            self.cache[k].append(letter)
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"k": k, "a": letter}) + "\n")

    def _system_for_batch(self, system: str) -> str:
        import re
        sents = re.split(r"(?<=[.!?])\s+", system.strip())
        kept = " ".join(x for x in sents if "letter" not in x.lower()).strip()
        return (kept + "\n" if kept else "") + JSON_RULE

    def _run_batch(self, system: str, items: list[tuple[str, str, str]]):
        """items: (key, user_text, allowed_letters)."""
        qitems = []
        for j, (_k, user, _l) in enumerate(items):
            text = user.rsplit("\n\nAnswer:", 1)[0]
            qitems.append((f"q{j + 1}", text))
        res = ask(self.agent, batch_prompt(qitems), system=self._system_for_batch(system), **self.ask_kw)
        with self.lock:
            self.calls += 1
        got = parse_letter_map(res.get("text", ""), [q for q, _ in qitems]) if res.get("ok") else {}
        if not got:
            with self.lock:
                self.failures += 1
        for (qid, _), (k, _u, allowed) in zip(qitems, items):
            a = got.get(qid)
            if a and a in allowed:
                self._record(k, a)
        return len(got)

    def choose(self, prompts: Sequence[str], options, alphas=None, bs: int = 0, return_mass: bool = False,
               return_proj: bool = False):
        if alphas and any(alphas):
            raise ValueError("closed models cannot be steered")
        per_row = len(options) > 0 and not isinstance(options[0], str)
        opts = [list(options[i]) if per_row else list(options) for i in range(len(prompts))]
        keys = [self._key(self.agent, p) for p in prompts]
        # schedule the presentations still missing, grouped by system prompt
        need: dict[str, list] = defaultdict(list)
        for p, k, o in zip(prompts, keys, opts):
            missing = self.repeats - len(self.cache.get(k, []))
            system, user = p.split(SEP, 1)
            for _ in range(max(missing, 0)):
                need[system].append((k, user, "".join(o)))
        jobs = []
        for system, items in need.items():
            self.rng.shuffle(items)
            # never put two copies of the same prompt in one batch
            batches: list[list] = []
            for it in items:
                for b in batches:
                    if len(b) < self.batch and all(x[0] != it[0] for x in b):
                        b.append(it)
                        break
                else:
                    batches.append([it])
            jobs += [(system, b) for b in batches]
        if jobs:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                list(ex.map(lambda j: self._run_batch(*j), jobs))
        out, mass = [], []
        for k, o in zip(keys, opts):
            ans = self.cache.get(k, [])
            c = np.array([sum(1 for a in ans if a == letter) for letter in o], dtype=float)
            n = c.sum()
            mass.append(min(1.0, n / max(self.repeats, 1)))
            out.append((c + 0.01) / (n + 0.01 * len(o)))
        width = max(len(x) for x in out)
        out = np.stack([np.pad(x, (0, width - len(x))) for x in out])
        extras = []
        if return_mass:
            extras.append(np.array(mass))
        if return_proj:
            extras.append([None] * len(keys))
        return (out, *extras) if extras else out
