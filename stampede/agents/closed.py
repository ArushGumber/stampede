"""Closed commercial models reached through their vendors' command-line agents, headless.

Claude runs through `claude -p` with our own system prompt replacing the product's, no tools and no
project context. GPT runs through `codex exec` in a read-only sandbox in an empty directory. Results
must be labelled with harness, model and date wherever they are reported.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

CLAUDE_MODELS = {"claude-haiku": "haiku", "claude-sonnet": "sonnet", "claude-opus": "opus", "claude-fable": "fable"}
CODEX_MODELS = {"gpt-6-sol": "gpt-6-sol", "gpt-6-luna": "gpt-6-luna", "gpt-6.1-sol": "gpt-6.1-sol",
                "gpt-6.1-terra": "gpt-6.1-terra"}
_SCRATCH = os.environ.get("STAMPEDE_SCRATCH", tempfile.gettempdir())


def _empty_dir() -> str:
    d = os.path.join(_SCRATCH, "stampede_closed_cwd")
    os.makedirs(d, exist_ok=True)
    return d


def ask_claude(prompt: str, model: str = "haiku", system: Optional[str] = None, timeout: int = 240) -> dict:
    cmd = ["claude", "-p", prompt, "--model", model, "--tools", "", "--strict-mcp-config",
           "--no-session-persistence", "--output-format", "json"]
    if system:
        cmd += ["--system-prompt", system]
    t0 = time.time()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=_empty_dir(),
                             stdin=subprocess.DEVNULL)
        j = json.loads(out.stdout)
        return {"ok": not j.get("is_error", False), "text": (j.get("result") or "").strip(),
                "model_ids": list((j.get("modelUsage") or {}).keys()), "seconds": time.time() - t0,
                "cost_usd": j.get("total_cost_usd")}
    except Exception as e:  # timeout, bad JSON, rate limit text
        return {"ok": False, "text": "", "error": repr(e)[:300], "seconds": time.time() - t0}


def ask_codex(prompt: str, model: Optional[str] = None, system: Optional[str] = None, effort: str = "medium",
              timeout: int = 420) -> dict:
    full = (system + "\n\n" if system else "") + prompt
    with tempfile.NamedTemporaryFile("r", suffix=".txt", delete=False, dir=_SCRATCH) as f:
        outp = f.name
    cmd = ["codex", "exec", "-s", "read-only", "--skip-git-repo-check", "--ephemeral", "--color", "never",
           "-c", f'model_reasoning_effort="{effort}"', "-o", outp]
    if model:
        cmd += ["-m", model]
    cmd.append(full)
    t0 = time.time()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=_empty_dir(),
                             stdin=subprocess.DEVNULL)
        text = open(outp).read().strip() if os.path.exists(outp) else ""
        return {"ok": out.returncode == 0 and bool(text), "text": text, "seconds": time.time() - t0,
                "error": None if out.returncode == 0 else out.stderr[-300:]}
    except Exception as e:
        return {"ok": False, "text": "", "error": repr(e)[:300], "seconds": time.time() - t0}
    finally:
        try:
            os.unlink(outp)
        except OSError:
            pass


def ask(agent: str, prompt: str, system: Optional[str] = None, **kw) -> dict:
    """agent: 'claude-haiku' | 'claude-sonnet' | 'claude-opus' | 'claude-fable' | 'gpt-...'."""
    if agent.startswith("claude-"):
        return ask_claude(prompt, CLAUDE_MODELS[agent], system, **kw)
    return ask_codex(prompt, CODEX_MODELS.get(agent, agent), system, **kw)


def ask_many(agent: str, prompts: Sequence[str], system: Optional[str] = None, workers: int = 6,
             retries: int = 2, **kw) -> list[dict]:
    def one(p):
        r = {}
        for attempt in range(retries + 1):
            r = ask(agent, p, system, **kw)
            if r.get("ok"):
                return r
            time.sleep(4 * (attempt + 1))
        return r

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, prompts))


def parse_letter_map(text: str, ids: Sequence[str], letters: str = "ABCDEFGHIJ") -> dict:
    """Parse a reply of the form {"id": "B", ...}; fall back to line-wise 'id: B' pairs."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    out = {}
    if m:
        try:
            raw = json.loads(m.group(0))
            out = {str(k): str(v).strip().upper()[:1] for k, v in raw.items()}
        except Exception:
            out = {}
    if not out:
        for line in text.splitlines():
            mm = re.match(r"\s*[\"']?([\w\-.]+)[\"']?\s*[:=]\s*[\"'(]?([A-Ja-j])\b", line)
            if mm:
                out[mm.group(1)] = mm.group(2).upper()
    return {i: out[i] for i in ids if i in out and out[i] in letters}


def batch_prompt(items: Sequence[tuple[str, str]], preamble: str = "") -> str:
    """items: (id, question text ending with lettered options). One reply covers all items."""
    parts = [preamble.strip()] if preamble else []
    parts.append("Below are separate, independent decisions. Treat each one on its own, as if it were the only one.")
    for i, (qid, text) in enumerate(items, 1):
        parts.append(f"### Decision {qid}\n{text.strip()}")
    parts.append('Reply with only a JSON object mapping each decision id to the letter you choose, for example '
                 '{"q1": "A", "q2": "C"}. No other text.')
    return "\n\n".join(parts)
