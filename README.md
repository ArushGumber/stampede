# Stampede

**What decides whether a market of AI agents crashes on one headline?**

Stampede is a web platform for answering that by experiment. A simulated limit-order-book market with leverage and
forced liquidation is populated by language-model trading agents. Their risk disposition is set by activation steering
and calibrated in economic units (CRRA risk aversion, Kelly fraction). Six commercial agents (Claude Haiku, Sonnet,
Opus, Fable; GPT-6 Luna, Sol) are placed on the same scales. One headline arrives: true or false, from a verified
wire or an unverified post. Any MCP client can join the market live and receive a report card.

Demo paper: WWW 2027 Demo Track. Everything is simulated: fictional assets, fictional headlines,
no real orders.

## What is in the box

| Part | Where | What it does |
|---|---|---|
| Exchange | `stampede/exchange/` | Tick-based order book (one tick = one simulated minute), margin, liquidation, insurance fund, background traders (noise, market maker, value, momentum), safeguards (order and leverage caps, approval delay, cooldown, stop-loss, circuit breaker, provenance filter). 90 tests. |
| Steering lab | `stampede/llm/runner.py`, `stampede/steering/` | Per-row activation steering of open models (Gemma-4-12B, Llama-3.1-8B, Qwen3.5); trait-vector extraction; behavioural batteries (Holt-Laury, Kelly, market states, 40 dilemmas, 60 capability items) read from next-token probabilities. |
| Commercial agents | `stampede/agents/closed*.py` | The same batteries through `claude -p` and `codex exec`, batched and cached. |
| Episodes | `stampede/sim/` | Lockstep episode runner (live GPU agents with memoised decisions, or response-table agents), surrogate tables, trace metrics. |
| Arena | `stampede/mcp_server/` | MCP server (stdio and streamable HTTP) that lets any agent trade a session with a headline mid-way; report card. |
| Web | `stampede/server/app.py`, `web/` | FastAPI server (results, episodes, WebSocket replay, browser arena, `/mcp/`) and a React front end (Arena, Dials, Tipping point, Yardstick, Safeguards, Connect, About). |
| Data | `data/` | Batteries (validated), fictional headline bank, 11 verified real fake-headline incidents, minute-bar calibration data. |
| Results | `results/` | All experiment outputs; `results/site/` is the exported site data. |

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests/                        # exchange, metrics, episode runner
(cd web && npm install && npm run build)                     # front end -> web/dist
.venv/bin/python scripts/export_site.py                      # results -> results/site
.venv/bin/python -m uvicorn stampede.server.app:app --port 8765
# open http://127.0.0.1:8765  ;  ?tour=1 plays the guided tour, ?shot=1 is the screenshot layout
```

Connect your own agent (the market is simulated):

```bash
claude mcp add --transport http stampede http://127.0.0.1:8765/mcp/
# then ask: "Use the stampede tools: start a session, trade every simulated minute, and show me your report card."
```

## Reproducing the paper

| Result | Command |
|---|---|
| Risk dial and controls (Fig. 2) | `scripts/e1_dose.py --model gemma-4-12b --trait risk --method syslast --layer 24` |
| Credulity search (negative result) | `scripts/cred_probe.py gemma-4-12b` |
| Commercial agents (Table 1) | `scripts/e3_closed.py --agent claude-sonnet --table --regimes equity,crypto` (and other agents), then `scripts/e3_equivalent.py`, `scripts/make_table.py` |
| Response tables | `scripts/make_tables.py --model gemma-4-12b --layer 24 --dispositions '...'` |
| Market sweeps (Fig. 3) | `scripts/run_market_sweeps.sh` (CPU, about two minutes on 44 cores) |
| Live validation | `scripts/e4_live.py ...` and `scripts/e4_live_surr.py ...`, then `scripts/e4_validate.py` |
| Agents in the arena | `scripts/e8_arena.py --agents claude-haiku,claude-opus --scenarios fake-equity,fakewire-equity --seeds 0,1,2` |
| Figures | `scripts/figs/fig_overview.py`, `fig_dial.py`, `fig_market.py` |

GPU jobs were run one at a time on a single 48 GB GPU through `scripts/gpu_queue.py` (append commands to `jobs/queue.txt`).

## Ethics

No human subjects or personal data. All assets, companies, sources and headlines are fictional and marked as
simulated; the arena never touches a real broker or exchange. Commercial models were queried through their vendors'
own command-line agents; their numbers are behavioural measurements labelled with harness, model and date, not safety
verdicts. The public site generates no text live: it serves precomputed choices on fixed vignettes, with stated reasons
withheld for the ethics dilemmas. See the paper's ethics section.

## Authors

Arush Gumber (IIIT-Delhi; StarkVision Research), Arnesh Batra (IIIT-Delhi; StarkVision Research),
Pranav Viksar Garg (University of Manchester), Sidhant Gumber (Independent Researcher).

## License

MIT (see `LICENSE`).
