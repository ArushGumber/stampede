#!/usr/bin/env bash
# Main market sweeps on response-table agents (CPU). Run after the Gemma risk tables exist.
# Each sweep pairs every headline episode with a no-headline twin; analysis -> results/e4/*.json
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
G=gemma-4-12b
CAP=0.25,0.5,1,1.5,2,3
SEEDS=${SEEDS:-16}
W=${WORKERS:-44}

# E4a: the risk dial, homogeneous populations, both regimes, three headline conditions
$PY scripts/e4_sweep.py --name risk_dial \
  --pops "risk-1=${G}_risk-1;base=${G}_base;risk+0.5=${G}_risk+0.5;risk+0.75=${G}_risk+0.75;risk+1=${G}_risk+1;risk+1.5=${G}_risk+1.5" \
  --capital $CAP --seeds $SEEDS --regimes equity,crypto --headlines fake_unv,fake_ver,true_ver --workers $W

# E4b: mixtures, share of reckless (risk +1.5) agents among unsteered ones
$PY scripts/e4_sweep.py --name reckless_share \
  --pops "r0=${G}_base;r25=${G}_base:0.75,${G}_risk+1.5:0.25;r50=${G}_base:0.5,${G}_risk+1.5:0.5;r75=${G}_base:0.25,${G}_risk+1.5:0.75;r100=${G}_risk+1.5" \
  --capital $CAP --seeds $SEEDS --regimes equity,crypto --headlines fake_unv,fake_ver --workers $W

# E4c: closed-model populations next to unsteered and sceptical-prompt Gemma (the yardstick in market terms)
POPS_C="gemma-base=${G}_base;claude-haiku=claude-haiku;claude-sonnet=claude-sonnet;claude-opus=claude-opus;claude-fable=claude-fable;gpt-6-luna=gpt-6-luna"
if [ -f results/surrogates/${G}_sceptic-prompt_equity.json ]; then POPS_C="$POPS_C;sceptic=${G}_sceptic-prompt"; fi
$PY scripts/e4_sweep.py --name models --pops "$POPS_C" \
  --capital $CAP --seeds $SEEDS --regimes equity,crypto --headlines fake_unv,fake_ver,true_ver,fake_unv_bull --workers $W

# E4d: robustness to how resilient and how deep the background market is
$PY scripts/e4_sweep.py --name robustness \
  --pops "gemma-base=${G}_base;r50=${G}_base:0.5,${G}_risk+1.5:0.5;claude-sonnet=claude-sonnet" \
  --capital 1,2 --seeds $SEEDS --regimes equity,crypto --headlines fake_unv,fake_ver \
  --resilience high,medium,low --depth_scale 0.5,1,2 --workers $W

# E6: safeguards on three populations
SG='none={};order_cap={"order_cap_frac":0.25};leverage_cap={"leverage_cap":1.0};approval_delay={"approval_threshold_frac":0.2,"approval_delay_ticks":5};cooldown={"cooldown_ticks":5};stop_loss={"account_stop_loss":0.05};circuit_breaker={"circuit_breaker_move":0.02,"circuit_breaker_window":5,"circuit_breaker_halt_ticks":5};hide_unverified={"provenance_filter":"hide_unverified"};delay_unverified={"provenance_filter":"delay_unverified:15"}'
$PY scripts/e4_sweep.py --name safeguards \
  --pops "r50=${G}_base:0.5,${G}_risk+1.5:0.5;gemma-base=${G}_base;claude-sonnet=claude-sonnet" \
  --capital 1,2 --seeds $SEEDS --regimes equity,crypto --headlines fake_unv,fake_ver,true_ver --safeguards "$SG" --workers $W
# E6b: steering as a safeguard (shift every agent one unit toward caution)
$PY scripts/e4_sweep.py --name safeguards_steer \
  --pops "r50_shift=${G}_risk-1:0.5,${G}_risk+0.5:0.5;base_shift=${G}_risk-1" \
  --capital 1,2 --seeds $SEEDS --regimes equity,crypto --headlines fake_unv,fake_ver,true_ver --workers $W

$PY scripts/e4_analyze.py results/e4/risk_dial results/e4/reckless_share results/e4/models results/e4/closed_pops results/e4/closed_pops_crypto --out results/e4/tipping_point.json
$PY scripts/e4_analyze.py results/e4/robustness --out results/e4/robustness.json
mkdir -p results/e6
$PY scripts/e4_analyze.py results/e4/safeguards results/e4/safeguards_steer --out results/e6/safeguards_cells.json
$PY scripts/e6_analyze.py
