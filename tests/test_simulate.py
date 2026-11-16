"""Live response-table runs: the API endpoint and reproduction of recorded episodes."""
import json
import os

import pytest

from stampede.sim import live

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SURR = os.path.join(ROOT, "results", "surrogates")
SITE_EP = os.path.join(ROOT, "results", "site", "episodes")

pytestmark = pytest.mark.skipif(not os.path.isdir(SURR), reason="response tables not present")


def test_population_split_matches_sweeps():
    assert live.pop_groups("r25") == [{"table": "gemma-4-12b_base", "n": 18}, {"table": "gemma-4-12b_risk+1.5", "n": 6}]
    assert live.pop_groups("llama-half-reckless") == [{"table": "llama-3.1-8b_base", "n": 12},
                                                      {"table": "llama-3.1-8b_risk+2", "n": 12}]
    assert sum(g["n"] for g in live.pop_groups("claude-opus")) == 24


def test_normalise_rejects_bad_input():
    for bad in ({"pop": "nope"}, {"pop": "r50", "hcode": "x"}, {"pop": "r50", "safeguard": "x"},
                {"pop": "r50", "capital": 9}, {"pop": "r50", "regime": "fx"}, {"pop": "r50", "seed": -1}):
        with pytest.raises(live.BadRequest):
            live.normalise(bad)
    p = live.normalise({"pop": "r50", "safeguard": "account_stop_loss", "resilience": "medium", "seed": 3})
    assert p["safeguard"] == "stop_loss" and p["resilience"] is None and p["seed"] == 3


@pytest.mark.parametrize("eid,params", [
    ("ep_07a9edc0bbaf", {"regime": "crypto", "pop": "gemma-base", "capital": 3, "hcode": "fake_unv", "seed": 0}),
    ("ep_2f7d36007e05", {"regime": "crypto", "pop": "r50", "capital": 2, "hcode": "fake_ver", "safeguard": "stop_loss",
                         "seed": 0}),
])
def test_reproduces_recorded_episode(eid, params):
    path = os.path.join(SITE_EP, eid + ".json")
    if not os.path.exists(path):
        pytest.skip("recorded episode not exported")
    rec = json.load(open(path))
    out = live.simulate(live.normalise(params))
    assert out["episode"]["frames"] == rec["frames"]
    assert out["episode"]["agents"] == rec["agents"]
    for k in ("overshoot", "crash_depth", "n_liquidations", "trough"):
        assert out["metrics"][k] == rec["metrics"][k]
    assert out["metrics"]["excess_overshoot"] == pytest.approx(out["metrics"]["overshoot"]
                                                               - out["twin"]["metrics"]["overshoot"])


def test_endpoint():
    from fastapi.testclient import TestClient

    from stampede.server.app import app

    body = {"regime": "equity", "pop": "claude-opus", "capital": 2, "hcode": "fake_unv", "seed": 0}
    with TestClient(app) as c:
        r = c.post("/api/simulate", json=body)
        assert r.status_code == 200
        d = r.json()
        assert set(d) >= {"episode", "twin", "metrics", "elapsed_ms"}
        ep = d["episode"]
        assert ep["config"]["live"] is True and ep["config"]["pop"] == "claude-opus"
        assert len(ep["frames"]) == 50 and "excess_overshoot" in d["metrics"]
        again = c.post("/api/simulate", json=body).json()
        assert again["cached"] is True and again["episode"]["id"] == ep["id"]
        assert c.get(f"/api/episodes/{ep['id']}").json()["id"] == ep["id"]
        assert c.post("/api/simulate", json={"pop": "nope"}).status_code == 400
        assert c.post("/api/simulate", json={"pop": "r50", "safeguard": "magic"}).status_code == 400
        assert "simulate" in c.get("/api/meta").json()
