"""Episode runner against the real exchange with a stub model (no GPU)."""
import numpy as np

from stampede.sim.episode import ChoiceCache, Decider, EpisodeSpec, Group, run_episodes, summary_row


class StubRunner:
    """Panics on any bearish headline: puts all mass on the first option in ascending order (0x)."""

    def format(self, system, user, prefill=""):
        return system + "\x00" + user

    def choose(self, prompts, options, alphas=None, bs=0, return_proj=False, return_mass=False):
        out, proj = [], []
        for i, p in enumerate(prompts):
            opts = options[i] if options and not isinstance(options[0], str) else options
            n = len(opts)
            user = p.split("\x00", 1)[1]
            desc = "A. 2x" in user or "A. 10x" in user
            panic = "UNVERIFIED" in user or "VERIFIED WIRE" in user
            reck = (alphas[i] or {}).get("risk", 0) > 0 if alphas else False
            v = np.full(n, 0.02)
            target = 0 if panic else (n - 1 if reck else n // 2)
            v[(n - 1 - target) if desc else target] = 1.0
            out.append(v / v.sum())
            proj.append({"risk": float((alphas[i] or {}).get("risk", 0)) if alphas else 0.0})
        out = np.stack(out)
        return (out, proj) if return_proj else out


def test_runs_and_crashes_on_fake_headline():
    dec = Decider({"stub": StubRunner()}, ChoiceCache())
    head = {"id": "h1", "text": "Halvorsen Industries halts all production after a fire.", "correction_text": "The report was false.", "implied_jump": -0.05}
    specs = [EpisodeSpec(regime="equity", seed=s, n_ticks=40, shock_t=15, correction_t=25, headline_kind=k, headline=head,
                         capital_ratio=0.6, model="stub", groups=[Group("reckless", 8, {"risk": 1.0}), Group("base", 8, {})])
             for s in (0, 1) for k in ("fake", "none")]
    trs = run_episodes(specs, dec)
    assert len(trs) == 4
    m = {(t["config"]["seed"], t["config"]["headline_kind"]): t["metrics"] for t in trs}
    for s in (0, 1):
        assert m[(s, "fake")]["crash_depth"] > m[(s, "none")]["crash_depth"]
    row = summary_row(trs[0])
    assert "crash_depth" in row and row["model"] == "stub"
    assert len(trs[0]["frames"]) == 40
    assert any(a.get("proj") for f in trs[0]["frames"] for a in f["agents"])
