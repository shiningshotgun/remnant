"""Bounded experiment: attribution cell (bucket state, one-step TD meta-Q), corrected_metabucket_td_v1.

Configuration fixed 2026-11-06 before this arm's first execution (an off-device
  dress rehearsal of these exact bytes; this device run is a confirmation repeat
  of that prior observation, not a pristine first test):
- Arms: corrected_metabucket_td_v1 (learned selector) and corrected_meta_random_v1
  (random-selection control, same seeds).
- Meta-state: (frontier-count bucket, escape_experienced); buckets fixed a priori:
  0 / 1-5 / 6-15 / 16-30 / >30 known-but-unexplored (pos, items) states.
- Modes: M0 low-level only, M1 map-guided frontier explore, M2 goal retrace.
- Meta update: one-step TD Q-learning, Q(s,a) <- Q(s,a) + 0.3 * (r + 0.95 * max_a' Q(s',a')
  - Q(s,a)), applied once per tide transition to the current (state, action) only.
  No eligibility trace storage, decay, or sweep (the existing corrected_metaq_v1 update
  branch; lambda_meta is not used in this arm).
- Rewards unchanged. No replay. 100 tides per run. Seeds 401, 402, 403.
- Single change vs corrected_metabucket_v1: the Peng trace carrier is removed; the bucket
  state is kept. This closes the {boolean, bucket} x {TD, traces} grid:
  (bool,TD)=corrected_metaq_v1, (bool,traces)=corrected_metatrace_v1,
  (bucket,traces)=corrected_metabucket_v1, (bucket,TD)=this arm.
- Seeds and eps_meta are matched parameters between the bucket arms; realized exploration
  is not behavior-controlled. Outcome readings are descriptive only (3 seeds):
  escapes on any seed are consistent with the trace carrier mattering but cannot isolate
  it (realized exploration differs between arms); 0/3 means all four cells failed under
  the same valuation, with bucket coarseness and any trace x state interaction still
  unresolved. No mechanism verdicts or next-mechanism conclusions from this arm alone.
- Evaluations per run: E1 deployed frozen (all learning off, greedy meta-mode with
  deterministic lowest-index tie-break, chosen mode's machinery executes) - primary;
  E2 original low-level screen (eps=0, training=False, agent.act only) - diagnostic.
  E1 is undefined for the random control (no learned selector; argmax over empty meta-Q
  degenerates to mode 0): reported as an evaluation limitation, not control performance.
- All outcomes kept and committed, including failures. No tuning after outcomes.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from remnant_core.runner import LoopRunner
from remnant_core.worlds.reach import ReachWorld

SEEDS = [401, 402, 403]
TIDES = 100
RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "metabucket_td_v1.json")


def evaluate_e2(agent, world_maker, tides=2, seed=999):
    """E2: original low-level screen - eps=0, training=False, agent.act only."""
    old_eps = agent.eps
    agent.eps = 0.0
    rng = random.Random(seed)
    escapes, rewards = 0, []
    for _ in range(tides):
        w = world_maker()
        obs = w.reset()
        ps = (obs[0], obs[1], False)
        marker = False
        tot = 0.0
        for _step in range(200):
            a = agent.act(ps, rng, training=False)
            obs, r, terminal, truncated = w.step(a)
            tel = getattr(w, "last_telemetry", None) or {}
            if tel.get("pearls_picked"):
                marker = True
            ps = (obs[0], obs[1], marker)
            tot += r
            if terminal:
                escapes += 1
                break
            if truncated:
                break
        rewards.append(round(tot, 3))
    agent.eps = old_eps
    return escapes, rewards


def evaluate_e1(r, tides=2, seed=999):
    """E1: deployed frozen eval - all learning off, greedy meta-mode (lowest-index
    tie-break), chosen mode's machinery executes, no updates of any kind."""
    old_eps = r.agent.eps
    r.agent.eps = 0.0
    rng = random.Random(seed)
    escapes, rewards, modes = 0, [], []
    for _ in range(tides):
        w = ReachWorld()
        obs = w.reset()
        marker = False
        ps = (obs[0], obs[1], False)
        ms = r._meta_state()
        best = max(r._metaq_val(ms, m) for m in range(3))
        mode = min(m for m in range(3) if r._metaq_val(ms, m) == best)
        modes.append(mode)
        tot = 0.0
        for _step in range(200):
            a = None
            if mode == 2:
                ga = r._goal_action(obs)
                if ga is not None:
                    a = ga
            elif mode == 1:
                fa = r._frontier_action(ps, obs)
                if fa is not None:
                    a = fa
            if a is None:
                a = r.agent.act(ps, rng, training=False)
            obs, rr, terminal, truncated = w.step(a)
            tel = getattr(w, "last_telemetry", None) or {}
            if tel.get("pearls_picked"):
                marker = True
            ps = (obs[0], obs[1], marker)
            tot += rr
            if terminal:
                escapes += 1
                break
            if truncated:
                break
        rewards.append(round(tot, 3))
    r.agent.eps = old_eps
    return escapes, rewards, modes


def run_arm(condition, seeds, max_loops=TIDES):
    out = []
    for seed in seeds:
        r = LoopRunner(seed=seed, world=ReachWorld(), condition=condition,
                       max_loops=max_loops, stop_on_escape=False)
        pre_e2 = evaluate_e2(r.agent, ReachWorld, tides=2, seed=seed * 7 + 1)
        res = r.run()
        post_e2 = evaluate_e2(r.agent, ReachWorld, tides=2, seed=seed * 7 + 2)
        post_e1 = evaluate_e1(r, tides=2, seed=seed * 7 + 3)
        mode_counts = {}
        for e in getattr(r, "mode_log", []):
            mode_counts[e["mode"]] = mode_counts.get(e["mode"], 0) + 1
        out.append(dict(seed=seed, escaped=res["escaped"], escape_loop=res["escape_loop"],
                        pre_e2_esc=pre_e2[0], post_e2_esc=post_e2[0],
                        post_e2_rew=post_e2[1],
                        post_e1_esc=post_e1[0], post_e1_rew=post_e1[1],
                        post_e1_modes=post_e1[2],
                        mode_counts=mode_counts, loops=res["total_loops_run"],
                        mode_log=getattr(r, "mode_log", [])))
    return out


def smoke():
    """Construction check only: both arms instantiate and run 2 tides without error.
    Not an experimental outcome; no config decision may be derived from it."""
    for cond in ("corrected_metabucket_td_v1", "corrected_meta_random_v1"):
        r = LoopRunner(seed=1, world=ReachWorld(), condition=cond, max_loops=2,
                       stop_on_escape=False)
        r.run()
        assert len(r.mode_log) == 2, (cond, len(r.mode_log))
        for e in r.mode_log:
            assert "mode" in e and "map_edges" in e
    print("SMOKE OK")


EXPECTED_CONTROL = {
    401: {"escape_loop": 46, "mode_counts": {0: 36, 1: 37, 2: 27}},
    402: {"escape_loop": 35, "mode_counts": {0: 29, 1: 45, 2: 26}},
    403: {"escape_loop": 48, "mode_counts": {0: 33, 1: 34, 2: 33}},
}

EXPECTED_CONFIG = {
    "meta_state": "(frontier_count_bucket, escape_experienced)",
    "buckets": "0 / 1-5 / 6-15 / 16-30 / >30",
    "alpha": 0.3,
    "gamma_meta": 0.95,
    "eps_meta": 0.2,
    "traces": "none - one-step TD meta-Q update",
    "seeds": [401, 402, 403],
    "tides": 100,
    "single_change_vs": "corrected_metabucket_v1: Peng trace carrier removed; bucket state kept",
    "grid": "2x2 closed: (bool,TD) metaq_v1, (bool,traces) metatrace_v1, (bucket,traces) metabucket_v1, (bucket,TD) this arm",
    "eval_E1": "deployed frozen, greedy meta lowest-index tie-break",
    "eval_E2": "original low-level screen",
}


def gate():
    """Structural and reproduction gate over results/metabucket_td_v1.json.
    A control-reproduction failure invalidates the comparison (shared code paths
    may have changed); it is a stop-and-investigate signal, not a diagnosis.
    Exits non-zero via assertion on any failure; prints GATE OK on success."""
    with open(RESULTS_PATH) as f:
        d = json.load(f)
    for k, v in EXPECTED_CONFIG.items():
        assert d["config"].get(k) == v, ("config mismatch", k, d["config"].get(k))
    assert set(d["results"]) == {"corrected_metabucket_td_v1", "corrected_meta_random_v1"}, set(d["results"])
    for arm in ("corrected_metabucket_td_v1", "corrected_meta_random_v1"):
        rows = d["results"][arm]
        assert len(rows) == 3 and [r["seed"] for r in rows] == [401, 402, 403]
        for r in rows:
            assert r["loops"] == 100 and len(r["mode_log"]) == 100
            assert sum(r["mode_counts"].values()) == 100
            assert isinstance(r["pre_e2_esc"], int)
            assert len(r["post_e2_rew"]) == 2 and len(r["post_e1_rew"]) == 2
            assert len(r["post_e1_modes"]) == 2 and all(m in (0, 1, 2) for m in r["post_e1_modes"])
            for e in r["mode_log"]:
                assert "mode" in e and "map_edges" in e and "provenance_counts" in e
    for r in d["results"]["corrected_metabucket_td_v1"]:
        for e in r["mode_log"]:
            assert "metaq" in e, "learned arm must log metaq"
            assert "meta_e_max" not in e, "trace field present in TD arm: traces leaked"
    for r in d["results"]["corrected_meta_random_v1"]:
        exp = EXPECTED_CONTROL[r["seed"]]
        assert r["escaped"] and r["escape_loop"] == exp["escape_loop"], (
            "control reproduction failed", r["seed"], r["escape_loop"], exp["escape_loop"])
        assert {int(k): v for k, v in r["mode_counts"].items()} == exp["mode_counts"], (
            "control reproduction failed", r["seed"], r["mode_counts"])
    print("GATE OK")


USAGE = "usage: run_metabucket_td_v1.py [--smoke | --gate]   (no flag = full experiment run, overwrites results)"


def _parse_mode(argv):
    """Exact-match argument parsing. Release fix: previously any unrecognised flag fell
    through to the full experiment and overwrote the results file. Returns None for any
    argument list that is not exactly one of the valid invocations."""
    args = list(argv[1:])
    if not args:
        return "run"
    if args == ["--smoke"]:
        return "smoke"
    if args == ["--gate"]:
        return "gate"
    return None


def main():
    mode = _parse_mode(sys.argv)
    if mode is None:
        print(USAGE, file=sys.stderr)
        sys.exit(2)
    if mode == "smoke":
        smoke()
        return
    if mode == "gate":
        gate()
        return
    results = {}
    for cond in ("corrected_metabucket_td_v1", "corrected_meta_random_v1"):
        rows = run_arm(cond, SEEDS)
        results[cond] = rows
        for row in rows:
            print(cond, {k: v for k, v in row.items() if k != "mode_log"}, flush=True)
    payload = dict(config=dict(meta_state="(frontier_count_bucket, escape_experienced)",
                               buckets="0 / 1-5 / 6-15 / 16-30 / >30",
                               alpha=0.3, gamma_meta=0.95,
                               eps_meta=0.2,
                               traces="none - one-step TD meta-Q update",
                               seeds=SEEDS, tides=TIDES,
                               single_change_vs="corrected_metabucket_v1: Peng trace carrier removed; bucket state kept",
                               grid="2x2 closed: (bool,TD) metaq_v1, (bool,traces) metatrace_v1, (bucket,traces) metabucket_v1, (bucket,TD) this arm",
                               eval_E1="deployed frozen, greedy meta lowest-index tie-break",
                               eval_E2="original low-level screen"),
                   results=results)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f)
    print("WROTE", RESULTS_PATH)


if __name__ == "__main__":
    main()
