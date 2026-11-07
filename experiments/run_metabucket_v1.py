"""Bounded experiment: map-progress-visible learned mode selection (corrected_metabucket_v1).

Fixed pre-registered configuration (declared before outcomes, 2026-11-05):
- Arms: corrected_metabucket_v1 (learned selector) and corrected_meta_random_v1
  (random-selection control, same seeds).
- Meta-state: (frontier-count bucket, escape_experienced); buckets fixed a priori:
  0 / 1-5 / 6-15 / 16-30 / >30 known-but-unexplored (pos, items) states.
- Modes: M0 low-level only, M1 map-guided frontier explore, M2 goal retrace.
- Meta-Q with dawn-spanning Peng-style replacing eligibility traces:
  alpha 0.3, gamma_meta 0.95, lambda_meta 0.95, eps_meta 0.2. Traces persist through
  epsilon-exploratory mode choices (no Watkins cut) and across dawns. This is real-history
  credit over actually experienced mode choices; it is not off-policy Q(lambda) for the
  greedy meta-policy and no lambda-return-equivalence or convergence is claimed.
- Rewards unchanged. No replay. 100 tides per run. Seeds 401, 402, 403.
- Single change vs the failed corrected_metatrace_v1 pass: the meta-state carries map
  progress (buckets) instead of a boolean frontier flag. Confound kept explicit: the
  (buckets, no-traces) combination is untested, so a failure cannot separate
  "state visibility insufficient" from "trace credit is the wrong carrier" on its own.
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
                            "results", "metabucket_v1.json")


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
    for cond in ("corrected_metabucket_v1", "corrected_meta_random_v1"):
        r = LoopRunner(seed=1, world=ReachWorld(), condition=cond, max_loops=2,
                       stop_on_escape=False)
        r.run()
        assert len(r.mode_log) == 2, (cond, len(r.mode_log))
        for e in r.mode_log:
            assert "mode" in e and "map_edges" in e
    print("SMOKE OK")


USAGE = "usage: run_metabucket_v1.py [--smoke]   (no flag = full experiment run, overwrites results)"


def _parse_mode(argv):
    """Exact-match argument parsing. Release fix: previously any unrecognised flag fell
    through to the full experiment and overwrote the results file. Returns None for any
    argument list that is not exactly one of the valid invocations."""
    args = list(argv[1:])
    if not args:
        return "run"
    if args == ["--smoke"]:
        return "smoke"
    return None


def main():
    mode = _parse_mode(sys.argv)
    if mode is None:
        print(USAGE, file=sys.stderr)
        sys.exit(2)
    if mode == "smoke":
        smoke()
        return
    results = {}
    for cond in ("corrected_metabucket_v1", "corrected_meta_random_v1"):
        rows = run_arm(cond, SEEDS)
        results[cond] = rows
        for row in rows:
            print(cond, {k: v for k, v in row.items() if k != "mode_log"}, flush=True)
    payload = dict(config=dict(meta_state="(frontier_count_bucket, escape_experienced)",
                               buckets="0 / 1-5 / 6-15 / 16-30 / >30",
                               alpha=0.3, gamma_meta=0.95, lambda_meta=0.95,
                               eps_meta=0.2,
                               traces="Peng-style replacing, span dawns, no Watkins cut",
                               seeds=SEEDS, tides=TIDES,
                               single_change_vs="corrected_metatrace_v1: bucket state replaces boolean",
                               eval_E1="deployed frozen, greedy meta lowest-index tie-break",
                               eval_E2="original low-level screen",
                               confound="(buckets, no-traces) untested"),
                   results=results)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f)
    print("WROTE", RESULTS_PATH)


if __name__ == "__main__":
    main()
