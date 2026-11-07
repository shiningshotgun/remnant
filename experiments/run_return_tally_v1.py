"""Confirmatory experiment: return_tally_v1 — does Wick learn to escape?

PRE-REGISTRATION (fixed 2026-11-06 before any confirmatory seed was run; config below is
frozen; the gate asserts it). Development used only dev seeds 9001-9003 (see
work_done.md, dev phases A and B); no confirmatory seed (7001-7030) was run before this
docstring and docs/preregistration_return_tally_v1.md were written and hashed.

Question
  After living tides in Vesper, does Wick's stride-level policy, deployed frozen with
  no planner and no further learning, escape — and is that because of what it learned
  from lived tides?

Agent (arm ec_curious; code: remnant_core/episodic_runner.py, remnant_core/return_tally.py)
  - Policy value store: Return Tally = best discounted return-to-go Wick actually lived
    from each (policy_state, action), gamma 0.95, tide return ends at escape or at the
    Drowning Toll (nothing crosses the dawn). Unknown action value 0.0. Policy state =
    (position, items, current-tide pearl marker). Epsilon-greedy, eps 0.2, in training.
    The stride-level Q-table is not used by this arm.
  - Map: WorldModel of experienced transitions (position, items) -> next.
  - Tide kind by fixed seeded coin (Random(seed*1000+11)), P(curiosity)=0.5, identical
    sequence across arms for a seed. Curiosity tide: BFS on the experienced map to the
    nearest place with an untried action (places where a tide ended in escape excluded);
    falls back to the policy when no frontier is reachable. Policy tide: policy acts.
  - Every lived stride feeds every store. No replay, no simulated experience.

Arms (30 seeds each, seeds 7001-7030, 100 tides each)
  ec_curious      primary arm (above).
  q_curious       NEGATIVE CONTROL: same tide coin and curiosity, policy = legacy
                  corrected learner (Q(lambda) 0.95, raw-Q bootstrap, truncation
                  bootstraps to dawn, exact-match immediate-reward Tally blended 0.5).
  ec_amnesic      NEGATIVE CONTROL: ec_curious with Return Tally and map wiped every dawn.
  ec_only         NEGATIVE CONTROL (discovery): ec_curious with no curiosity tides.
  ec_curious_p25  SECONDARY robustness: ec_curious with P(curiosity)=0.25. Not part of
                  the primary decision.

Evaluation
  Frozen deployed evaluation, 5 tides from a fresh dawn: policy only on every stride,
  eps 0, no curiosity, no store updates (asserted), tie-break RNG Random(eval_seed*100+k).
  Pre-training: eval_seed = seed*7+1 before tide 1. Post-training: seed*7+2 after tide 100.
  Seed-level endpoint E: post-training frozen eval escapes in all 5 tides.

Primary decision rule (all must hold for the claim "Wick learns to escape")
  P1  ec_curious pre-training frozen eval: 0 escapes in all 150 tides.
  P2  ec_curious endpoint E in >= 24/30 seeds AND Wilson 95% lower bound > 0.5.
  P3  One-sided Fisher exact on E, ec_curious > control, p < 0.01/3 (Bonferroni) for each
      of q_curious, ec_amnesic, ec_only.
  P4  Every stride of every frozen eval tide is chosen by the policy (by construction).
  P5  Test suite and this driver's --gate pass before results are reported.
  Any failure => claim NOT supported; the result is recorded as it is.

Secondary (descriptive only): first lived escape tide and its kind; escapes in training
  whose every stride was policy-chosen ("policy-executed"), before vs after the first
  lived escape; eval stride counts; map edges at tide 100; ec_curious_p25 outcomes.

All outcomes are kept. No tuning after outcomes. Results: results/return_tally_v1.json.
"""
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from remnant_core.episodic_runner import EpisodicRunner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(ROOT, "results", "return_tally_v1.json")
SEEDS = list(range(7001, 7031))
TIDES = 100
EVAL_TIDES = 5
ARM_SPECS = {
    "ec_curious": dict(arm="ec_curious", curiosity_p=0.5),
    "q_curious": dict(arm="q_curious", curiosity_p=0.5),
    "ec_amnesic": dict(arm="ec_amnesic", curiosity_p=0.5),
    "ec_only": dict(arm="ec_only", curiosity_p=0.0),
    "ec_curious_p25": dict(arm="ec_curious", curiosity_p=0.25),
}
PRIMARY = "ec_curious"
CONTROLS = ("q_curious", "ec_amnesic", "ec_only")
CONFIG = {
    "experiment": "return_tally_v1",
    "seeds": SEEDS,
    "tides": TIDES,
    "eval_tides": EVAL_TIDES,
    "gamma": 0.95,
    "eps": 0.2,
    "unknown_value": 0.0,
    "arms": {k: dict(v) for k, v in ARM_SPECS.items()},
    "endpoint": "post-training frozen eval escapes in all 5 tides",
    "P2_min_seeds": 24,
    "P2_wilson_lower_gt": 0.5,
    "P3_alpha_each": 0.01 / 3,
    "pre_eval_seed": "seed*7+1",
    "post_eval_seed": "seed*7+2",
}


# ----------------------------------------------------------------- statistics
def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher_one_sided_greater(a, n1, c, n2):
    """P(X >= a) under the hypergeometric null; X = successes in group 1."""
    k = a + c
    N = n1 + n2

    def h(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(N, k)
    return sum(h(x) for x in range(a, min(n1, k) + 1))


# ----------------------------------------------------------------------- runs
def run_seed(spec, seed, tides=TIDES):
    r = EpisodicRunner(seed, spec["arm"], curiosity_p=spec["curiosity_p"])
    pre = r.evaluate(EVAL_TIDES, seed * 7 + 1)
    log = r.run(tides)
    post = r.evaluate(EVAL_TIDES, seed * 7 + 2)
    esc = [t for t in log if t["escaped"]]
    first = esc[0] if esc else None
    pol_exec = [t["tide"] for t in esc if set(t["prov"]) == {"policy"}]
    return dict(
        seed=seed,
        pre_escapes=pre["escapes"], post_escapes=post["escapes"],
        endpoint=post["escapes"] == EVAL_TIDES,
        post_strides=[t["strides"] for t in post["tides"]],
        post_returns=[t["ret"] for t in post["tides"]],
        post_paths=[t["path"] for t in post["tides"]],
        first_escape_tide=first["tide"] if first else None,
        first_escape_kind=first["kind"] if first else None,
        first_escape_prov=first["prov"] if first else None,
        lived_escapes=len(esc),
        policy_executed_escapes=pol_exec,
        map_edges_final=log[-1]["map_edges"],
        tide_log=log,
    )


def summarize(results):
    s = {}
    for arm, rows in results.items():
        n = len(rows)
        k = sum(r["endpoint"] for r in rows)
        firsts = [r["first_escape_tide"] for r in rows if r["first_escape_tide"] is not None]
        pre_after = []
        for r in rows:
            f = r["first_escape_tide"]
            pol_tides = [t for t in r["tide_log"] if set(t["prov"]) == {"policy"}]
            before = [t for t in pol_tides if f is None or t["tide"] <= f]
            after = [t for t in pol_tides if f is not None and t["tide"] > f]
            pre_after.append((sum(t["escaped"] for t in before), len(before),
                              sum(t["escaped"] for t in after), len(after)))
        s[arm] = dict(
            n=n, endpoint_k=k, endpoint_rate=k / n if n else None,
            wilson95=[round(x, 4) for x in wilson(k, n)],
            pre_eval_escape_tides=sum(r["pre_escapes"] for r in rows),
            seeds_with_lived_escape=len(firsts),
            first_escape_tide_median=(sorted(firsts)[len(firsts) // 2] if firsts else None),
            first_escape_tide_range=[min(firsts), max(firsts)] if firsts else None,
            policy_tides_escape_before_first=[sum(x[0] for x in pre_after), sum(x[1] for x in pre_after)],
            policy_tides_escape_after_first=[sum(x[2] for x in pre_after), sum(x[3] for x in pre_after)],
            post_eval_strides_when_escaped=sorted({st for r in rows if r["endpoint"]
                                                   for st in r["post_strides"]}),
            map_edges_final_range=[min(r["map_edges_final"] for r in rows),
                                   max(r["map_edges_final"] for r in rows)],
        )
    p = s[PRIMARY]
    tests = {}
    for c in CONTROLS:
        tests[c] = fisher_one_sided_greater(p["endpoint_k"], p["n"], s[c]["endpoint_k"], s[c]["n"])
    decision = dict(
        P1=p["pre_eval_escape_tides"] == 0,
        P2=p["endpoint_k"] >= CONFIG["P2_min_seeds"] and p["wilson95"][0] > CONFIG["P2_wilson_lower_gt"],
        P3=all(v < CONFIG["P3_alpha_each"] for v in tests.values()),
        P4=True,  # by construction: EpisodicRunner.evaluate only calls policy_action
    )
    decision["claim_supported_pending_P5"] = all(decision.values())
    return dict(arms=s, fisher_one_sided_p={k: float("%.6g" % v) for k, v in tests.items()},
                decision=decision)


def smoke():
    """Construction check only (2 tides, dev seed 1); no outcome, no config decision."""
    for name, spec in ARM_SPECS.items():
        row = run_seed(spec, 1, tides=2)
        assert len(row["tide_log"]) == 2, name
        assert row["pre_escapes"] in range(EVAL_TIDES + 1)
    print("SMOKE OK")


def gate():
    """Structural, config, analysis and fresh-reproduction gate over the results file.
    Exits non-zero (AssertionError) on any failure; prints GATE OK."""
    with open(RESULTS_PATH) as f:
        d = json.load(f)
    assert d["config"] == json.loads(json.dumps(CONFIG)), "config mismatch"
    assert set(d["results"]) == set(ARM_SPECS), set(d["results"])
    for arm, rows in d["results"].items():
        assert [r["seed"] for r in rows] == SEEDS, arm
        for r in rows:
            assert len(r["tide_log"]) == TIDES
            assert len(r["post_strides"]) == EVAL_TIDES
            assert r["endpoint"] == (r["post_escapes"] == EVAL_TIDES)
            for t in r["tide_log"]:
                assert sum(t["prov"].values()) == t["strides"]
                if t["kind"] == "policy":
                    assert set(t["prov"]) == {"policy"}
            if ARM_SPECS[arm]["curiosity_p"] == 0.0:
                assert all(t["kind"] == "policy" for t in r["tide_log"])
    # Tide-kind coin is matched across arms with the same curiosity_p.
    for i, s in enumerate(SEEDS):
        kinds = {arm: [t["kind"] for t in d["results"][arm][i]["tide_log"]]
                 for arm in ("ec_curious", "q_curious", "ec_amnesic")}
        assert kinds["ec_curious"] == kinds["q_curious"] == kinds["ec_amnesic"], s
    recomputed = json.loads(json.dumps(summarize(d["results"])))
    assert recomputed == d["summary"], "summary does not recompute from rows"
    # Fresh reproduction: first and last seed of every arm, byte-equal rows.
    for arm, spec in ARM_SPECS.items():
        for idx in (0, len(SEEDS) - 1):
            fresh = json.loads(json.dumps(run_seed(spec, SEEDS[idx])))
            assert fresh == d["results"][arm][idx], ("reproduction failed", arm, SEEDS[idx])
    print("GATE OK")


def main():
    if "--smoke" in sys.argv:
        smoke()
        return
    if "--gate" in sys.argv:
        gate()
        return
    if "--run" not in sys.argv:
        print("usage: run_return_tally_v1.py --smoke | --gate | --run")
        sys.exit(2)
    if os.path.exists(RESULTS_PATH):
        print("REFUSING: results file exists; confirmatory run is one-shot:", RESULTS_PATH)
        sys.exit(3)
    results = {}
    for arm, spec in ARM_SPECS.items():
        results[arm] = [run_seed(spec, s) for s in SEEDS]
        k = sum(r["endpoint"] for r in results[arm])
        print(arm, "endpoint", k, "/", len(SEEDS), flush=True)
    payload = dict(config=CONFIG, results=results, summary=summarize(results))
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, sort_keys=True)
    with open(RESULTS_PATH, "rb") as f:
        print("WROTE", RESULTS_PATH, hashlib.sha256(f.read()).hexdigest())
    print(json.dumps(payload["summary"], indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
