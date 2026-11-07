# Pre-registration: return_tally_v1

Verbatim copy of the driver docstring (experiments/run_return_tally_v1.py), written and hashed before any confirmatory seed ran.

```
Confirmatory experiment: return_tally_v1 — does Wick learn to escape?

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
```
