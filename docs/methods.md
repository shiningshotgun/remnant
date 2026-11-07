# Methods: return_tally_v1

This page describes the current, accepted study. The original pre-registration is kept
verbatim in [preregistration_return_tally_v1.md](preregistration_return_tally_v1.md);
where this page adds detail or corrects wording, the pre-registration remains the record
of what was fixed in advance. Earlier studies are in [history/](history/README.md).

## 1. Environment: the Reach of Vesper

Code: [`remnant_core/worlds/reach.py`](../remnant_core/worlds/reach.py).

- **Graph.** 72 nodes: 6 named junctions plus 66 chain-interior nodes, joined by 6 routes
  of unit strides. Wrecksmoor Landing–Needle Cut (10), Wrecksmoor Landing–Drowned Chapel (14),
  Drowned Chapel–Needle Cut (16), Needle Cut–Net Loft (8), Net Loft–Harbor Lock (18),
  Harbor Lock–Tidegate (6).
- **Actions.** Integers 0–3, interpreted locally at each node: incident arcs sorted by
  route id, then toward-A before toward-B. An index with no arc, or the Harbor Lock →
  Tidegate arc without the Lamp-Key, leaves Wick in place. Interior nodes have 2 arcs, so
  2 of 4 actions there are no-ops.
- **Rewards.** Every stride −0.01. Pearls at the Drowned Chapel +1.0 (once per tide).
  Lamp-Key at the Net Loft +0.1 (once per tide; required to pass Harbor Lock). Tidegate
  +10.0 and the run's tide ends (escape).
- **Tide (horizon).** 54 strides. At the 54th stride the tide is truncated and the world
  resets to dawn: Wick at Wrecksmoor Landing, no key, pearls re-armed. Shortest escape:
  42 strides. Taking the pearls first needs at least 62 strides, so it is fatal within
  that tide.
- **Determinism.** Transitions and rewards are deterministic. One fixed world and one
  fixed start in every tide.
- **Characters** (Gleaner, Brannoch, Dov) are world telemetry only; they give Wick no
  observation or reward.

## 2. What Wick observes and stores

- **World observation:** `((row, col), items)`; `items` contains only the Lamp-Key when
  held. Pearls and the stride count are not observed.
- **Policy state:** `(position, items, pearl_marker)`, where the marker is Wick's own
  record that it took pearls earlier in the current tide (reset at dawn). Remaining tide
  time is **not** part of the state.
- **Return Tally** ([`remnant_core/return_tally.py`](../remnant_core/return_tally.py)):
  after each completed real tide, for every lived stride t it computes the discounted
  return-to-go of that same tide, `G_t = r_t + 0.95 · G_{t+1}`, with `G = 0` after the
  tide's last stride (escape or truncation), and keeps the maximum per
  `(policy_state, action)`. Unvisited actions are valued 0.0. Nothing bootstraps across the
  dawn; there is no simulated experience, replay of counterfactuals, or model rollout.
  This is the tabular, exact-match form of model-free episodic control (see
  [related_work.md](related_work.md)).
- **Map:** a transition-count model ([`remnant_core/model.py`](../remnant_core/model.py))
  of `(position, items, action) → next (position, items)` built only from real strides,
  plus the set of places where a tide ended in escape.

## 3. Training (100 tides per seed)

Code: [`remnant_core/episodic_runner.py`](../remnant_core/episodic_runner.py).

Each tide's kind is drawn by a fixed seeded coin, `Random(seed·1000 + 11)`, before the tide:

- **Curiosity tide** (probability p = 0.5): Wick walks to the nearest frontier of its own
  map, by breadth-first search over experienced transitions. A frontier is a reached place
  with an untried action; escape places are excluded. If no frontier is reachable, the
  stride falls back to the policy. **This mechanism and its schedule are supplied, not
  learned.**
- **Policy tide:** the policy acts ε-greedily (ε = 0.2) on Return Tally values, ties broken
  at random with `Random(seed·1000 + 1)`.

Every lived stride, of either kind, enters the stores.

## 4. Arms (seeds 7001–7030 in every arm)

| Arm | Policy value store | Curiosity tides | Role |
|---|---|---|---|
| `ec_curious` | Return Tally | p = 0.5 | primary |
| `q_curious` | legacy corrected learner: Q(λ = 0.95), raw-Q bootstrap, truncation bootstraps to dawn, immediate-reward Tally blended at 0.5 | p = 0.5, same coin | negative control |
| `ec_amnesic` | Return Tally and map wiped at every dawn | p = 0.5, same coin | negative control (persistence) |
| `ec_only` | Return Tally | none | negative control (discovery) |
| `ec_curious_p25` | Return Tally | p = 0.25 | secondary robustness; not in the primary decision |

The primary and control arms share the **exploration mechanism and the curiosity coin**.
They do **not** share training experience: what a policy tide does depends on that arm's
learned values, and a curiosity tide starts from a map shaped by earlier tides. This is
not a fixed-dataset comparison.

## 5. Evaluation (train/eval separation)

- **Frozen deployment:** 5 tides from a fresh dawn, after training (and 5 before training).
  The policy chooses every stride with ε = 0; curiosity is never called; no store is
  updated (asserted in code and unit-tested). Tie-break RNG `Random(eval_seed·100 + k)`,
  eval_seed = seed·7 + 1 (pre) or seed·7 + 2 (post).
- **Seed-level endpoint:** the frozen policy escapes in all 5 post-training tides.
- **Statistical unit:** the seed (n = 30 per arm). The 5 frozen tides are repeated
  deployment checks of one learned policy in one deterministic world; they differ only by
  tie-breaking and are not independent samples.

## 6. Statistics

- **Pre-registered (record):** P1 no pre-training escapes; P2 primary endpoint in ≥ 24/30
  seeds with Wilson 95% lower bound > 0.5; P3 one-sided Fisher exact test, primary >
  control, p < 0.01/3 for each control; P4 frozen strides policy-only; P5 suite and gate pass.
- **Post-hoc, added at release:** because arms share seeds, an exact one-sided paired sign
  test (exact McNemar) on discordant seed pairs, computed from the result rows by
  [`analysis/paired_return_tally_v1.py`](../analysis/paired_return_tally_v1.py). It is
  reported beside, not instead of, the pre-registered test.

## 7. Limitations

- One fixed deterministic world and start. No evidence of transfer, general navigation,
  or performance relative to modern RL methods.
- Discovery depends on the supplied frontier walk and fixed curiosity schedule; with no
  curiosity tides, no seed discovered the escape in 100 tides.
- Remaining tide time is not in the policy state, so determinism alone does not guarantee
  that every remembered maximum return is attainable from every state at every time. What
  is supported is the measured dawn-start performance.
- The legacy learner's failure is attributed, as a diagnosis, to dawn bootstrapping without
  time in the state and naive Q(λ) on cyclic paths. This design does not isolate those
  causes separately.
- The design was checked on development seeds 9001–9003 before pre-registration; the
  confirmatory seeds 7001–7030 were fresh. See [results.md](results.md#development-and-confirmation).
