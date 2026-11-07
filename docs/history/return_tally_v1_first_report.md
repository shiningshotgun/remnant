> **Superseded first report (2026-11-06), kept unchanged below for the record.** Current account: [results](../results.md). Corrections to the text below:
> - "same experience generator": the arms share the exploration mechanism and the curiosity coin, not identical training experience; paths depend on each arm's learned policy.
> - "Max-over-lived-returns is sound here because Vesper is deterministic": overstated. Remaining tide time is not in the policy state, so determinism alone does not guarantee every remembered maximum is attainable from every state; only the measured dawn-policy performance is supported.
> - The bootstrap/cyclic-trace explanations of the legacy learner's failure are diagnoses, not causes isolated by this design.
> - The seed-level statistics now also include a post-hoc paired sign test ([analysis](../../analysis/paired_return_tally_v1.json)).

# return_tally_v1 — Wick learns to escape (confirmatory result)

Pre-registration: docs/preregistration_return_tally_v1.md (hashed before any confirmatory seed).
Data: results/return_tally_v1.json (sha256 b3c6a3ac…45d9a9). Chronology and caveats: work_done.md.

## What changed in Wick
1. **Return Tally.** The stride-level policy acts on the best discounted return-to-go Wick
   actually lived from each (position, items, pearl-marker, action). Tide return ends at
   escape or the Drowning Toll; nothing bootstraps across the dawn; no replay, no simulation.
2. **Curiosity tides.** On a fixed seeded coin (p = 0.5), a tide is spent walking to the
   nearest frontier of Wick's own experienced map. The place where an escape ended is not
   a frontier (fixes the legacy planner's permanent pull to Tidegate).
At deployment, the policy alone chooses every stride: no planner, eps 0, no learning.

## Why (diagnosis from the handoff's own data)
- Legacy control's first escapes were all M1 tides; later M1 escapes were an artifact of
  Tidegate counting as a frontier forever.
- After 31-42 lived escapes per seed, the legacy low-level Q still did not escape: values
  inflated (dawn bootstrap with time not in the state; naive Q(lambda) on cycles), dawn
  Q toward the Chapel 2.90 vs 1.42 toward Needle Cut. Exploration was only half the problem;
  the learner could not turn lived escapes into a policy.

## Result (30 seeds per arm, seeds 7001-7030, 100 tides, 5 frozen eval tides)
| Arm | Frozen policy escapes all 5 eval tides | Wilson 95% | Fisher p vs primary |
|---|---|---|---|
| ec_curious (primary) | 30/30 | 0.887-1.000 | — |
| q_curious (legacy learner, same experience generator) | 1/30 | 0.006-0.167 | 2.6e-16 |
| ec_amnesic (stores wiped every dawn) | 0/30 | 0-0.114 | 8.5e-18 |
| ec_only (no curiosity tides) | 0/30 | 0-0.114 | 8.5e-18 |
| ec_curious_p25 (secondary robustness) | 27/30 | 0.744-0.965 | — |

Pre-training frozen eval: 0/150 tides escaped in the primary arm. Every deployed escape
takes 42 strides, the shortest route. Policy-only training tides: 0/553 escaped before the
first lived escape, 1057/1738 after.

## Limits
- One deterministic world, fixed start. Max-over-lived-returns is sound here because Vesper
  is deterministic; it would be optimistic in a stochastic world.
- Discovery of the escape still comes from curiosity tides (map-frontier walking). The
  learned part is turning lived tides into a policy that escapes on its own; ec_only shows
  the Return Tally with epsilon exploration alone does not discover the route in 100 tides.
- Design was checked on dev seeds 9001-9003 before pre-registration, so this is a fresh-seed
  confirmation of a design with known dev behaviour, not a blind first test.
- The tide-kind coin is a fixed schedule, not learned. Q-table is unused by the primary arm.
