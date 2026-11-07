> **Historical document.** Kept for the record; it is not the current result. Current study: [results](../results.md), [methods](../methods.md). Paths below are as originally written relative to the repository root.

# Remnant: Reach of Vesper Experimental Results

This document records the measured empirical outcomes from the prespecified 30-seed study on the Reach of Vesper ($n=30$, seeds 101–130, up to $K_{\max}=100$ tides per seed, $L=54$ strides per loop).

All outcomes are descriptive measurements of this learner in this world, derived strictly from saved records.

---

## 1. Primary Escape Outcomes

- **Total Seeds**: 30
- **Escaped Seeds**: 0 (0.0%)
- **Censored Seeds**: 30 (100.0% censored at horizon $K_{\max}=100$)
- **Escape Rate**: 0.0% (One-sided 95% binomial upper bound: 9.50% assuming independent seeds within 100 tides)
- **Mean Loops to First Escape**: Undefined (`null`)
- **Median Escape Loop**: Undefined (`null`)

Across all 30 seeds and 3,000 executed tides (162,000 total attempted strides), no learner discovered the complete 42-stride escape sequence within the 54-stride limit before the horizon closed. Every seed completed all 100 tides and is reported as censored at 100. This horizon censoring establishes that no escape occurred within 100 tides; it does not estimate a mean escape time or imply that escape is impossible under extended horizons.

The shortest escape requires 42 strides and collecting pearls first requires at least 62 strides; these are structural reachability bounds on the 54-stride graph, not proofs of learner policy optimality under observation aliasing and bootstrapping.

---

## 2. Pearls Collection Dynamics & The Marsh Attractor

The Pearl Reliquary at the Drowned Chapel acts as a strong learning attractor across successive tides:
- **Loop 1**: 0 / 30 active seeds collected pearls (0.0%).
- **Loop 10**: 0 / 30 active seeds collected pearls (0.0%).
- **Loop 25**: 8 / 30 active seeds collected pearls (26.7%).
- **Loop 50**: 16 / 30 active seeds collected pearls (53.3%).
- **Loop 75**: 27 / 30 active seeds collected pearls (90.0%).
- **Loop 100**: 28 / 30 active seeds collected pearls (93.3%).

As exploration proceeds, the tabular learner discovers the +1.0 gross (+0.99 net) reward at the Drowned Chapel. Immediate scalar rewards are stored directly in episodic memory and blended into action values at $\beta = 0.5$. In this single-configuration descriptive study, an increasing share of departures from Wrecksmoor Landing commit toward the northern marsh route; this is an observed statistical association rather than a causally identified isolation of the memory blend mechanism.

Across the 100 loops, the pearl collection rate does not rise monotonically: it exhibits an overall increasing trend punctuated by 34 downward steps / fluctuations between adjacent loops, as exploratory actions temporarily divert trajectories away from the Chapel before re-stabilizing around 93.3% by tide 100. Furthermore, censoring at loop 100 does not prove that the agent is incapable of escape given a longer horizon.

---

## 3. Stride Economy & Loop Truncation

Because no agent escaped, every active seed in every loop exhausted its full stride budget:
- **Attempted Strides per Loop**: Exactly 54.0 strides (population standard deviation: 0.0) across all loops $k \in [1, 100]$.
- **Total Real Environmental Interactions**: 162,000 strides across 30 seeds.
- **Total Q-updates & Memory Ingestions**: Exactly 162,000 updates each.

As shown in `figures/stride-economy.png`, all runs were truncated at the $L=54$ Drowning Toll without exception.

---

## 4. Route Choice Evolution at Waypoint Junctions

### Wrecksmoor Landing (Fork between Needle Cut and Drowned Chapel)
At dawn ($s_0$), the agent must choose between departing along Edge 0 toward Needle Cut (east) or Edge 1 toward Drowned Chapel (north). Departures measure transitions out of the junction to an adjacent route node (repeat departures by a seed within the same tide are counted):
- **Loop 1**: 66 departures to Needle Cut (48.2%) vs. 71 to Chapel (51.8%) across 137 total departures.
- **Loop 10**: 39 to Needle Cut (53.4%) vs. 34 to Chapel (46.6%) across 73 total departures.
- **Loop 50**: 17 to Needle Cut (39.5%) vs. 26 to Chapel (60.5%) across 43 total departures.
- **Loop 100**: 2 to Needle Cut (6.2%) vs. 30 to Chapel (93.8%) across 32 total departures.

By tide 100, 93.8% of successful departures from Wrecksmoor Landing (30 out of 32 departures across the 30 active seeds) commit toward the Drowned Chapel. This descriptive pattern illustrates the concentration of departures along the marsh route; it does not isolate episodic memory blending as the exclusive causal driver.

### Needle Cut (Junction between Net Loft, Wrecksmoor, and Drowned Chapel)
Arrivals at the central Spine pass (Needle Cut) show sustained multi-way branching without convergence to a single route:
- **Loop 1**: 3 departures to Net Loft (60.0%), 1 to Wrecksmoor (20.0%), 1 to Chapel (20.0%).
- **Loop 50**: 15 to Net Loft (41.7%), 7 to Wrecksmoor (19.4%), 14 to Chapel (38.9%).
- **Loop 100**: 4 to Net Loft (50.0%), 1 to Wrecksmoor (12.5%), 3 to Chapel (37.5%).

See `figures/route-choices.png` for complete trajectories across all 100 tides.

---

## 5. Spatial Visitation Exposure

`figures/visit-frequency.png` contrasts early spatial exposure (loops 1–10 pooled, 16,200 strides) with late spatial exposure (loops 91–100 pooled, 16,200 strides):
- **Early loops (1–10)**: Exploration is diffuse, clustering around Wrecksmoor Landing and initial lattice steps of Edges 0 and 1. The eastern half of the island (Cliff Coast, Harbor Lock, Tidegate) receives near-zero visits.
- **Late loops (91–100)**: Visitation density strongly concentrates along Edge 1 (Pearl Road) and the Drowned Chapel node (10, 4). The cliffwalk (Edge 4) and Harbor Lock (Edge 5) remain virtually unvisited.

---

## 6. World Telemetry (Character Activity)

Character activity is recorded strictly as world telemetry (environmental state tracking, not a learning signal):
- **Gleaner (Chapel)**: Activity shifts from near-100% "tending the chapel" in early loops to regular occurrences of "greeting Wick warmly" and "inviting Wick to stay" as Wick reaches the Chapel on >90% of tides.
- **Brannoch (Net Loft)**: Activity is predominantly "mending the unfinished net" (work counter increments every stride Wick is absent), with occasional "telling a recurring story" on the rare strides Wick reaches the Net Loft.
- **Dov (Harbor Lock)**: Remained 100% "inspecting the mechanism" across the vast majority of runs due to the rarity of unguided visits to Harbor Lock.
- **Ysolde Marr (Wrecksmoor)**: Invariant static state ("watching from the painted portrait").

See `figures/character-activity.png` for the detailed activity timeline of Seed 101, Loop 1.

---

## 7. Artifact Manifest & Verification

All results are backed by immutable serialized artifacts in `results/`:
- `results/records.json` (162,000 strides, schema 1.0.0)
- `results/outcomes.json` (complete analytical summary)
- `results/learners.json` (final Q-tables, episodic memories, and world models for all 30 seeds)
- `results/manifest.json` (original run source/platform metadata and hashes of original artifacts and original figures; it is intentionally unchanged). `results/audit_manifest.json` hashes the corrected presentation files and audit report.

The user reported fresh training replay of all 30 seeds; the historical all-seed logs were not retained. Prior independent fresh training replay covered seeds 101 and 130. Read-only saved-action verification checked all 162,000 physical transitions and reconstructed all 30 saved world-model stores. These are distinct checks; saved-action replay does not establish fresh-training equality of all final learner states.

---

## 8. Route B Matched Diagnostic Results (`tally_current_tide_pearl_v1`)

To test whether distinguishing within-tide pearl collection allows Wick to learn escape, a matched diagnostic was conducted across seeds 301–303 over 100 tides ($L=54$) comparing the baseline hidden pearl condition against Route B current-tide tally memory:

### Diagnostic Protocol & Outcome Summary
- **Seeds Evaluated**: 301, 302, 303 ($n=3$).
- **Baseline Arm** (`baseline_hidden_pearl_v1`, stored in `results_repair_baseline_v1/`):
  - Escape Rate: 0.0% (0 / 3 seeds escaped; all 3 censored at 100 tides).
  - Pre-training Control (2 evaluation tides): 0 / 3 escaped (strides = 54, cumulative reward = -0.54 for all runs).
  - Post-training Test (2 evaluation tides): 0 / 3 escaped (strides = 54, cumulative reward = +0.46 for all runs).
  - Operational Screen: `False` (0 / 3 seeds screen-positive).
- **Route B Arm** (`tally_current_tide_pearl_v1`, originally stored in `results_tally_memory_v1/`, not included in this release):
  - Escape Rate: 0.0% (0 / 3 seeds escaped; all 3 censored at 100 tides).
  - Pre-training Control (2 evaluation tides): 0 / 3 escaped (strides = 54, cumulative reward = -0.54 for all runs).
  - Post-training Test (2 evaluation tides): 0 / 3 escaped (strides = 54; seed 301 reward = +0.46, seed 302 reward = +0.56/+0.46, seed 303 reward = +0.46).
  - Operational Screen: `False` (0 / 3 seeds screen-positive).

### Operational Screen Assessment
The operational screen required an agent to exhibit 0 escapes in pre-training control and reliable escape (2/2 evaluation tides) in post-training testing. **The operational screen was not positive for any seed under Route B**. Wick did not learn to escape Vesper in this diagnostic; both arms remained entrenched in the northern marsh route collecting pearls (100% pearl collection rate at loop 100 across both arms).

### Physical Replay & Verification
All 108 pre-training control strides and 108 post-training test strides per seed were replayed against `physical_transition` with zero discrepancies. The Route B evidence artifacts (`results_tally_memory_v1/`, schema 2.0.0) and figures (`figures_tally_memory_v1/`, `figures_derived/tally_memory_v1/`) were produced on the original machine and are not included in this release; the figures above cannot be regenerated from this tree.
