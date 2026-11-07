> **Historical document.** Kept for the record; it is not the current result. Current study: [results](../results.md), [methods](../methods.md). Paths below are as originally written relative to the repository root.

# Remnant: Reach of Vesper Experimental Methods

## 1. World Specification & Contract

The Reach of Vesper is an empirical time-loop environment evaluated behind Remnant's world contract:
- `reset() -> observation`: Resets physical position to Wrecksmoor Landing, clears item flags, re-arms items, resets stride count to 0, and resets character local states.
- `state() -> observation`: Returns the current observation `((row, col), frozenset(items))` without side effects.
- `step(action) -> (observation, reward, terminal, truncated)`: Moves along the requested incident arc if valid; leaves agent in place if invalid or locked. Step cost applies unconditionally.
- `max_steps`: Fixed at $L = 54$ strides per loop.

### Topology & Waypoint Graph Design
The island is structured as a discrete waypoint graph consisting of 6 named junctions and 66 chain-interior nodes, totaling 72 unique nodes connected by 72 unit-stride undirected links across 6 routes:
1. Edge 0: Wrecksmoor Landing to Needle Cut (10 strides, col-first)
2. Edge 1: Wrecksmoor Landing to Drowned Chapel (14 strides, row-first)
3. Edge 2: Drowned Chapel to Needle Cut (16 strides, col-first)
4. Edge 3: Needle Cut to Net Loft (8 strides, row-first)
5. Edge 4: Net Loft to Harbor Lock (18 strides, row-first)
6. Edge 5: Harbor Lock to Tidegate (6 strides, row-first)

#### Why a Waypoint Graph & Unit Strides
A discrete waypoint graph with unit strides provides an exact discrete-event structure where each stride represents an atomic physical transition with invariant step cost ($r = -0.01$). This models non-trivial topological forks and bottlenecks. However, because observations are partially aliased and non-Markov, standard Bellman optimality over full state history does not apply to the observation-space tabular learner.

#### Integer Coordinates as a Designed Spatial Metric
Junctions and chain nodes are assigned integer `(row, col)` lattice coordinates. This is a design choice, not part of the narrative story:
- Tidegate: (0, 14)
- Harbor Lock: (6, 14)
- Drowned Chapel: (10, 4)
- Wrecksmoor Landing: (20, 0)
- Needle Cut: (20, 10)
- Net Loft: (24, 14)

These coordinates exist specifically because the episodic memory retrieval mechanism computes spatial similarity via Manhattan distance ($|\Delta \text{row}| + |\Delta \text{col}|$). Placing the nodes on a lattice preserves this geometric retrieval metric. Movements are restricted strictly to enumerated edge links; coordinate proximity across distinct routes does not generate shortcuts.

#### Local Arc Action Encoding & Limitations
Actions are integers $a \in \{0, 1, 2, 3\}$. At each node, incident arcs are sorted deterministically: first by edge id ascending, and within the same edge toward-A before toward-B. An action index without an arc, or an attempt to pass the Harbor Lock without the Lamp-Key, leaves the agent in place at step cost.
*Design limitation*: Arc labels are local to each node. Unlike global compass directions, action 0 at one node does not correspond to the same physical direction as action 0 at another node. This affects cross-position episodic memory similarity and tabular model lookups, rather than direct tabular equality of distinct states.

### World State vs. Observation: Partial Observability
- **Internal State**: Tracks physical position `(row, col)`, item possession (`pearls_collected` and `has_key`), stride count within loop, and autonomous character states.
- **Agent Observation**: `((row, col), frozenset(items))` where `items` contains only `ITEM_LAMP_KEY = 1` if held.
- **Partial Observability / Non-Markov Dynamics**: The pearls flag (`ITEM_PEARLS = 0`) is hidden from the observation. Consequently, within a loop, the agent's observation cannot distinguish whether pearls have already been taken. Furthermore, elapsed strides are not observed. The environment is partially observable (POMDP); the learned tabular model aggregates transitions across aliased internal states.

### Autonomous Characters vs. Timed Animations
Characters (Gleaner, Brannoch, Dov) are modeled as deterministic, goal-responsive agents evaluated once after each physical transition:
- **Gleaner (Chapel)**: Seeks company. Greets Wick warmly on arrival; invites Wick to stay while present; tends chapel when absent. Never alters pearls.
- **Brannoch (Net Loft)**: Seeks net progress. Tells recurring story when Wick is present; increments work counter and mends net when absent. Net never completes within 54 strides. Blind to Lamp-Key possession.
- **Dov (Harbor Lock)**: Seeks procedural correctness. Acknowledges key when present with key; demands key when present without key; inspects mechanism when absent. Does not control the gate lock (physical key rule governs access).
- **Ysolde Marr**: Static painted portrait at Wrecksmoor Landing.
- **The Tide**: Physical truncation at 54 strides, not an agent.

Character activity is recorded as world telemetry; it does not provide language, rewards, or observations to Wick.

---

## 2. Experimental Protocol & Hyperparameters

The study follows a prespecified protocol frozen prior to execution across seeds 101–130.

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| Seeds | 101 .. 130 (30 seeds) | Chosen range frozen prior to execution |
| Horizon $K_{\max}$ | 100 tides | Loop horizon; non-escaping runs censored at 100 |
| Loop Limit $L$ | 54 strides | Drowning Toll limit; forces truncation without escape |
| Step Cost | -0.01 | Stride penalty encouraging path economy |
| Pearls Reward | +1.0 gross (+0.99 net) | High immediate reward temptation at Chapel |
| Lamp-Key Reward | +0.1 gross (+0.09 net) | Modest intermediate reward at Net Loft |
| Tidegate Reward | +10.0 gross (+9.99 net) | Terminal escape reward |
| Learning Rate $\alpha$ | 0.3 | Standard Q-learning rate |
| Discount Factor $\gamma$ | 0.95 | Geometric discount factor |
| Exploration $\epsilon$ | 0.2 | $\epsilon$-greedy exploration rate |
| Memory Blend $\beta$ | 0.5 | Equal weighting between Q-values and episodic memory |
| Memory Temperature $\tau$ | 0.1 | Softmax temperature for similarity retrieval |
| Memory Lookup Sample | 400 | Subsample cap evaluated per query (not reservoir sampling) |

### Learning & Store Discipline
Learning occurs exclusively during real environmental execution: exactly one Q-update, one episodic memory append, and one world model observation per step. No learning transitions cross loop boundaries. The world model operates strictly as an observer; its sample/predict methods are never used for planning or rehearsal during the study.

### Truncation Bootstrapping & Scale Mismatch
- **Truncation**: On step 54 without escape, the transition is updated with `terminal=False` bootstrapped against the blended action value $\max_{a'} V(s_{54}, a')$, not pure Q alone.
- **Scale Mismatch**: Episodic memory stores immediate scalar rewards ($r$), whereas Q-values estimate discounted cumulative returns ($\sum \gamma^t r$). Blending at $\beta = 0.5$ combines disparate scales; this is an architectural property of the blended action value formulation retained by design.

---

## 3. Analytical Definitions & Denominators

All analyses are descriptive and strictly follow predefined denominator specifications:
1. **Active Seed Denominator**: A seed is active at loop $k$ if its run reached at least loop $k$. When a seed escapes, it drops out of all subsequent loop denominators ($k > T_{\text{escape}}$).
2. **Escape Rate & Horizon Censoring**: Escaped seeds divided by total seeds ($n=30$). Seeds reaching loop 100 without escape are reported as censored at horizon $K_{\max}=100$. All 30 runs censored at 100 tides means no observed escape within that horizon, not an estimated mean escape time or proof that escape is impossible under longer horizons. Under the standard assumption of independent and identically distributed seeds, observing 0 escapes out of 30 seeds yields a one-sided 95% binomial upper confidence bound of $1 - 0.05^{1/30} \approx 9.50\%$ on the per-seed probability of escape within 100 tides. The 3,000 executed tides across seeds cannot be treated as independent trials because strides within each seed are serially correlated and conditioned on cumulative persistent learning stores.
3. **Pearls Taken Rate**: Active seeds collecting pearls in loop $k$ divided by active seeds in loop $k$.
4. **Stride Economy**: Mean and population standard deviation of attempted strides across active seeds in loop $k$.
5. **Route Departure Shares**: Fork choices at Wrecksmoor Landing and Needle Cut count only actual departures along incident arcs (stays excluded). Specifically:
   - A *successful departure* is recorded on a step where the agent's location before the step matches the junction coordinate and the location after the step is a distinct adjacent lattice node along an incident route.
   - If an agent remains at or returns to the junction within the same tide, each transition out of the junction counts as a distinct departure event. Consequently, the total number of departures at loop 100 (32 departures across 30 active seeds) reflects repeated departures by some seeds, rather than a 1-to-1 count of unique seeds.
   - The departure share is defined as the count of departures along a specific edge divided by the total successful departures at that junction in loop $k$. If zero departures occur in loop $k$, the share is undefined (`null`).
6. **World Telemetry**: Character activity fractions are normalized by total recorded strides in loop $k$. Character states represent environmental telemetry and do not provide observations, rewards, or language feedback to the agent.
7. **Empirical Claims & Delimitation**:
   - The reported 42-stride escape path and 62-stride pearls-first path establish physical graph reachability under the 54-stride limit; they do not establish what policy is optimal for an observation-aliased learner with discounting and truncation bootstrapping.
   - Descriptive terms such as "marsh attractor" or "behavioral entrenchment" denote the observed statistical shift toward Chapel departures (93.8% of Wrecksmoor departures at loop 100); they do not prove that episodic memory blending causally induced this behavior in isolation, nor do they generalize to unstudied configurations.

---

## 4. Software Dependencies & Environment Isolation

The study executes in an isolated repository-local virtual environment (`.venv/`) using Python 3.9.6.
- **`matplotlib==3.9.2`**: Justified solely for rendering publication-quality diagnostic evidence figures into `figures/`. Plotting modules are strictly decoupled from core simulation and testing logic.

---

## 5. Route B Architecture & Matched Diagnostic Protocol (`tally_current_tide_pearl_v1`)

To address the pearl observability limitation without violating the POMDP world contract, Route B introduces an internal tally memory state within the agent while keeping world observations strictly external:

### Separation of External Observation and Policy State
- **External World Observation**: $o_t = ((row, col), \text{frozenset}(items))$, where $items$ contains only `ITEM_LAMP_KEY = 1` if held. Pearls (`ITEM_PEARLS = 0`) remain completely hidden from world observation.
- **Agent Policy State**: $s_t = (pos, inventory, marker)$, where $pos = (row, col)$, $inventory = \text{frozenset}(items)$, and $marker \in \{True, False\}$ is an internal boolean tally indicating whether pearls have been collected during the current tide.

### Marker Semantics & Dawn Resets
- The tally marker initializes to `False` at dawn ($t=0$).
- When the agent executes a physical transition collecting pearls ($r = +0.99$ net), the resulting next policy state updates $marker \leftarrow True$.
- The marker remains $True$ for subsequent strides within the same tide, allowing Q-values and episodic memory keys to differentiate states before versus after pearl pickup.
- At dawn truncation ($t=54$), the marker resets to `False`. No Q-learning update or memory lookup crosses the dawn reset boundary.

### World Model Decoupling
The tabular `WorldModel` operates strictly as an observer of external environmental transitions:
- Model transition and empirical reward counts are keyed purely by external observation and action: $(o_t, a_t, o_{t+1})$.
- The world model does not observe, store, or branch on the internal policy tally marker.

### Schema 2.0.0 & Diagnostic Telemetry
- Route B records use schema `2.0.0` with condition identifier `tally_current_tide_pearl_v1`.
- Per-stride records explicitly distinguish `obs`, `policy_state`, `prev_obs`, `prev_policy_state`, `next_obs`, and `next_policy_state`.
- Evaluation protocols execute non-learning probes (pre-training control and post-training test) with bit-for-bit RNG state capture and restoration across both action and memory streams.

### Diagnostic Scope & Limitations
- The matched diagnostic evaluates seeds 301–303 over 100 tides ($K_{\max}=100$) against a matched baseline arm (`baseline_hidden_pearl_v1`).
- This is an exploratory operational screen, not a confirmatory powered study. Findings characterize policy behavior under this specific 3-seed configuration and do not establish general theoretical bounds across unstudied hyperparameter regimes.
