> **Historical draft protocol, never executed** (it was already marked invalid as written). Current study: [pre-registration](../preregistration_return_tally_v1.md), [results](../results.md).

# Protocol: Controlled Memory Ablation Study on the Reach of Vesper

> [!WARNING]
> **HELD / INVALID AS WRITTEN**
> This prospective protocol is currently held and cannot be executed as drafted due to four structural deficiencies:
> 1. **Parameter blend gap**: The $\beta = 0.0$ pure Q-learning ablation is not implemented in the CLI parameter blend interface.
> 2. **Unverified power claim**: The $\ge 90\%$ statistical power assertion across $n=30$ runs per arm is unverified against empirical multi-loop variance.
> 3. **Unproven resource scaling**: RAM and disk scaling for 60 unsegmented runs is unproven on host infrastructure.
> 4. **Misaligned objective**: The protocol measures behavioral pearl preference ($\Delta_{\text{pearl}}$), not whether Wick learns to escape Vesper.

**Document Version**: 1.0.0 (Held Protocol)  
**Status**: HELD / INVALID AS WRITTEN  
**Author**: Remnant Project Research Team  

---

## 1. Primary Research Questions & Estimands

The completed baseline study on the Reach of Vesper demonstrated that an agent blending Q-learning with non-parametric episodic memory at weight $\beta = 0.5$ exhibited 0/30 escapes within 100 tides, with 28/30 seeds taking pearls at tide 100 and 93.8% of Wrecksmoor departures committing to the northern marsh route. 

However, because the baseline study evaluated only a single parameter configuration, it could not determine whether the episodic memory blend causally drove the concentration on the pearl attractor.

### Primary Estimands
Because escape is a rare, absorbing event under horizon $K_{\max}=100$, primary comparisons evaluate behavioral and policy outcomes that exhibit continuous variance, rather than escape alone:
1. **Pearl Collection Rate Difference at Horizon**:
   $$\Delta_{\text{pearl}} = \mu_{\text{pearl}}(\text{Control: } \beta=0.5) - \mu_{\text{pearl}}(\text{Ablation: } \beta=0.0)$$
   evaluated at loop $K = 100$.
2. **Marsh Fork Commitment Difference**:
   $$\Delta_{\text{marsh}} = \text{Share}_{\text{chapel}}(\text{Control: } \beta=0.5) - \text{Share}_{\text{chapel}}(\text{Ablation: } \beta=0.0)$$
   evaluated across successful departures from Wrecksmoor Landing at loop $K = 100$.
3. **Cumulative Escape Difference**:
   Difference in empirical escape incidence $\hat{p}_{\text{esc}}(\text{Control}) - \hat{p}_{\text{esc}}(\text{Ablation})$ with exact binomial confidence intervals. (Note: equal zero rates in both arms is inconclusive regarding equivalence).

---

## 2. Experimental Arms & Controlled Treatments

To rigorously isolate the functional contribution of episodic memory without introducing confounding model estimators:
- **Control Arm (`memory_blend_0.5`)**:
  - $Q$-learning ($\alpha = 0.3, \gamma = 0.95, \epsilon = 0.2$).
  - Episodic memory blend $\beta = 0.5$, softmax temperature $\tau = 0.1$, lookup sample cap $400$.
  - Truncation bootstrapping on $\max_{a'} V(s_{54}, a')$.
- **Ablation Arm (`memory_blend_0.0` - Pure Q-learning)**:
  - Identical $Q$-learning parameters ($\alpha = 0.3, \gamma = 0.95, \epsilon = 0.2$).
  - Memory blend $\beta = 0.0$ (disables episodic value lookup during action selection and truncation bootstrapping: $V(s, a) \equiv Q(s, a)$).
  - Memory buffer logging maintained identically for recording and accounting parity.
  - Truncation bootstrapping on $\max_{a'} Q(s_{54}, a')$.

All other environment rules, transition costs ($r=-0.01$), rewards, 72-node graph geometry, and 54-stride loop limit are held strictly invariant.

---

## 3. RNG Stream Handling & Experimental Independence

In reinforcement learning with stochastic $\epsilon$-greedy action selection, seeding both arms with the same integer does **not** generate matched random action consumption once trajectories diverge after the first exploratory branch. 
- Evaluating seeds as paired observations under the assumption of variance reduction is therefore invalid once divergence occurs.
- Each seed-arm trajectory constitutes an independent experimental unit. Analysis will employ independent two-sample comparison methods (e.g., Welch's $t$-test / permutation testing for continuous shares, Fisher's exact test for binary endpoints).

---

## 4. Seed Allocation & Sample Size Determination

To maintain strict firewalling between exploratory feasibility work and confirmatory inference:
- **Baseline Study (Historical)**: Seeds 101–130 ($n=30$).
- **Resource Pilot (Exploratory Feasibility)**: Seeds 201–203 ($n=3$). Completed and stored in `pilot_results/`.
- **Confirmatory Study (Prospective)**: Disjoint seeds 301–360 ($n=30$ per arm, 60 runs total).
  - *Power Justification*: Under the baseline pearl collection rate of $\approx 93.3\%$ at loop 100, a two-arm sample size of $n=30$ per arm achieves $>90\%$ statistical power at two-sided $\alpha=0.05$ to detect a 30 percentage point reduction in pearl collection rate ($\delta = 0.30$) in the ablation arm.

---

## 5. Audit-Sufficient Storage & Failure Accounting

- Every run is published to a dedicated, new directory (`results_controlled_confirmatory/`).
- Full transition records (`records.json`), final learner states (`learners.json`), analytical outcomes (`outcomes.json`), and run manifests (`manifest.json`) are retained for all seeds.
- Zero seed dropping: If any seed encounters a runtime error or unexpected interruption, it is recorded with explicit status and error traces; seeds are never silently pruned to clean metrics.

---

## 6. Resource Feasibility & Operational Ceilings (Measured Pilot)

A serial exploratory resource pilot across 3 fresh seeds (201, 202, 203) at horizon $K=100$ was conducted to measure real operational demands:
- **Wall-clock Duration**: 22.57 seconds total ($\approx 7.52$ seconds per seed).
- **Peak Resident Memory**: 84.33 MB.
- **Disk Utilization**: 34.59 MB total for 3 seeds ($\approx 11.53$ MB per seed).
  - `records.json`: 27.79 MB
  - `learners.json`: 6.61 MB
  - `outcomes.json`: 0.19 MB
  - `manifest.json`: 0.003 MB
- **Store Growth per Seed**:
  - Episodic memory: exactly 5,400 transitions per seed ($54 \text{ strides} \times 100 \text{ loops}$).
  - $Q$-table entries: 225–233 state-action pairs (representing $\approx 78\%$ of the $72 \times 4 = 288$ theoretical state-action lattice).
  - Tabular model transitions: 225–233 entries.

### Scalability Projections for Confirmatory Study ($n=60$ runs)
- **Projected Wall-clock**: $\approx 7.5 \text{ s} \times 60 \approx 450 \text{ seconds}$ (7.5 minutes), well within operational limits.
- **Projected Disk Footprint**: $\approx 11.5 \text{ MB} \times 60 \approx 692 \text{ MB}$, well within the 40 GB available local filesystem capacity.
- Confirmatory execution is confirmed operationally feasible on this host.
