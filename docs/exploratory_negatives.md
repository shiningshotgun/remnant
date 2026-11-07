# Exploratory negatives (development only)

These were run on development seeds 9001–9003 before the pre-registration. They are not
confirmatory results. They are kept because they shaped the design. The code is in
[`dev/`](../dev/README.md).

## Model-free optimism for the stride-level Q-learner (abandoned)

Idea: value any action Wick had never taken at a place at a fixed optimistic constant
`v_opt`, in both action choice and the Q-learning bootstrap, so the learner seeks the
unknown.

- Grid tried: `v_opt` ∈ {1, 3, 10, 30} × eligibility traces {0.95, none} × truncation
  {bootstrap to dawn, zero} × learned no-op mask {off, on}; 3 dev seeds, 100 tides each,
  3 frozen evaluation tides.
- Outcome: 0 frozen-policy escapes in every setting. One setting (v_opt 10, traces 0.95,
  dawn bootstrap, mask on) lived 3 escapes in training on one seed, from tide 96; its frozen
  policy still did not escape. Maps stalled at 130–383 of 472 place–action pairs.
- Diagnosis (not isolated experimentally): optimism propagates back roughly one state per
  visit, too slowly for a 42-stride route in 100 tides; 2 of 4 actions on chain nodes are
  no-ops and Wick dithered on them (about 35% of late strides in one inspected run); naive
  Q(λ) on cyclic paths produced values above `v_opt`.

## Legacy learner given lived escapes (diagnostic, from the earlier selector study)

In the earlier random-mode control (`results/metabucket_td_v1.json`), the legacy low-level
Q-learner lived 31–42 escapes per seed, yet its frozen low-level policy escaped 0/2 per
seed. Inspection of one seed showed inflated dawn values favouring the pearl route. This
observation motivated the Return Tally. It is a diagnosis, not an isolated cause.

## Return Tally dev check

One pass per arm on dev seeds 9001–9003 with the final design: the primary arm escaped
5/5 frozen tides on all three seeds; the controls did not. No parameter was changed after
this pass.
