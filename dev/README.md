# dev/ — exploratory development code

Not part of the accepted study. Run only on development seeds 9001–9003. Outcomes are
summarised in [docs/exploratory_negatives.md](../docs/exploratory_negatives.md).

- `optimistic_runner.py`, `optimism.py`, `agent_with_optimism_hooks.py`, `dev_sweep.py`:
  the abandoned model-free optimism design. `agent_with_optimism_hooks.py` is a copy of
  `remnant_core/agent.py` with optimism hooks, so the shipped agent is unchanged.
- `dev_ec.py`: the single development check of the Return Tally design.

Example: `python3 dev/dev_sweep.py optimistic 10 9001 0.95 dawn mask`
(arguments: arm, v_opt, seeds, trace decay or `none`, truncation `dawn`/`zero`, `mask`).
