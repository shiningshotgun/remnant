"""World contract package for Remnant.

World Contract:
A world implementing the contract must provide:
- reset() -> observation
    Resets the environment physically to its initial starting state, returning the initial observation.
- state() -> observation
    Returns the current observation of the agent without mutating environment state.
- step(action) -> (observation, reward, terminal, truncated)
    Executes one action step in the environment.
    Returns:
      observation: The agent's observation of the resulting state.
      reward (float): The scalar reward observed from the transition.
      terminal (bool): True only on successful escape; False otherwise.
      truncated (bool): True when the loop's step limit is reached without escape; False otherwise.
    Semantics:
      - terminal is True only on escape.
      - truncated is True when the loop's step limit is reached without escape.
      - An invalid action leaves the agent in place and still costs one step's reward.
- max_steps: int
    Attribute indicating the maximum allowed physical steps per loop before truncation.
"""

from remnant_core.worlds.grid import LoopWorld

__all__ = ["LoopWorld"]
