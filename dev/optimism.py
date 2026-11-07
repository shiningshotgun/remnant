"""Map-level optimism under uncertainty for Wick's stride-level Q-learner.

An action Wick has never actually taken at a place (position + held items), as
recorded in its transition WorldModel, is valued at a fixed optimistic constant
``v_opt`` instead of its (uninformed) Q-value. This is used in two places only:

1. action selection during training (untried actions look promising), and
2. the Q-learning bootstrap target (a real transition that lands next to an
   untried action carries that optimism backward through real experience).

Nothing is simulated: optimism is read off the set of (place, action) pairs that
real strides have already covered. Once an action has been taken anywhere with
the same (position, items), it is "known" and loses its optimism for every
policy state at that place, including the tally-marker variant. Keying on the
map rather than the full policy state prevents permanent optimism on
(place, marker) combinations that the tide clock makes unreachable.

At frozen evaluation (``training=False``) optimism is not used for action
selection; the policy is the learned Q (blended with the Tally) alone.
"""
from typing import Any


class MapOptimism:
    def __init__(self, model, v_opt: float, mask_noops: bool = False):
        self.model = model
        self.v_opt = float(v_opt)
        self.mask_noops = mask_noops

    def known_noop(self, policy_state: Any, action: int) -> bool:
        """True when every real experience of this (place, action) left Wick at the
        same place (a wall, a locked gate, a missing arc). Learned, not given."""
        if not self.mask_noops:
            return False
        key = ((policy_state[0], policy_state[1]), action)
        if key not in self.model.counts:
            return False
        here = (policy_state[0], policy_state[1])
        return all(sn == here for sn in self.model.counts[key])

    def allowed(self, policy_state: Any, n_actions: int):
        acts = [a for a in range(n_actions) if not self.known_noop(policy_state, a)]
        return acts if acts else list(range(n_actions))

    def untried(self, policy_state: Any, action: int) -> bool:
        key = ((policy_state[0], policy_state[1]), action)
        # WorldModel.counts is a defaultdict: membership tests do not insert.
        return key not in self.model.counts
