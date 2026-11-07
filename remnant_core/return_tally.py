"""Return Tally: Wick's episodic memory of what each choice actually led to.

The original Tally (memory.EpisodicMemory) stores the immediate scalar reward of each
lived stride and is blended into Q at beta=0.5, mixing reward scale with return scale
(see README, "Scale mismatch"). The Return Tally instead stores, for every lived
(policy_state, action), the best discounted return-to-go that Wick actually received
from that stride to the end of that same tide:

    G_t = r_t + gamma * G_{t+1},   G_{end of tide} = 0

where the tide ends at escape (terminal) or at the Drowning Toll (truncation). Nothing
crosses the dawn: the next tide starts from the same dawn whatever Wick did, so it is not
a consequence of this tide's choices. Every number in the table is a return Wick lived;
no bootstrapping, no model rollouts, no replay of counterfactuals. Vesper is
deterministic, so the best lived return from (s, a) is attainable again from (s, a).

This is model-free episodic control (Blundell et al., 2016) in tabular, exact-match form.
"""
import random
from typing import Any, Dict, List, Optional, Tuple

UNKNOWN_VALUE = 0.0  # value of an action Wick has never taken in this policy state


class ReturnTally:
    def __init__(self, gamma: float = 0.95):
        self.gamma = float(gamma)
        self.best: Dict[Tuple[Any, int], float] = {}
        self.visits: Dict[Tuple[Any, int], int] = {}

    def value(self, policy_state: Any, action: int) -> Optional[float]:
        return self.best.get((policy_state, action))

    def ingest_tide(self, lived: List[Tuple[Any, int, float]]) -> None:
        """lived: the tide's (policy_state, action, reward) strides in order."""
        g = 0.0
        returns = [0.0] * len(lived)
        for i in range(len(lived) - 1, -1, -1):
            g = float(lived[i][2]) + self.gamma * g
            returns[i] = g
        for (s, a, _r), g_i in zip(lived, returns):
            k = (s, a)
            self.visits[k] = self.visits.get(k, 0) + 1
            if k not in self.best or g_i > self.best[k]:
                self.best[k] = g_i

    def act(self, policy_state: Any, rng: random.Random, eps: float, n_actions: int = 4) -> int:
        """Epsilon-greedy over remembered best returns (unknown actions = UNKNOWN_VALUE)."""
        if rng.random() < eps:
            return rng.randrange(n_actions)
        vals = [self.best.get((policy_state, a), UNKNOWN_VALUE) for a in range(n_actions)]
        top = max(vals)
        return rng.choice([a for a in range(n_actions) if vals[a] == top])

    def __len__(self) -> int:
        return len(self.best)
