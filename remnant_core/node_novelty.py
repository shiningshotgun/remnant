"""Node novelty: first-visit novelty over physical nodes, from the agent's own Tally.

Tracks which physical positions have ever been visited (lifetime, persists across tides).
Training-time action-selection bonus for actions whose expected next node is unvisited
or whose consequence has never been experienced. Uses the WorldModel only as a
one-step lookup of experienced transitions (recall, not simulation).
"""
from typing import Any, Optional, Set, Tuple


class NodeNovelty:
    def __init__(self, model, coefficient: float = 1.0):
        self.visited: Set[Tuple[int, int]] = set()
        self.model = model
        self.coefficient = coefficient

    def mark(self, pos: Tuple[int, int]) -> None:
        self.visited.add(tuple(pos))

    def bonus(self, policy_state: Any, action: int) -> float:
        obs = (policy_state[0], policy_state[1])
        pred_next, _ = self.model.predict(obs, action)
        if pred_next is None:
            return float(self.coefficient)
        if tuple(pred_next[0]) not in self.visited:
            return float(self.coefficient)
        return 0.0
