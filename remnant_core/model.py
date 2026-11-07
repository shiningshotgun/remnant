"""Tabular learned world model: transition counts and empirical rewards.

Ingests only real environmental transitions.
"""

from collections import defaultdict
import random
from typing import Dict, Optional, Tuple, FrozenSet

State = Tuple[Tuple[int, int], FrozenSet[int]]


class WorldModel:
    """Tabular transition and reward model estimated from experience."""

    def __init__(self):
        # (s, a) -> s' -> count
        self.counts: Dict[Tuple[State, int], Dict[State, int]] = defaultdict(lambda: defaultdict(int))
        # (s, a) -> sum of observed scalar rewards
        self.reward_sum: Dict[Tuple[State, int], float] = defaultdict(float)

    def observe(self, s: State, a: int, r: float, s_next: State) -> None:
        """Ingest one verified real environmental transition."""
        self.counts[(s, a)][s_next] += 1
        self.reward_sum[(s, a)] += float(r)

    def sample_next(self, s: State, a: int, rng: random.Random) -> Tuple[State, float]:
        """Sample next state from empirical transition distribution."""
        succ = self.counts.get((s, a))
        if not succ:
            return s, 0.0

        total = sum(succ.values())
        mean_reward = self.reward_sum[(s, a)] / total
        x = rng.uniform(0, total)
        for sn, cnt in succ.items():
            x -= cnt
            if x <= 0:
                return sn, mean_reward

        # Numerical fallback
        sn = next(iter(succ))
        return sn, mean_reward

    def predict(self, s: State, a: int) -> Tuple[Optional[State], Optional[float]]:
        """Return the most frequent next state and empirical mean reward (out-of-sample prediction)."""
        succ = self.counts.get((s, a))
        if not succ:
            return None, None

        total = sum(succ.values())
        sn_best = max(succ, key=succ.get)
        mean_reward = self.reward_sum[(s, a)] / total
        return sn_best, mean_reward

    def to_dict(self):
        """Serialize world model to plain JSON-serializable structure."""
        serialized_counts = []
        for (s, a), succ in self.counts.items():
            pos, inv = s
            succ_list = [
                {"next_pos": list(sn[0]), "next_inv": sorted(list(sn[1])), "count": cnt}
                for sn, cnt in succ.items()
            ]
            serialized_counts.append({
                "pos": list(pos),
                "inventory": sorted(list(inv)),
                "action": a,
                "reward_sum": self.reward_sum[(s, a)],
                "successors": succ_list,
            })
        return {"transitions": serialized_counts}

    @classmethod
    def from_dict(cls, data):
        """Reconstruct WorldModel from serialized dict."""
        model = cls()
        for item in data.get("transitions", []):
            s: State = (tuple(item["pos"]), frozenset(item["inventory"]))
            a = item["action"]
            model.reward_sum[(s, a)] = float(item["reward_sum"])
            for sn_data in item.get("successors", []):
                sn: State = (tuple(sn_data["next_pos"]), frozenset(sn_data["next_inv"]))
                model.counts[(s, a)][sn] = int(sn_data["count"])
        return model
