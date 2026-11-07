"""Cross-tide exploration memory for Remnant.

Maintains visit counts across tides keyed by policy state and action.
Counts persist across tides (do not reset at dawn).
Generates an exploration bonus score during training action selection:
    exploration_score(s, a) = exploration_coefficient / sqrt(1 + visit_count(s, a))
"""

import math
from typing import Dict, Tuple, Any, List


class CrossTideExplorationMemory:
    """Tracks visit counts across environment resets for policy state and action pairs."""

    def __init__(self):
        # Key: (policy_state, action) -> count
        self.counts: Dict[Tuple[Any, int], int] = {}

    def visit_count(self, policy_state: Any, action: int) -> int:
        """Return the number of times (policy_state, action) was executed in real training."""
        return self.counts.get((policy_state, action), 0)

    def increment(self, policy_state: Any, action: int) -> None:
        """Increment the visit count by 1 for (policy_state, action)."""
        key = (policy_state, action)
        self.counts[key] = self.counts.get(key, 0) + 1

    def exploration_score(self, policy_state: Any, action: int, exploration_coefficient: float = 1.0) -> float:
        """Compute generic exploration score: exploration_coefficient / sqrt(1 + visit_count)."""
        n = self.visit_count(policy_state, action)
        return float(exploration_coefficient) / math.sqrt(1.0 + n)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize exploration memory to plain JSON-serializable dictionary."""
        entries = []
        for (s, a), count in self.counts.items():
            if len(s) == 3:
                pos, inv, marker = s
                entries.append({
                    "pos": list(pos),
                    "inventory": sorted(list(inv)),
                    "tally_pearl_picked": bool(marker),
                    "action": a,
                    "count": count,
                })
            else:
                pos, inv = s
                entries.append({
                    "pos": list(pos),
                    "inventory": sorted(list(inv)),
                    "action": a,
                    "count": count,
                })
        return {
            "total_entries": len(entries),
            "entries": entries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrossTideExplorationMemory":
        """Reconstruct CrossTideExplorationMemory from serialized dict."""
        mem = cls()
        for item in data.get("entries", []):
            if "tally_pearl_picked" in item:
                s = (tuple(item["pos"]), frozenset(item["inventory"]), bool(item["tally_pearl_picked"]))
            else:
                s = (tuple(item["pos"]), frozenset(item["inventory"]))
            mem.counts[(s, item["action"])] = int(item["count"])
        return mem
