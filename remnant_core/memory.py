"""Episodic memory: non-parametric similarity-weighted value lookup.

Non-parametric kernel value lookup over stored episodic transitions.
Outcome stores immediate scalar transition reward r.
"""

import math
import random
from typing import List, Optional, Tuple, FrozenSet

State = Tuple[Tuple[int, int], FrozenSet[int]]
MemoryEntry = Tuple[State, int, float]


class EpisodicMemory:
    """Buffer of episodic transitions providing similarity-weighted value lookup."""

    def __init__(self, temperature: float = 0.1, lookup_sample: int = 400, seed: int = 0,
                 exact_match: bool = False):
        # exact_match=False keeps the original similarity-weighted contract; the
        # corrected_* experiment arms opt into exact-match because node-local arc
        # indexing makes cross-position action blending a semantic type error, and
        # subsampled lookup made read values history-dependent.
        self.entries: List[MemoryEntry] = []  # (state, action, outcome)
        self.temperature = temperature
        self.lookup_sample = lookup_sample
        self._rng = random.Random(seed)
        self.exact_match = exact_match

    def append(self, state: State, action: int, outcome: float) -> None:
        """Append an experience tuple (s, a, r)."""
        self.entries.append((state, action, float(outcome)))

    def value(self, state: State, action: int) -> Optional[float]:
        """Value lookup over stored entries. Default: similarity-weighted (original
        contract). With exact_match: mean outcome for this exact (state, action)."""
        if not self.entries:
            return None

        if self.exact_match:
            num, den = 0.0, 0.0
            for s, a, out in self.entries:
                if a == action and s == state:
                    num += out
                    den += 1.0
            return (num / den) if den > 0 else None

        if len(self.entries) > self.lookup_sample:
            pool = self._rng.sample(self.entries, self.lookup_sample)
        else:
            pool = self.entries

        num, den = 0.0, 0.0
        for s, a, out in pool:
            if a == action:
                sim = self._sim(state, s)
                if sim > 0:
                    w = math.exp(sim / self.temperature)
                    num += w * out
                    den += w

        return (num / den) if den > 0 else None

    @staticmethod
    def _sim(s1: State, s2: State) -> float:
        """Manhattan distance on grid position plus symmetric difference on held inventory.
        
        Under Route B tally memory, marker mismatch adds unit distance so differing
        belief states yield lower kernel similarity.
        """
        if len(s1) == 3 and len(s2) == 3:
            (p1, c1, m1), (p2, c2, m2) = s1, s2
            marker_dist = 1 if m1 != m2 else 0
        elif len(s1) == 3:
            (p1, c1, m1), (p2, c2) = s1, s2
            marker_dist = 1 if m1 else 0
        elif len(s2) == 3:
            (p1, c1), (p2, c2, m2) = s1, s2
            marker_dist = 1 if m2 else 0
        else:
            (p1, c1), (p2, c2) = s1, s2
            marker_dist = 0
        dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + len(c1 ^ c2) + marker_dist
        return 1.0 / (1.0 + dist)

    def get_rng_state(self):
        """Serialize internal RNG state."""
        raw = self._rng.getstate()
        return (raw[0], list(raw[1]), raw[2])

    def set_rng_state(self, state_tuple):
        """Restore internal RNG state."""
        st = (state_tuple[0], tuple(state_tuple[1]), state_tuple[2])
        self._rng.setstate(st)

    def to_dict(self):
        """Serialize memory to plain JSON-serializable dict."""
        serialized_entries = []
        for s, a, out in self.entries:
            if len(s) == 3:
                pos, inv, marker = s
                serialized_entries.append({
                    "pos": list(pos),
                    "inventory": sorted(list(inv)),
                    "tally_pearl_picked": bool(marker),
                    "action": a,
                    "outcome": out,
                })
            else:
                pos, inv = s
                serialized_entries.append({
                    "pos": list(pos),
                    "inventory": sorted(list(inv)),
                    "action": a,
                    "outcome": out,
                })
        return {
            "temperature": self.temperature,
            "lookup_sample": self.lookup_sample,
            "rng_state": self.get_rng_state(),
            "entries": serialized_entries,
        }

    @classmethod
    def from_dict(cls, data):
        """Reconstruct EpisodicMemory from serialized dict."""
        mem = cls(
            temperature=data.get("temperature", 0.1),
            lookup_sample=data.get("lookup_sample", 400),
        )
        for e in data.get("entries", []):
            if "tally_pearl_picked" in e:
                s = (tuple(e["pos"]), frozenset(e["inventory"]), bool(e["tally_pearl_picked"]))
            else:
                s = (tuple(e["pos"]), frozenset(e["inventory"]))
            mem.append(s, e["action"], e["outcome"])
        if data.get("rng_state") is not None:
            mem.set_rng_state(data["rng_state"])
        return mem
