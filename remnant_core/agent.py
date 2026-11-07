"""Tabular Q-learning agent with non-parametric episodic memory blending.

Computes standard Bellman updates augmented with an explicit terminal
transition target flag.
"""

import random
from typing import Dict, List, Optional, Tuple, FrozenSet

from remnant_core.config import (
    DEFAULT_LR,
    DEFAULT_GAMMA,
    DEFAULT_EPS,
    DEFAULT_MEMORY_BLEND,
)
from remnant_core.memory import EpisodicMemory
from remnant_core.exploration_memory import CrossTideExplorationMemory

State = Tuple[Tuple[int, int], FrozenSet[int]]


class QAgent:
    """Tabular Q-learning agent with episodic memory blending."""

    def __init__(
        self,
        n_actions: int = 4,
        lr: float = DEFAULT_LR,
        gamma: float = DEFAULT_GAMMA,
        eps: float = DEFAULT_EPS,
        memory: Optional[EpisodicMemory] = None,
        memory_blend: float = DEFAULT_MEMORY_BLEND,
        exploration_memory: Optional[CrossTideExplorationMemory] = None,
        exploration_coefficient: float = 1.0,
        node_novelty=None,
        trace_decay=None,
        raw_q_bootstrap: bool = False,
    ):
        self.q: Dict[Tuple[State, int], float] = {}
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.memory = memory
        self.memory_blend = memory_blend
        self.exploration_memory = exploration_memory
        self.exploration_coefficient = exploration_coefficient
        self.node_novelty = node_novelty
        self.trace_decay = trace_decay
        self.raw_q_bootstrap = raw_q_bootstrap
        self.traces: Dict[Tuple[State, int], float] = {}

    def reset_traces(self):
        self.traces = {}

    def qval(self, s: State, a: int) -> float:
        """Compute blended action value: V(s, a) = (1 - beta) * Q(s, a) + beta * M(s, a)."""
        qr = self.q.get((s, a), 0.0)
        if self.memory is not None:
            mv = self.memory.value(s, a)
            if mv is not None:
                return (1.0 - self.memory_blend) * qr + self.memory_blend * mv
        return qr

    def act(self, s: State, rng: random.Random, training: bool = True) -> int:
        """Select action via epsilon-greedy policy over blended action values (with exploration bonus in training)."""
        if rng.random() < self.eps:
            return rng.randrange(self.n_actions)

        blended_vals = []
        for a in range(self.n_actions):
            v = self.qval(s, a)
            if training and self.exploration_memory is not None:
                v += self.exploration_memory.exploration_score(s, a, self.exploration_coefficient)
            if training and self.node_novelty is not None:
                v += self.node_novelty.bonus(s, a)
            blended_vals.append(v)

        best = max(blended_vals)
        cands = [a for a, val in enumerate(blended_vals) if val == best]
        return rng.choice(cands)

    def learn(self, s: State, a: int, r: float, s_next: State, terminal: bool = False) -> None:
        """Update tabular Q-value via Bellman equation.
        
        If terminal=True (escape), target is r alone (terminal state value is 0).
        If terminal=False, target is r + gamma * max_a' V(s_next, a').
        """
        old = self.q.get((s, a), 0.0)
        if terminal:
            target = float(r)
        elif self.raw_q_bootstrap:
            # Corrected arms only: bootstrap against raw Q. Blending episodic memory
            # into the learning target makes targets stochastic and halves
            # propagated value. Default keeps the original blended bootstrap.
            next_max = max(self.q.get((s_next, an), 0.0) for an in range(self.n_actions))
            target = float(r) + self.gamma * next_max
        else:
            next_max = max(self.qval(s_next, an) for an in range(self.n_actions))
            target = float(r) + self.gamma * next_max

        if self.trace_decay is None:
            new_q = old + self.lr * (target - old)
            self.q[(s, a)] = new_q
            return
        # Q(lambda) with replacing traces
        for k in list(self.traces):
            self.traces[k] *= self.gamma * self.trace_decay
        self.traces[(s, a)] = 1.0
        delta = target - old
        for k, e in self.traces.items():
            self.q[k] = self.q.get(k, 0.0) + self.lr * delta * e

    def to_dict(self):
        """Serialize agent parameters and Q-table to plain JSON-serializable dict."""
        serialized_q = []
        for (s, a), val in self.q.items():
            if len(s) == 3:
                pos, inv, marker = s
                serialized_q.append({
                    "pos": list(pos),
                    "inventory": sorted(list(inv)),
                    "tally_pearl_picked": bool(marker),
                    "action": a,
                    "val": val,
                })
            else:
                pos, inv = s
                serialized_q.append({
                    "pos": list(pos),
                    "inventory": sorted(list(inv)),
                    "action": a,
                    "val": val,
                })
        data = {
            "n_actions": self.n_actions,
            "lr": self.lr,
            "gamma": self.gamma,
            "eps": self.eps,
            "memory_blend": self.memory_blend,
            "q": serialized_q,
        }
        if self.memory is not None:
            data["memory"] = self.memory.to_dict()
        if self.exploration_memory is not None:
            data["exploration_memory"] = self.exploration_memory.to_dict()
            data["exploration_coefficient"] = self.exploration_coefficient
        return data

    @classmethod
    def from_dict(cls, data):
        """Reconstruct QAgent from serialized dict."""
        mem = None
        if "memory" in data and data["memory"] is not None:
            mem = EpisodicMemory.from_dict(data["memory"])

        exp_mem = None
        if "exploration_memory" in data and data["exploration_memory"] is not None:
            exp_mem = CrossTideExplorationMemory.from_dict(data["exploration_memory"])

        agent = cls(
            n_actions=data["n_actions"],
            lr=data["lr"],
            gamma=data["gamma"],
            eps=data["eps"],
            memory=mem,
            memory_blend=data["memory_blend"],
            exploration_memory=exp_mem,
            exploration_coefficient=data.get("exploration_coefficient", 1.0),
        )
        for item in data.get("q", []):
            if "tally_pearl_picked" in item:
                s = (tuple(item["pos"]), frozenset(item["inventory"]), bool(item["tally_pearl_picked"]))
            else:
                s = (tuple(item["pos"]), frozenset(item["inventory"]))
            agent.q[(s, item["action"])] = float(item["val"])
        return agent
