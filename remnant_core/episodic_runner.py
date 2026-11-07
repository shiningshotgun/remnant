"""Runner for the return-Tally study (experiments/run_return_tally_v1.py).

Self-contained so every legacy LoopRunner condition stays byte-identical.

Each tide is one of two kinds, drawn by a fixed seeded coin (independent of anything
Wick learns, identical sequence across arms for a given seed):

- curiosity tide: Wick walks toward the nearest frontier of its own map (an action it
  has never taken at a place it has reached), BFS over transitions it has actually
  experienced. A place where a tide ended in escape is not a frontier (nothing can be
  done there); this removes the legacy frontier planner's permanent pull to Tidegate.
  When no frontier is reachable on the map, the stride falls back to the policy.
- policy tide: Wick's stride-level policy acts, epsilon-greedy (eps 0.2).

Every lived stride, of either kind, is real experience for every store.

Arms
----
ec_curious  policy = Return Tally (best lived discounted return-to-go). Curiosity coin p.
q_curious   policy = legacy corrected learner (Q(lambda) 0.95, raw-Q bootstrap, truncation
            bootstraps to dawn, exact-match immediate-reward Tally blended at 0.5), same
            coin. Matched negative control: same experience generator, legacy learner.
ec_amnesic  ec_curious, but the Return Tally and the map are wiped at every dawn
            (persistence ablated). Negative control.
ec_only     ec_curious with the coin fixed to "policy" (no curiosity tides). Tests
            whether the Return Tally alone discovers the escape.

Frozen evaluation (primary): a fresh dawn, the policy alone acts every stride, eps 0,
no curiosity, no updates to any store. Tie-breaks use a separate seeded RNG.
"""
import collections
import random
from typing import Any, Dict, List, Optional

from remnant_core.agent import QAgent
from remnant_core.memory import EpisodicMemory
from remnant_core.model import WorldModel
from remnant_core.return_tally import ReturnTally
from remnant_core.worlds.reach import ReachWorld

ARMS = ("ec_curious", "q_curious", "ec_amnesic", "ec_only")


class EpisodicRunner:
    def __init__(self, seed: int, arm: str, curiosity_p: float = 0.5, eps: float = 0.2,
                 gamma: float = 0.95, world_factory=ReachWorld):
        if arm not in ARMS:
            raise ValueError(arm)
        self.seed = seed
        self.arm = arm
        self.curiosity_p = 0.0 if arm == "ec_only" else float(curiosity_p)
        self.eps = eps
        self.gamma = gamma
        self.world_factory = world_factory
        self.world = world_factory()
        self.action_rng = random.Random(seed * 1000 + 1)
        self.coin_rng = random.Random(seed * 1000 + 11)
        self.tide_log: List[Dict[str, Any]] = []
        self._build_stores()

    # ------------------------------------------------------------------ stores
    def _build_stores(self):
        self.model = WorldModel()
        self.terminal_places = set()
        self.tally = ReturnTally(gamma=self.gamma)
        if self.arm == "q_curious":
            self.memory = EpisodicMemory(seed=self.seed * 1000 + 6, exact_match=True)
            self.agent = QAgent(memory=self.memory, eps=self.eps, trace_decay=0.95,
                                raw_q_bootstrap=True)
        else:
            self.memory = None
            self.agent = None

    def map_edges(self) -> int:
        return len(self.model.counts)

    # ---------------------------------------------------------------- policy
    def policy_action(self, ps, rng, frozen: bool) -> int:
        eps = 0.0 if frozen else self.eps
        if self.arm == "q_curious":
            old = self.agent.eps
            self.agent.eps = eps
            try:
                return self.agent.act(ps, rng, training=not frozen)
            finally:
                self.agent.eps = old
        return self.tally.act(ps, rng, eps)

    # ------------------------------------------------------------- curiosity
    def frontier_action(self, place) -> Optional[int]:
        """First action toward the nearest frontier on Wick's experienced map, or None."""
        tried = self.model.counts

        def is_frontier(p):
            return p not in self.terminal_places and any((p, a) not in tried for a in range(4))

        if is_frontier(place):
            return min(a for a in range(4) if (place, a) not in tried)
        prev = {place: None}
        q = collections.deque([place])
        target = None
        while q:
            cur = q.popleft()
            if cur != place and is_frontier(cur):
                target = cur
                break
            for a in range(4):
                if (cur, a) not in tried:
                    continue
                nxt, _r = self.model.predict(cur, a)
                if nxt is None or nxt == cur or nxt in prev:
                    continue
                prev[nxt] = (cur, a)
                q.append(nxt)
        if target is None:
            return None
        node = target
        while prev[node][0] != place:
            node = prev[node][0]
        return prev[node][1]

    # -------------------------------------------------------------- one tide
    def train_tide(self) -> Dict[str, Any]:
        if self.arm == "ec_amnesic":
            self._build_stores()
        kind = "curiosity" if self.coin_rng.random() < self.curiosity_p else "policy"
        obs = self.world.reset()
        marker = False
        ps = (obs[0], obs[1], marker)
        dawn_ps = ps
        if self.agent is not None:
            self.agent.reset_traces()
        lived = []
        prov = collections.Counter()
        ret, escaped, pearls, key = 0.0, False, False, False
        while True:
            a = None
            if kind == "curiosity":
                a = self.frontier_action(obs)
                if a is not None:
                    prov["curiosity"] += 1
            if a is None:
                a = self.policy_action(ps, self.action_rng, frozen=False)
                prov["policy"] += 1
            obs2, r, terminal, truncated = self.world.step(a)
            tel = self.world.last_telemetry or {}
            if tel.get("pearls_picked"):
                pearls = True
                marker = True
            if tel.get("key_picked"):
                key = True
            ps2 = (obs2[0], obs2[1], marker)
            self.model.observe(obs, a, r, obs2)
            if terminal:
                self.terminal_places.add(obs2)
            if self.agent is not None:
                learn_next = dawn_ps if truncated else ps2
                self.agent.learn(ps, a, r, learn_next, terminal=terminal)
                self.memory.append(ps, a, r)
            lived.append((ps, a, r))
            ret += r
            obs, ps = obs2, ps2
            if terminal:
                escaped = True
                break
            if truncated:
                break
        self.tally.ingest_tide(lived)
        rec = dict(tide=len(self.tide_log) + 1, kind=kind, escaped=escaped,
                   strides=len(lived), ret=round(ret, 4), pearls=pearls, key=key,
                   map_edges=self.map_edges(), prov=dict(prov))
        self.tide_log.append(rec)
        return rec

    def run(self, tides: int) -> List[Dict[str, Any]]:
        for _ in range(tides):
            self.train_tide()
        return self.tide_log

    # ------------------------------------------------------- frozen evaluation
    def snapshot_sizes(self):
        return (len(self.tally.best), sum(self.tally.visits.values()), self.map_edges(),
                0 if self.agent is None else len(self.agent.q),
                0 if self.memory is None else len(self.memory.entries))

    def evaluate(self, tides: int, seed: int) -> Dict[str, Any]:
        """Frozen deployed policy from a fresh dawn: policy only, eps 0, no updates.
        The amnesic arm meets each dawn with wiped stores, per its own rule."""
        if self.arm == "ec_amnesic":
            self._build_stores()
        before = self.snapshot_sizes()
        out = []
        for k in range(tides):
            rng = random.Random(seed * 100 + k)
            w = self.world_factory()
            obs = w.reset()
            marker = False
            ps = (obs[0], obs[1], marker)
            path = [list(obs[0])]
            ret, escaped, pearls = 0.0, False, False
            while True:
                a = self.policy_action(ps, rng, frozen=True)
                obs, r, terminal, truncated = w.step(a)
                tel = w.last_telemetry or {}
                if tel.get("pearls_picked"):
                    pearls = True
                    marker = True
                ps = (obs[0], obs[1], marker)
                path.append(list(obs[0]))
                ret += r
                if terminal:
                    escaped = True
                    break
                if truncated:
                    break
            out.append(dict(escaped=escaped, strides=len(path) - 1, ret=round(ret, 4),
                            pearls=pearls, path=path))
        assert self.snapshot_sizes() == before, "frozen evaluation modified a store"
        return dict(escapes=sum(o["escaped"] for o in out), tides=out)
