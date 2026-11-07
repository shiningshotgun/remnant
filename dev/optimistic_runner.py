"""Self-contained loop runner for the optimism study (optimistic_map_v1 and its controls).

Kept separate from runner.LoopRunner so that every legacy condition stays
byte-identical. Wick here acts only through its stride-level policy
(QAgent.act): there are no planners, no mode selector, no replay and no
simulated experience. Every Q-update, Tally entry and WorldModel count comes
from a stride Wick actually lived.

Arms
----
optimistic    Q(lambda) + Tally blend + tally marker, with MapOptimism (v_opt).
optimism_off  identical, MapOptimism removed (matched negative control).
amnesic       identical to optimistic, but Q-table, Tally, WorldModel and traces
              are wiped at every dawn (memory persistence ablated; negative control).
random        uniform random action every stride (floor control).

Semantics shared with the corrected_* arms of runner.LoopRunner: exact-match
Tally, raw-Q bootstrap, truncation bootstraps to the dawn policy state, Q(lambda)
replacing traces reset at dawn (trace_decay 0.95), tally marker in policy state.
"""
import random
from typing import Any, Dict, List, Optional

from dev.agent_with_optimism_hooks import QAgent
from remnant_core.memory import EpisodicMemory
from remnant_core.model import WorldModel
from dev.optimism import MapOptimism
from remnant_core.worlds.reach import ReachWorld

ARMS = ("optimistic", "optimism_off", "amnesic", "random")


class OptimisticRunner:
    def __init__(self, seed: int, arm: str, v_opt: float, eps: float = 0.2,
                 trace_decay=0.95, truncation="dawn", mask_noops=False,
                 world_factory=ReachWorld):
        if arm not in ARMS:
            raise ValueError(arm)
        self.seed = seed
        self.arm = arm
        self.v_opt = v_opt
        self.eps = eps
        self.trace_decay = trace_decay
        if truncation not in ("dawn", "zero"):
            raise ValueError(truncation)
        self.truncation = truncation
        self.mask_noops = mask_noops
        self.world_factory = world_factory
        self.world = world_factory()
        self.action_rng = random.Random(seed * 1000 + 1)
        self._build_stores()
        self.tide_log: List[Dict[str, Any]] = []

    # -- stores -----------------------------------------------------------
    def _build_stores(self):
        self.memory = EpisodicMemory(seed=self.seed * 1000 + 6, exact_match=True)
        self.model = WorldModel()
        if self.arm in ("optimistic", "amnesic"):
            opt = MapOptimism(self.model, self.v_opt, mask_noops=self.mask_noops)
        elif self.arm == "optimism_off" and self.mask_noops:
            # Matched control: same learned no-op mask, optimism value never applies.
            opt = MapOptimism(self.model, 0.0, mask_noops=True)
            opt.untried = lambda s, a: False
        else:
            opt = None
        self.agent = QAgent(memory=self.memory, eps=self.eps, trace_decay=self.trace_decay,
                            raw_q_bootstrap=True, optimism=opt)

    def map_edges(self) -> int:
        return len(self.model.counts)

    # -- one lived tide ---------------------------------------------------
    def train_tide(self) -> Dict[str, Any]:
        if self.arm == "amnesic":
            self._build_stores()
        obs = self.world.reset()
        marker = False
        ps = (obs[0], obs[1], marker)
        dawn_ps = ps
        self.agent.reset_traces()
        ret, strides, escaped, pearls, key = 0.0, 0, False, False, False
        while True:
            if self.arm == "random":
                a = self.action_rng.randrange(4)
            else:
                a = self.agent.act(ps, self.action_rng, training=True)
            obs2, r, terminal, truncated = self.world.step(a)
            tel = self.world.last_telemetry or {}
            if tel.get("pearls_picked"):
                pearls = True
                marker = True
            if tel.get("key_picked"):
                key = True
            ps2 = (obs2[0], obs2[1], marker)
            learn_next = dawn_ps if truncated else ps2
            # truncation="zero": the Drowning Toll ends this tide's return (value 0 beyond
            # it), since the dawn that follows is the same whatever Wick did.
            trunc_terminal = truncated and self.truncation == "zero"
            # Real transition enters the map first, so the bootstrap sees it as tried.
            self.model.observe(obs, a, r, obs2)
            if self.arm != "random":
                self.agent.learn(ps, a, r, learn_next, terminal=(terminal or trunc_terminal))
            self.memory.append(ps, a, r)
            ret += r
            strides += 1
            obs, ps = obs2, ps2
            if terminal:
                escaped = True
                break
            if truncated:
                break
        rec = dict(tide=len(self.tide_log) + 1, escaped=escaped, strides=strides,
                   ret=round(ret, 4), pearls=pearls, key=key, map_edges=self.map_edges())
        self.tide_log.append(rec)
        return rec

    def run(self, tides: int) -> List[Dict[str, Any]]:
        for _ in range(tides):
            self.train_tide()
        return self.tide_log

    # -- frozen evaluation ------------------------------------------------
    def evaluate(self, tides: int, seed: int) -> Dict[str, Any]:
        """Frozen deployed policy: eps=0, training=False (no optimism), no Q,
        Tally or map updates, stride-level policy only. For the random arm the
        deployed policy is the random policy itself. The amnesic arm is evaluated
        at a fresh dawn with its stores wiped, as its own dawn rule dictates."""
        rng = random.Random(seed)
        if self.arm == "amnesic":
            self._build_stores()
        old_eps = self.agent.eps
        self.agent.eps = 0.0
        out = []
        try:
            for _ in range(tides):
                w = self.world_factory()
                obs = w.reset()
                marker = False
                ps = (obs[0], obs[1], marker)
                ret, strides, escaped, pearls = 0.0, 0, False, False
                path = [obs[0]]
                while True:
                    if self.arm == "random":
                        a = rng.randrange(4)
                    else:
                        a = self.agent.act(ps, rng, training=False)
                    obs, r, terminal, truncated = w.step(a)
                    tel = w.last_telemetry or {}
                    if tel.get("pearls_picked"):
                        pearls = True
                        marker = True
                    ps = (obs[0], obs[1], marker)
                    path.append(obs[0])
                    ret += r
                    strides += 1
                    if terminal:
                        escaped = True
                        break
                    if truncated:
                        break
                out.append(dict(escaped=escaped, strides=strides, ret=round(ret, 4),
                                pearls=pearls, path=[list(p) for p in path]))
        finally:
            self.agent.eps = old_eps
        return dict(escapes=sum(o["escaped"] for o in out), tides=out)
