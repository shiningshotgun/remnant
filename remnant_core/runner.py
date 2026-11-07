"""Loop runner for Remnant: executes an agent through time loops in a resetting world.

Connects the world, agent, memory, and model across loops.
Enforces:
1. No learning transition crosses a reset boundary.
2. Complete RNG isolation via seeded streams.
3. Separate accounting of real interactions, Q-updates, and wall-clock time.
"""

import random
import time
from typing import Callable, Dict, List, Optional, Tuple, Any

from remnant_core.config import (
    LOOP_LIMIT_L,
    HORIZON_K_MAX,
    EXIT_POS,
)
from remnant_core.worlds.grid import LoopWorld
from remnant_core.memory import EpisodicMemory
from remnant_core.agent import QAgent
from remnant_core.model import WorldModel
from remnant_core.exploration_memory import CrossTideExplorationMemory

State = Tuple[Tuple[int, int], Any]
Transition = Tuple[State, int, float, State, bool]


class LoopRunner:
    """Runs an agent through time loops with persistent learning stores."""

    def __init__(
        self,
        seed: int,
        max_loops: int = HORIZON_K_MAX,
        on_learn_callback: Optional[Callable[..., None]] = None,
        world: Optional[Any] = None,
        condition: str = "tally_current_tide_pearl_v1",
        exploration_coefficient: float = 1.0,
        stop_on_escape: bool = True,
    ):
        self.seed = seed
        self.max_loops = max_loops
        self.on_learn_callback = on_learn_callback
        self.condition = condition
        self.stop_on_escape = stop_on_escape
        self.exploration_coefficient = exploration_coefficient
        self.use_tally_marker = (condition in ("tally_current_tide_pearl_v1", "cross_tide_exploration_memory_v1", "corrected_control_v1", "corrected_node_novelty_v1", "corrected_frontier_v1", "corrected_meta_v1", "corrected_meta_random_v1", "corrected_metaq_v1", "corrected_metatrace_v1", "corrected_metabucket_v1", "corrected_metabucket_td_v1"))
        self.tally_pearl_picked_this_tide: bool = False

        # Seeded components
        self.action_rng = random.Random(seed * 1000 + 1)
        self.world = world if world is not None else LoopWorld(max_steps=LOOP_LIMIT_L)
        self.memory = EpisodicMemory(seed=seed * 1000 + 6,
                                     exact_match=condition.startswith("corrected_"))
        if condition == "cross_tide_exploration_memory_v1":
            self.exploration_memory = CrossTideExplorationMemory()
        else:
            self.exploration_memory = None
        self.model = WorldModel()
        if condition in ("corrected_node_novelty_v1", "corrected_frontier_v1", "corrected_meta_v1", "corrected_meta_random_v1", "corrected_metaq_v1", "corrected_metatrace_v1", "corrected_metabucket_v1", "corrected_metabucket_td_v1"):
            from remnant_core.node_novelty import NodeNovelty
            self.node_novelty = NodeNovelty(self.model)
        else:
            self.node_novelty = None
        self.agent = QAgent(
            memory=self.memory,
            exploration_memory=self.exploration_memory,
            exploration_coefficient=exploration_coefficient,
            node_novelty=self.node_novelty,
            trace_decay=(0.95 if condition in ("corrected_frontier_v1", "corrected_meta_v1", "corrected_meta_random_v1", "corrected_metaq_v1", "corrected_metatrace_v1", "corrected_metabucket_v1", "corrected_metabucket_td_v1") else None),
            raw_q_bootstrap=condition.startswith("corrected_"),
        )

        self.escape_trajectories: list = []
        self.meta_q = [0.0, 0.0, 0.0]          # context-free bandit (failed candidate, kept for record)
        self.metaq = {}
        self.meta_e = {}          # dawn-spanning replacing traces (Peng semantics: persist through epsilon picks)
        self.meta_lambda = 0.95   # lambda_meta, fixed a priori
        self.meta_gamma = 0.95                  # dawn-level discount across loops
        self.meta_rng = random.Random(seed * 1000 + 7)
        self.meta_eps = 0.2
        self.meta_lr = 0.3
        self.mode_log = []

        # Cumulative stores
        self.loop_records: List[Dict[str, Any]] = []

        # Accounting counters
        self.total_real_interactions: int = 0
        self.total_q_updates: int = 0
        self.start_time: float = 0.0
        self.total_wall_clock: float = 0.0

    def get_action_rng_state(self):
        """Serialize action RNG state."""
        raw = self.action_rng.getstate()
        return (raw[0], list(raw[1]), raw[2])

    def set_action_rng_state(self, state_tuple):
        """Restore action RNG state."""
        st = (state_tuple[0], tuple(state_tuple[1]), state_tuple[2])
        self.action_rng.setstate(st)

    def build_policy_state(self, obs: State, marker: bool):
        """Construct decision-usable policy state from external observation and tally marker."""
        if self.use_tally_marker:
            return (obs[0], obs[1], bool(marker))
        return obs

    def run(self) -> Dict[str, Any]:
        """Execute the multi-loop run."""
        self.start_time = time.perf_counter()
        first_escape_loop: Optional[int] = None

        for loop_idx in range(1, self.max_loops + 1):
            loop_record = self._run_single_loop(loop_idx)
            self.loop_records.append(loop_record)

            if loop_record["escaped"] and first_escape_loop is None:
                first_escape_loop = loop_idx
            if loop_record["escaped"] and self.stop_on_escape:
                break

        self.total_wall_clock = time.perf_counter() - self.start_time

        if first_escape_loop is not None:
            status = "escaped"
            escaped = True
        else:
            escaped = False
            status = "loop_cap_reached"

        model_st_actions = len(self.model.counts)
        model_transitions = sum(len(sn) for sn in self.model.counts.values())

        return {
            "seed": self.seed,
            "max_loops": self.max_loops,
            "status": status,
            "escaped": escaped,
            "escape_loop": first_escape_loop,
            "total_loops_run": len(self.loop_records),
            "total_real_interactions": self.total_real_interactions,
            "total_q_updates": self.total_q_updates,
            "total_wall_clock_seconds": self.total_wall_clock,
            "model_size": {
                "state_actions": model_st_actions,
                "transitions": model_transitions,
            },
            "q_table_size": len(self.agent.q),
            "memory_entries": len(self.memory.entries),
            "loop_records": self.loop_records,
        }

    def _frontier_action(self, policy_state, obs):
        """Plan-to-frontier over the learned model: untried action here, else BFS to nearest
        visited position with an untried action. None if no frontier is known."""
        import collections
        pos, items = obs[0], obs[1]
        untried = [a for a in range(4) if (pos, items, a) not in
                   {(s[0], s[1], act) for (s, act) in self.model.counts.keys()}]
        # faster: build key set once
        tried = {(s, act) for (s, act) in self.model.counts.keys()}
        untried = [a for a in range(4) if ((pos, items), a) not in tried]
        if untried:
            return untried[0]
        # BFS over observed transitions (position level, most frequent next pos)
        def next_pos(p2, it2, a2):
            pred, _r = self.model.predict((p2, it2), a2)
            if pred is None:
                return None
            return pred[0]
        start = (pos, items)
        prev = {start: None}
        q = collections.deque([start])
        target = None
        while q:
            cur = q.popleft()
            cp, ci = cur
            if any(((cp, ci), a) not in tried for a in range(4)):
                target = cur
                break
            for a in range(4):
                if ((cp, ci), a) not in tried:
                    continue
                np_ = next_pos(cp, ci, a)
                if np_ is None or np_ == cp:
                    continue
                ns = (np_, ci)
                if ns not in prev:
                    prev[ns] = (cur, a)
                    q.append(ns)
        if target is None:
            return None
        # walk back to first action
        node = target
        while prev[node] is not None and prev[node][0] != start:
            node = prev[node][0]
        if prev[node] is None:
            return None
        return prev[node][1]

    def _meta_state(self):
        """(frontier_exists, escape_experienced) from Tally contents only."""
        tried = set(self.model.counts.keys())
        frontier = False
        for pos in self.node_novelty.visited:
            for items in (frozenset(), frozenset({1})):
                if any(((pos, items), a) not in tried for a in range(4)):
                    frontier = True
                    break
            if frontier:
                break
        if self.condition in ("corrected_metabucket_v1", "corrected_metabucket_td_v1"):
            # Map-progress-visible state: how much known-but-unexplored map remains.
            # frontier_n = visited (pos, items) states with at least one untried action.
            # Bucket edges fixed a priori: 0 / 1-5 / 6-15 / 16-30 / >30.
            fn = 0
            for pos2 in self.node_novelty.visited:
                for items2 in (frozenset(), frozenset({1})):
                    if any(((pos2, items2), a2) not in tried for a2 in range(4)):
                        fn += 1
            if fn == 0:
                b = 0
            elif fn <= 5:
                b = 1
            elif fn <= 15:
                b = 2
            elif fn <= 30:
                b = 3
            else:
                b = 4
            return (b, bool(self.escape_trajectories))
        return (frontier, bool(self.escape_trajectories))

    def _metaq_val(self, s, m):
        return self.metaq.get((s, m), 0.0)

    def _goal_action(self, obs):
        """BFS over the learned model toward the destination of the highest-reward transition
        ever experienced; first action of the shortest known path, or None."""
        import collections
        best_r, goal_state = None, None
        for (s, a) in self.model.counts:
            pred, mean_r = self.model.predict(s, a)
            if pred is not None and (best_r is None or mean_r > best_r):
                best_r, goal_state = mean_r, pred
        if goal_state is None:
            return None
        start = (obs[0], obs[1])
        if start == goal_state:
            return None
        prev = {start: None}
        q = collections.deque([start])
        goal = None
        while q:
            cur = q.popleft()
            if cur == goal_state:
                goal = cur
                break
            for a in range(4):
                pred, _r = self.model.predict(cur, a)
                if pred is None or pred[0] == cur[0]:
                    continue
                ns = (pred[0], pred[1])
                if ns not in prev:
                    prev[ns] = (cur, a)
                    q.append(ns)
        if goal is None:
            return None
        node = goal
        while prev[node] is not None and prev[node][0] != start:
            node = prev[node][0]
        if prev[node] is None:
            return None
        return prev[node][1]

    def _run_single_loop(self, loop_idx: int) -> Dict[str, Any]:
        """Run loop k: real episode followed by model ingestion.
        
        The internal tally marker is reset at dawn before the dawn observation
        is evaluated or acted upon. No learning transition crosses reset.
        """
        obs = self.world.reset()
        self.tally_pearl_picked_this_tide = False
        policy_state = self.build_policy_state(obs, self.tally_pearl_picked_this_tide)
        dawn_policy_state = policy_state

        self.agent.reset_traces()
        if self.condition in ("corrected_meta_v1", "corrected_meta_random_v1", "corrected_metaq_v1", "corrected_metatrace_v1", "corrected_metabucket_v1", "corrected_metabucket_td_v1"):
            if self.condition == "corrected_meta_random_v1":
                mode = self.meta_rng.randrange(3)
                ms = None
            elif self.condition in ("corrected_metaq_v1", "corrected_metatrace_v1", "corrected_metabucket_v1", "corrected_metabucket_td_v1"):
                ms = self._meta_state()
                if self.meta_rng.random() < self.meta_eps:
                    mode = self.meta_rng.randrange(3)
                else:
                    best = max(self._metaq_val(ms, m) for m in range(3))
                    cands = [m for m in range(3) if self._metaq_val(ms, m) == best]
                    mode = self.meta_rng.choice(cands)
            elif self.meta_rng.random() < self.meta_eps:
                mode = self.meta_rng.randrange(3)
                ms = None
            else:
                best = max(self.meta_q)
                cands = [m for m in range(3) if self.meta_q[m] == best]
                mode = self.meta_rng.choice(cands)
                ms = None
            meta_before = list(self.meta_q)
        else:
            mode = None
            meta_before = None
            ms = None
        loop_transitions: List[Transition] = []
        loop_prov: List[str] = []
        loop_policy_transitions = []
        loop_reward: float = 0.0
        loop_real_steps: int = 0
        escaped: bool = False

        while True:
            if self.condition == "corrected_frontier_v1":
                a = None
                if self.escape_trajectories:
                    ga = self._goal_action(obs)
                    if ga is not None:
                        a = ga
                if a is None:
                    fa = self._frontier_action(policy_state, obs)
                    if fa is not None:
                        a = fa
                if a is None:
                    a = self.agent.act(policy_state, self.action_rng)
            elif self.condition in ("corrected_meta_v1", "corrected_meta_random_v1", "corrected_metaq_v1", "corrected_metatrace_v1", "corrected_metabucket_v1", "corrected_metabucket_td_v1"):
                a = None
                prov = None
                if mode == 2:
                    ga = self._goal_action(obs)
                    if ga is not None:
                        a = ga
                        prov = "goal_planner"
                elif mode == 1:
                    fa = self._frontier_action(policy_state, obs)
                    if fa is not None:
                        a = fa
                        prov = "frontier_planner"
                if a is None:
                    a = self.agent.act(policy_state, self.action_rng)
                    prov = "policy"
                loop_prov.append(prov)
            else:
                a = self.agent.act(policy_state, self.action_rng)
            obs_next, r, terminal, truncated = self.world.step(a)

            loop_real_steps += 1
            self.total_real_interactions += 1
            loop_reward += r

            # Determine experienced pearl pickup outcome on just-executed transition
            tel = getattr(self.world, "last_telemetry", None)
            pearls_picked = bool(tel.get("pearls_picked", False)) if tel is not None else False
            if pearls_picked:
                next_tally_marker = True
            else:
                next_tally_marker = self.tally_pearl_picked_this_tide

            next_policy_state = self.build_policy_state(obs_next, next_tally_marker)

            if self.on_learn_callback is not None:
                try:
                    self.on_learn_callback(obs, a, r, obs_next, terminal, loop_idx, loop_real_steps, policy_state, next_policy_state)
                except TypeError:
                    self.on_learn_callback(obs, a, r, obs_next, terminal, loop_idx, loop_real_steps)

            # Agent updates and memory storage use policy_state
            # Toll correction: at truncation the world's true next state is dawn, not the stride-54 position.
            # Corrected arms only: at truncation the world's true next state is dawn,
            # not the never-acted-from stride-54 position. Default keeps original.
            learn_next = (dawn_policy_state if (truncated and self.condition.startswith("corrected_"))
                          else next_policy_state)
            r_learn = r
            if self.condition == "corrected_node_novelty_v1" and not terminal:
                if obs_next[0] not in self.node_novelty.visited:
                    r_learn = r + self.node_novelty.coefficient
            self.agent.learn(policy_state, a, r_learn, learn_next, terminal=terminal)
            self.total_q_updates += 1
            self.memory.append(policy_state, a, r)
            if self.exploration_memory is not None:
                self.exploration_memory.increment(policy_state, a)

            # World model observes real environmental transitions (external observations only)
            loop_transitions.append((obs, a, r, obs_next, terminal))
            if self.condition in ("corrected_frontier_v1", "corrected_meta_v1", "corrected_meta_random_v1", "corrected_metaq_v1", "corrected_metatrace_v1", "corrected_metabucket_v1", "corrected_metabucket_td_v1"):
                self.model.observe(obs, a, r, obs_next)

            loop_policy_transitions.append((policy_state, a, r, learn_next, terminal))
            obs = obs_next
            if self.node_novelty is not None:
                self.node_novelty.mark(obs_next[0])
            self.tally_pearl_picked_this_tide = next_tally_marker
            policy_state = next_policy_state

            if terminal:
                escaped = True
                break
            if truncated:
                break

        # Ingest real transitions into model (frontier/meta arms ingest live, per stride)
        if self.condition not in ("corrected_frontier_v1", "corrected_meta_v1", "corrected_meta_random_v1", "corrected_metaq_v1", "corrected_metatrace_v1", "corrected_metabucket_v1", "corrected_metabucket_td_v1"):
            for trans in loop_transitions:
                self.model.observe(trans[0], trans[1], trans[2], trans[3])

        # Frontier arm: replay experienced escape trajectories from episodic record
        if self.condition == "corrected_frontier_v1":
            if escaped:
                self.escape_trajectories.append(list(loop_policy_transitions))
        if self.condition in ("corrected_meta_v1", "corrected_meta_random_v1", "corrected_metaq_v1", "corrected_metatrace_v1", "corrected_metabucket_v1", "corrected_metabucket_td_v1"):
            if escaped:
                self.escape_trajectories.append(list(loop_policy_transitions))
            entry = {
                "loop": loop_idx, "mode": mode, "return": round(loop_reward, 4),
                "escaped": escaped,
                "map_edges": sum(len(sn) for sn in self.model.counts.values()),
            }
            if self.condition == "corrected_meta_v1":
                self.meta_q[mode] += self.meta_lr * (loop_reward - self.meta_q[mode])
                entry["meta_after"] = [round(v, 4) for v in self.meta_q]
            elif self.condition in ("corrected_metaq_v1", "corrected_metatrace_v1", "corrected_metabucket_v1", "corrected_metabucket_td_v1"):
                ms_next = self._meta_state()
                old_v = self._metaq_val(ms, mode)
                target = loop_reward + self.meta_gamma * max(self._metaq_val(ms_next, m) for m in range(3))
                delta = target - old_v
                if self.condition in ("corrected_metatrace_v1", "corrected_metabucket_v1"):
                    # Peng-style: no Watkins cut on epsilon-chosen modes; traces span dawns.
                    # Not exactly off-policy Q(lambda) for the greedy meta-policy; no
                    # lambda-return-equivalence or convergence claim. This is the approved
                    # real-history credit: real experienced choices get credit.
                    for k in list(self.meta_e):
                        self.meta_e[k] *= self.meta_gamma * self.meta_lambda
                    self.meta_e[(ms, mode)] = 1.0
                    for k, e in self.meta_e.items():
                        self.metaq[k] = self.metaq.get(k, 0.0) + self.meta_lr * delta * e
                else:
                    self.metaq[(ms, mode)] = old_v + self.meta_lr * delta
                entry["meta_state"] = ms
                entry["meta_state_next"] = ms_next
                entry["metaq"] = {str(k): round(v, 3) for k, v in sorted(self.metaq.items())}
                if self.meta_e:
                    entry["meta_e_max"] = round(max(self.meta_e.values()), 4)
            if loop_prov:
                entry["provenance_counts"] = {p2: loop_prov.count(p2) for p2 in ("goal_planner", "frontier_planner", "policy") if loop_prov.count(p2)}
                if escaped:
                    entry["escape_stride_provenance"] = list(loop_prov)
            self.mode_log.append(entry)


        return {
            "loop": loop_idx,
            "real_steps": loop_real_steps,
            "cumulative_reward": round(loop_reward, 4),
            "escaped": escaped,
            "q_table_size": len(self.agent.q),
            "memory_size": len(self.memory.entries),
        }


if __name__ == "__main__":
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 101
    runner = LoopRunner(seed=seed)
    result = runner.run()

    print(f"Seed: {result['seed']}")
    print(f"Status: {result['status']}")
    print(f"Escaped: {result['escaped']}")
    print(f"Escape loop: {result['escape_loop']}")
    print(f"Total loops run: {result['total_loops_run']}")
    print(f"Total real interactions: {result['total_real_interactions']}")
    print(f"Total Q-updates: {result['total_q_updates']}")
    print(f"Model size: {result['model_size']}")
    print(f"Q-table size: {result['q_table_size']}")
    print(f"Memory entries: {result['memory_entries']}")
    print(f"Wall-clock: {result['total_wall_clock_seconds']:.4f}s")
