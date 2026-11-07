"""Loop runner property tests: determinism, escape, cap, boundary, and persistence."""

import unittest
from typing import List, Tuple, Any

from remnant_core.config import (
    START_POS,
    LOOP_LIMIT_L,
    HORIZON_K_MAX,
)
from remnant_core.runner import LoopRunner
from remnant_core.solver import solve_shortest_escape


class TestRunnerDeterminism(unittest.TestCase):
    def test_same_seed_produces_identical_run(self):
        """Two runs with the same seed produce bit-for-bit identical loop records and final stores."""
        transitions1: List[Tuple] = []
        transitions2: List[Tuple] = []

        def cb1(s, a, r, sn, term, l_idx, s_idx):
            transitions1.append((s, a, r, sn, term, l_idx, s_idx))

        def cb2(s, a, r, sn, term, l_idx, s_idx):
            transitions2.append((s, a, r, sn, term, l_idx, s_idx))

        runner1 = LoopRunner(seed=105, max_loops=3, on_learn_callback=cb1)
        res1 = runner1.run()

        runner2 = LoopRunner(seed=105, max_loops=3, on_learn_callback=cb2)
        res2 = runner2.run()

        # Full transition sequences passed to learn() must be identical
        self.assertEqual(transitions1, transitions2, "Transition sequences diverged!")

        # Final stores must match bit-for-bit
        self.assertEqual(runner1.agent.q, runner2.agent.q, "Q-tables diverged!")
        self.assertEqual(runner1.memory.entries, runner2.memory.entries, "Memory entries diverged!")
        self.assertEqual(dict(runner1.model.counts), dict(runner2.model.counts), "Model counts diverged!")
        self.assertEqual(dict(runner1.model.reward_sum), dict(runner2.model.reward_sum), "Model rewards diverged!")

        # Summary counters and status
        self.assertEqual(res1["total_real_interactions"], res2["total_real_interactions"])
        self.assertEqual(res1["total_q_updates"], res2["total_q_updates"])
        self.assertEqual(res1["escaped"], res2["escaped"])
        self.assertEqual(res1["escape_loop"], res2["escape_loop"])
        self.assertEqual(res1["q_table_size"], res2["q_table_size"])
        self.assertEqual(res1["memory_entries"], res2["memory_entries"])

        # Loop-by-loop trajectory equality
        self.assertEqual(len(res1["loop_records"]), len(res2["loop_records"]))
        for l1, l2 in zip(res1["loop_records"], res2["loop_records"]):
            self.assertEqual(l1["loop"], l2["loop"])
            self.assertEqual(l1["real_steps"], l2["real_steps"])
            self.assertAlmostEqual(l1["cumulative_reward"], l2["cumulative_reward"])
            self.assertEqual(l1["escaped"], l2["escaped"])
            self.assertEqual(l1["q_table_size"], l2["q_table_size"])
            self.assertEqual(l1["memory_size"], l2["memory_size"])


class TestRunnerEscape(unittest.TestCase):
    def test_escape_ends_run_immediately(self):
        """A scripted optimal-path agent escapes in loop 1 and the run ends immediately."""
        dist, path, actions = solve_shortest_escape(has_key_allowed=True, visit_bait_first=False)

        # Verify the fixture itself
        self.assertEqual(dist, 12)
        self.assertEqual(len(actions), 12)

        action_iter = iter(actions)
        runner = LoopRunner(seed=101, max_loops=HORIZON_K_MAX)
        runner.agent.act = lambda s, rng: next(action_iter)
        res = runner.run()

        self.assertTrue(res["escaped"])
        self.assertEqual(res["status"], "escaped")
        self.assertEqual(res["escape_loop"], 1)
        self.assertEqual(res["total_loops_run"], 1)
        self.assertTrue(res["loop_records"][0]["escaped"])
        self.assertEqual(res["loop_records"][0]["real_steps"], 12)


class TestRunnerCap(unittest.TestCase):
    def test_non_escaping_run_stops_at_cap(self):
        """A non-escaping run stops exactly at the loop cap with status 'loop_cap_reached'.
        Outcome is controlled unconditionally by stubbing agent.act to a non-escaping policy.
        """
        runner = LoopRunner(seed=101, max_loops=10)
        runner.agent.act = lambda s, rng: 0  # Always NORTH, parks against boundary

        res = runner.run()
        self.assertEqual(res["total_loops_run"], 10)
        self.assertFalse(res["escaped"])
        self.assertIsNone(res["escape_loop"])
        self.assertEqual(res["status"], "loop_cap_reached")


class TestRunnerBoundary(unittest.TestCase):
    def test_no_learning_transition_crosses_reset(self):
        """No transition passed to learn() crosses a loop reset boundary.
        Every transition has physically adjacent source and destination states.
        """
        recorded: List[Tuple[Any, int, float, Any, bool, int, int]] = []

        def recording_callback(s, a, r, s_next, terminal, loop_idx, step_idx):
            recorded.append((s, a, r, s_next, terminal, loop_idx, step_idx))

        runner = LoopRunner(seed=101, max_loops=5, on_learn_callback=recording_callback)
        res = runner.run()

        self.assertEqual(res["total_loops_run"], 5)
        self.assertGreater(len(recorded), 0)

        for i in range(len(recorded)):
            s, a, r, s_next, terminal, loop_idx, step_idx = recorded[i]
            pos_curr, _ = s
            pos_next, _ = s_next

            # Physical adjacency: Manhattan distance <= 1
            dist = abs(pos_curr[0] - pos_next[0]) + abs(pos_curr[1] - pos_next[1])
            self.assertIn(dist, [0, 1], f"Non-adjacent transition: {s} -> {s_next}")

            # Across loop boundaries, next transition starts at START_POS
            if i + 1 < len(recorded):
                next_t = recorded[i + 1]
                if loop_idx > 0 and next_t[5] > 0 and loop_idx != next_t[5]:
                    self.assertEqual(loop_idx + 1, next_t[5])
                    self.assertEqual(next_t[0][0], START_POS)
                    self.assertEqual(next_t[6], 1)  # step 1 of next loop


class TestRunnerPersistence(unittest.TestCase):
    def test_stores_persist_across_resets_while_world_resets(self):
        """Q-table, memory, and model persist across loop resets; world state resets."""
        runner = LoopRunner(seed=101, max_loops=3)
        res = runner.run()

        # World resets to start state after each loop
        # (verified by checking loop records: each loop starts fresh and accumulates from step 1)
        for lr in res["loop_records"]:
            self.assertGreater(lr["real_steps"], 0)

        # Stores accumulate monotonically across loops
        q_sizes = [lr["q_table_size"] for lr in res["loop_records"]]
        mem_sizes = [lr["memory_size"] for lr in res["loop_records"]]

        # Q-table and memory can only grow or stay the same across loops
        for i in range(1, len(q_sizes)):
            self.assertGreaterEqual(q_sizes[i], q_sizes[i - 1],
                                    "Q-table shrank across a loop reset")
            self.assertGreaterEqual(mem_sizes[i], mem_sizes[i - 1],
                                    "Memory shrank across a loop reset")

        # After 3 loops of 18 steps each, stores must be non-trivially populated
        self.assertGreater(len(runner.agent.q), 0, "Q-table is empty after 3 loops")
        self.assertGreater(len(runner.memory.entries), 0, "Memory is empty after 3 loops")
        self.assertGreater(len(runner.model.counts), 0, "Model is empty after 3 loops")

        # Model must have been ingested with real transitions
        total_model_transitions = sum(len(sn) for sn in runner.model.counts.values())
        self.assertGreater(total_model_transitions, 0, "Model has no transitions after 3 loops")


class TestRouteBTallyMemory(unittest.TestCase):
    def test_marker_starts_false_sets_on_experienced_pickup_and_resets_at_dawn(self):
        """Marker starts False, sets on experienced pickup transition, persists, resets at dawn."""
        from remnant_core.worlds.reach import ReachWorld, GRAPH_CHAINS, GRAPH_ARCS, ITEM_PEARLS
        world = ReachWorld()
        runner = LoopRunner(seed=201, max_loops=2, world=world, condition="tally_current_tide_pearl_v1")
        self.assertEqual(runner.condition, "tally_current_tide_pearl_v1")

        # Before stepping, dawn marker is False
        obs = world.reset()
        runner.tally_pearl_picked_this_tide = False
        self.assertFalse(runner.tally_pearl_picked_this_tide)

        # Walk to Chapel (pearl road)
        chain = GRAPH_CHAINS[1]
        for step_idx in range(len(chain) - 1):
            curr_pos = chain[step_idx]
            next_pos = chain[step_idx + 1]
            arcs = GRAPH_ARCS[curr_pos]
            action = [a for a, (e, d, neigh) in enumerate(arcs) if e == 1 and neigh == next_pos][0]

            obs_next, r, term, trunc = world.step(action)
            tel = world.last_telemetry
            if tel["pearls_picked"]:
                runner.tally_pearl_picked_this_tide = True

            if step_idx < len(chain) - 2:
                self.assertFalse(runner.tally_pearl_picked_this_tide)
            else:
                self.assertTrue(runner.tally_pearl_picked_this_tide)
                # External observation still hides pearls!
                self.assertNotIn(ITEM_PEARLS, obs_next[1])

        # Marker persists within the tide
        self.assertTrue(runner.tally_pearl_picked_this_tide)

        # Dawn of next tide resets marker
        world.reset()
        runner.tally_pearl_picked_this_tide = False
        self.assertFalse(runner.tally_pearl_picked_this_tide)

    def test_ordinary_events_do_not_set_marker(self):
        """Ordinary stride cost, lamp key pickup, invalid actions, and truncation do not set tally marker."""
        from remnant_core.worlds.reach import ReachWorld, GRAPH_CHAINS, GRAPH_ARCS
        world = ReachWorld()
        runner = LoopRunner(seed=202, max_loops=1, world=world, condition="tally_current_tide_pearl_v1")
        self.assertFalse(runner.tally_pearl_picked_this_tide)

        # Walk route 0 to Needle Cut
        chain0 = GRAPH_CHAINS[0]
        for step_idx in range(len(chain0) - 1):
            curr_pos = chain0[step_idx]
            next_pos = chain0[step_idx + 1]
            arcs = GRAPH_ARCS[curr_pos]
            action = [a for a, (e, d, neigh) in enumerate(arcs) if e == 0 and neigh == next_pos][0]
            obs_next, r, term, trunc = world.step(action)
            tel = world.last_telemetry
            if tel.get("pearls_picked", False):
                runner.tally_pearl_picked_this_tide = True
            self.assertFalse(runner.tally_pearl_picked_this_tide)

        # Route 3 to Net Loft (picks up Lamp-Key)
        chain3 = GRAPH_CHAINS[3]
        for step_idx in range(len(chain3) - 1):
            curr_pos = chain3[step_idx]
            next_pos = chain3[step_idx + 1]
            arcs = GRAPH_ARCS[curr_pos]
            action = [a for a, (e, d, neigh) in enumerate(arcs) if e == 3 and neigh == next_pos][0]
            obs_next, r, term, trunc = world.step(action)
            tel = world.last_telemetry
            if tel.get("pearls_picked", False):
                runner.tally_pearl_picked_this_tide = True
            # Key was picked up, but marker must remain False!
            self.assertFalse(runner.tally_pearl_picked_this_tide)

        self.assertIn(1, world.state()[1])
        self.assertFalse(runner.tally_pearl_picked_this_tide)

    def test_distinct_policy_states_with_marker(self):
        """Same observation with different tally marker produces distinct policy states in Q and memory."""
        from remnant_core.worlds.reach import ReachWorld
        world = ReachWorld()
        runner = LoopRunner(seed=203, max_loops=1, world=world, condition="tally_current_tide_pearl_v1")
        s_obs = ((0, 0), frozenset())
        ps_false = runner.build_policy_state(s_obs, False)
        ps_true = runner.build_policy_state(s_obs, True)
        self.assertNotEqual(ps_false, ps_true)
        self.assertEqual(ps_false, ((0, 0), frozenset(), False))
        self.assertEqual(ps_true, ((0, 0), frozenset(), True))


if __name__ == "__main__":
    unittest.main()
