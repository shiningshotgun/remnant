"""Tests for CrossTideExplorationMemory and the cross-tide exploration diagnostic."""

import copy
import json
import math
import os
import random
import tempfile
import unittest
from pathlib import Path

from remnant_core.exploration_memory import CrossTideExplorationMemory
from remnant_core.agent import QAgent
from remnant_core.memory import EpisodicMemory
from remnant_core.runner import LoopRunner
from remnant_core.worlds.reach import ReachWorld, REACH_LOOP_LIMIT, physical_transition
from remnant_core.repair_diagnostic import (
    parse_diagnostic_args,
    evaluate_policy,
    replay_evaluation_trajectory,
    run_diagnostic_arm,
)
from remnant_core.study_config import (
    CONDITION_BASELINE,
    CONDITION_ROUTE_B,
    CONDITION_EXPLORATION,
    SUPPORTED_CONDITIONS,
)


class TestCrossTideExplorationMemory(unittest.TestCase):
    """Unit tests verifying CrossTideExplorationMemory functionality and generic properties."""

    def test_starts_empty_and_increments(self):
        """CrossTideExplorationMemory starts empty and increments only when called."""
        mem = CrossTideExplorationMemory()
        s1 = ((20, 0), frozenset(), False)
        s2 = ((10, 4), frozenset(), True)

        self.assertEqual(mem.visit_count(s1, 0), 0)
        self.assertEqual(mem.exploration_score(s1, 0, 1.0), 1.0)

        mem.increment(s1, 0)
        self.assertEqual(mem.visit_count(s1, 0), 1)
        self.assertAlmostEqual(mem.exploration_score(s1, 0, 1.0), 1.0 / math.sqrt(2))

        # Different action or state remains 0
        self.assertEqual(mem.visit_count(s1, 1), 0)
        self.assertEqual(mem.visit_count(s2, 0), 0)

    def test_generic_function_without_exceptions(self):
        """Exploration score is a generic function of visit counts with no exceptions."""
        mem = CrossTideExplorationMemory()
        states = [
            ((0, 14), frozenset({1}), False),  # Tidegate exit pos
            ((10, 4), frozenset({0}), True),   # Chapel pearls pos
            ((24, 14), frozenset({1}), False), # Net Loft key pos
            ((6, 14), frozenset(), False),     # Harbor lock pos
            ((20, 0), frozenset(), False),     # Start pos
        ]
        coeff = 1.0
        for s in states:
            for a in range(4):
                expected = coeff / math.sqrt(1 + 0)
                self.assertAlmostEqual(mem.exploration_score(s, a, coeff), expected)

        # Increment all once and verify invariant
        for s in states:
            for a in range(4):
                mem.increment(s, a)
                expected = coeff / math.sqrt(1 + 1)
                self.assertAlmostEqual(mem.exploration_score(s, a, coeff), expected)

    def test_serialization_round_trip(self):
        """CrossTideExplorationMemory serializes to dict and restores bit-for-bit."""
        mem = CrossTideExplorationMemory()
        s1 = ((20, 0), frozenset({1}), True)
        s2 = ((10, 4), frozenset(), False)
        mem.increment(s1, 2)
        mem.increment(s1, 2)
        mem.increment(s2, 1)

        data = mem.to_dict()
        restored = CrossTideExplorationMemory.from_dict(data)

        self.assertEqual(restored.visit_count(s1, 2), 2)
        self.assertEqual(restored.visit_count(s2, 1), 1)
        self.assertEqual(restored.visit_count(s1, 0), 0)
        self.assertEqual(mem.to_dict(), restored.to_dict())


class TestAgentExplorationIntegration(unittest.TestCase):
    """Tests verifying that exploration score affects training action selection only."""

    def test_exploration_affects_training_action_selection_only(self):
        """Bonus is added to action selection during training, but NOT in evaluation or qval."""
        exp_mem = CrossTideExplorationMemory()
        agent = QAgent(exploration_memory=exp_mem, exploration_coefficient=1.0, eps=0.0)
        s = ((20, 0), frozenset(), False)

        # Manually set Q-values equal to zero
        for a in range(4):
            self.assertEqual(agent.qval(s, a), 0.0)

        # Action 0 visited 99 times, action 1 visited 0 times
        for _ in range(99):
            exp_mem.increment(s, 0)

        # In training, action 1 has score 1.0, action 0 has score 1.0 / sqrt(100) = 0.1
        rng = random.Random(42)
        chosen_train = agent.act(s, rng, training=True)
        self.assertEqual(chosen_train, 1)

        # In evaluation (training=False), all qval are 0.0 so exploration bonus is excluded
        # qval must remain 0.0
        self.assertEqual(agent.qval(s, 0), 0.0)
        self.assertEqual(agent.qval(s, 1), 0.0)

        # And learn target must not include exploration bonus
        agent.learn(s, 1, 0.0, s, terminal=False)
        self.assertEqual(agent.q.get((s, 1)), 0.0)

    def test_runner_persists_exploration_counts_across_resets(self):
        """LoopRunner with cross_tide_exploration_memory_v1 persists counts across tides."""
        runner = LoopRunner(seed=301, max_loops=3, condition=CONDITION_EXPLORATION)
        res = runner.run()
        self.assertIsNotNone(runner.agent.exploration_memory)
        # Verify that total visits across loops equal total real interactions
        total_visits = sum(runner.agent.exploration_memory.counts.values())
        self.assertEqual(total_visits, res["total_real_interactions"])
        self.assertGreater(total_visits, 0)


class TestDiagnosticParserAndExecution(unittest.TestCase):
    """Tests verifying the diagnostic parser and trajectory storage contract."""

    def test_parser_options(self):
        """Parser accepts --exploration-only, --exploration-output-dir, and --exploration-coefficient."""
        args = parse_diagnostic_args([
            "--seeds", "301",
            "--horizon", "2",
            "--eval-tides", "1",
            "--exploration-only",
            "--exploration-output-dir", "results_ctem_test_v1",
            "--exploration-coefficient", "1.5",
        ])
        self.assertTrue(args.exploration_only)
        self.assertEqual(args.exploration_output_dir, "results_ctem_test_v1")
        self.assertEqual(args.exploration_coefficient, 1.5)
        self.assertEqual(args.seeds, [301])

    def test_evaluation_trajectories_saved_and_replayed(self):
        """Full evaluation trajectories save to evaluation_trajectories.json and replay cleanly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = os.path.join(tmp_dir, "test_arm")
            arm_res = run_diagnostic_arm(
                seeds=[301],
                horizon=2,
                condition=CONDITION_EXPLORATION,
                eval_tides=1,
                output_dir=out_dir,
                command_str="test_cmd",
                exploration_coefficient=1.0,
            )
            traj_file = os.path.join(out_dir, "evaluation_trajectories.json")
            self.assertTrue(os.path.exists(traj_file))
            with open(traj_file) as f:
                traj_data = json.load(f)

            self.assertIn("301", traj_data)
            pre_records = traj_data["301"]["pre_training_control"]["records"]
            post_records = traj_data["301"]["post_training_test"]["records"]
            self.assertEqual(len(pre_records), REACH_LOOP_LIMIT)
            self.assertEqual(len(post_records), REACH_LOOP_LIMIT)

            # Replay from saved file
            ok_pre, err_pre = replay_evaluation_trajectory(pre_records)
            self.assertTrue(ok_pre, err_pre)
            ok_post, err_post = replay_evaluation_trajectory(post_records)
            self.assertTrue(ok_post, err_post)


if __name__ == "__main__":
    unittest.main()
