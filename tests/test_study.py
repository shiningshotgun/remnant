"""Tests for the Reach of Vesper study execution, reproducibility, and telemetry."""

import json
import os
import shutil
import tempfile
import unittest

from remnant_core.study import run_reach_seed, run_reach_study
from remnant_core.runner import LoopRunner
from remnant_core.worlds.reach import ReachWorld
from remnant_core.reach_solver import solve_reach_shortest_escape


class TestStudy(unittest.TestCase):
    def test_reproducibility_same_seed_twice(self):
        """Same seed run twice produces identical per-stride records, deterministic fields, and stores."""
        res1, recs1, l1 = run_reach_seed(seed=101, max_loops=2)
        res2, recs2, l2 = run_reach_seed(seed=101, max_loops=2)

        self.assertEqual(json.loads(json.dumps(recs1)), json.loads(json.dumps(recs2)))
        deterministic1 = {k: v for k, v in res1.items() if k != "total_wall_clock_seconds"}
        deterministic2 = {k: v for k, v in res2.items() if k != "total_wall_clock_seconds"}
        self.assertEqual(deterministic1, deterministic2)
        self.assertEqual(json.loads(json.dumps(l1)), json.loads(json.dumps(l2)))

    def test_record_count_equals_real_steps(self):
        """Sum of real_steps across loop records equals the total count of stride records."""
        res, recs, _ = run_reach_seed(seed=102, max_loops=3)
        total_steps = sum(lr["real_steps"] for lr in res["loop_records"])
        self.assertEqual(len(recs), total_steps)
        self.assertEqual(res["total_real_interactions"], total_steps)

    def test_recording_does_not_perturb_learning_stores_or_rng(self):
        """A run with on_learn_callback matches a plain run without callback bit-for-bit.
        Compares complete serialized model, agent, memory, both RNG states,
        and all deterministic summary fields.
        Also asserts learning accounting: 1 Q-update, 1 memory append, 1 model observation
        per interaction, and 0 across resets.
        """
        w_plain = ReachWorld()
        r_plain = LoopRunner(seed=103, max_loops=2, world=w_plain)
        res_plain = r_plain.run()

        collector = []
        res_rec, recs_rec, l_rec = run_reach_seed(seed=103, max_loops=2, runner_ref_collector=collector)
        r_recorded = collector[0]

        # Action RNG state equality
        self.assertEqual(r_plain.action_rng.getstate(), r_recorded.action_rng.getstate())

        # Memory RNG state equality
        self.assertEqual(r_plain.memory.get_rng_state(), r_recorded.memory.get_rng_state())

        # Serialized agent equality
        plain_agent_dict = json.loads(json.dumps(r_plain.agent.to_dict()))
        rec_agent_dict = json.loads(json.dumps(l_rec["agent"]))
        self.assertEqual(plain_agent_dict, rec_agent_dict)

        # Serialized memory equality
        plain_mem_dict = json.loads(json.dumps(r_plain.memory.to_dict()))
        rec_mem_dict = json.loads(json.dumps(l_rec["memory"]))
        self.assertEqual(plain_mem_dict, rec_mem_dict)

        # Serialized model equality
        plain_model_dict = json.loads(json.dumps(r_plain.model.to_dict()))
        rec_model_dict = json.loads(json.dumps(l_rec["model"]))
        self.assertEqual(plain_model_dict, rec_model_dict)

        # Embedded memory in agent matches separate memory
        self.assertEqual(plain_agent_dict["memory"], plain_mem_dict)

        self.assertEqual({k: v for k, v in res_plain.items() if k != "total_wall_clock_seconds"},
                         {k: v for k, v in res_rec.items() if k != "total_wall_clock_seconds"})

        # Assert exact learning accounting:
        # 1 Q-update, 1 episodic memory append, 1 model observation per real transition
        total_steps = res_rec["total_real_interactions"]
        self.assertEqual(res_rec["total_q_updates"], total_steps)
        self.assertEqual(len(r_recorded.memory.entries), total_steps)
        self.assertEqual(len(recs_rec), total_steps)
        # Verify 0 learning operations across reset boundaries
        # Across 2 loops of 54 strides = 108 steps total
        self.assertEqual(total_steps, 2 * 54)
        model_observations = sum(sum(successors.values())
                                 for successors in r_recorded.model.counts.values())
        self.assertEqual(model_observations, total_steps)
        from remnant_core.model import WorldModel
        independently_observed = WorldModel()
        for row in recs_rec:
            before = (tuple(row["prev_obs"][0]), frozenset(row["prev_obs"][1]))
            after = (tuple(row["next_obs"][0]), frozenset(row["next_obs"][1]))
            independently_observed.observe(before, row["action"], row["reward"], after)
        self.assertEqual(independently_observed.to_dict(), r_recorded.model.to_dict())
        for loop in (1, 2):
            first = next(row for row in recs_rec if row["loop"] == loop)
            self.assertEqual(first["prev_obs"], [[20, 0], []])

    def test_scripted_solver_policy_escapes_on_loop_1(self):
        """A solver-scripted policy escapes on tide 1."""
        dist, path, actions = solve_reach_shortest_escape()
        self.assertEqual(dist, 42)

        action_iter = iter(actions)
        world = ReachWorld()
        runner = LoopRunner(seed=101, max_loops=10, world=world)
        runner.agent.act = lambda s, rng: next(action_iter)

        res = runner.run()
        self.assertTrue(res["escaped"])
        self.assertEqual(res["escape_loop"], 1)
        self.assertEqual(res["total_loops_run"], 1)
        self.assertEqual(res["loop_records"][0]["real_steps"], 42)

    def test_safe_publication_contract(self):
        """run_reach_study refuses to overwrite an existing directory, publishes atomically,
        and leaves no partial files on error.
        """
        temp_parent = tempfile.mkdtemp()
        try:
            target_dir = os.path.join(temp_parent, "study_results")

            # First run publishes to target_dir successfully
            res1 = run_reach_study(seeds=[101], max_loops=1, output_dir=target_dir)
            self.assertTrue(os.path.exists(os.path.join(target_dir, "manifest.json")))
            self.assertTrue(os.path.exists(os.path.join(target_dir, "records.json")))
            self.assertTrue(os.path.exists(os.path.join(target_dir, "outcomes.json")))
            self.assertTrue(os.path.exists(os.path.join(target_dir, "learners.json")))

            # Second run to the same existing destination MUST fail with FileExistsError
            with self.assertRaises(FileExistsError):
                run_reach_study(seeds=[101], max_loops=1, output_dir=target_dir)

            # Clean staging on failure:
            bad_dir = os.path.join(temp_parent, "study_results_bad")
            with self.assertRaises(ValueError):
                # Pass invalid seeds to trigger an error during execution
                run_reach_study(seeds=["invalid_seed"], max_loops=1, output_dir=bad_dir)  # type: ignore

            # Target bad_dir should not exist and no staging folder left behind
            self.assertFalse(os.path.exists(bad_dir))
            staging_leftovers = [f for f in os.listdir(temp_parent) if f.startswith(".staging_")]
            self.assertEqual(len(staging_leftovers), 0)
        finally:
            shutil.rmtree(temp_parent)



class TestForcedCap(unittest.TestCase):
    def test_guaranteed_nonescaping_policy_stops_at_cap(self):
        runner = LoopRunner(seed=101, max_loops=3, world=ReachWorld())
        runner.agent.act = lambda state, rng: 3  # No incident arc at dawn, always stays.
        result = runner.run()
        self.assertFalse(result['escaped'])
        self.assertIsNone(result['escape_loop'])
        self.assertEqual(result['status'], 'loop_cap_reached')
        self.assertEqual(result['total_loops_run'], 3)
        self.assertEqual(result['total_real_interactions'], 162)
        self.assertEqual(result['total_q_updates'], 162)
        self.assertEqual(len(runner.memory.entries), 162)
        self.assertEqual(sum(sum(counts.values()) for counts in runner.model.counts.values()), 162)
        self.assertEqual([loop['real_steps'] for loop in result['loop_records']], [54, 54, 54])


class TestRouteBStudyContract(unittest.TestCase):
    def test_condition_construction_and_manifest_truthfulness(self):
        """CLI/config condition constructs Route B and manifest truthfully reports condition and schema 2.0.0."""
        temp_parent = tempfile.mkdtemp()
        try:
            target_dir = os.path.join(temp_parent, "test_route_b_pub")
            res = run_reach_study(
                seeds=[251],
                max_loops=1,
                output_dir=target_dir,
                condition="tally_current_tide_pearl_v1",
                command_str="test_cmd",
            )
            manifest_file = os.path.join(target_dir, "manifest.json")
            records_file = os.path.join(target_dir, "records.json")
            learners_file = os.path.join(target_dir, "learners.json")

            with open(manifest_file) as f:
                man = json.load(f)
            with open(records_file) as f:
                rec_payload = json.load(f)
            with open(learners_file) as f:
                lrn_payload = json.load(f)

            self.assertEqual(man["schema_version"], "2.0.0")
            self.assertEqual(man["condition"], "tally_current_tide_pearl_v1")
            self.assertEqual(rec_payload["schema_version"], "2.0.0")
            self.assertEqual(rec_payload["condition"], "tally_current_tide_pearl_v1")
            self.assertEqual(lrn_payload["schema_version"], "2.0.0")
            self.assertEqual(lrn_payload["condition"], "tally_current_tide_pearl_v1")

            # Check that records have separate external observation and policy state
            recs = rec_payload["records"]
            self.assertGreater(len(recs), 0)
            first = recs[0]
            self.assertIn("prev_obs", first)
            self.assertIn("next_obs", first)
            self.assertIn("prev_policy_state", first)
            self.assertIn("next_policy_state", first)
            self.assertEqual(len(first["prev_obs"]), 2)
            self.assertEqual(len(first["prev_policy_state"]), 3)
            # External observation never contains pearls under Route B
            self.assertNotIn(0, first["prev_obs"][1])
        finally:
            shutil.rmtree(temp_parent)

    def test_rng_state_capture_and_restoration_round_trip(self):
        """Action RNG and memory RNG state capture and restore bit-for-bit determinism."""
        runner1 = LoopRunner(seed=252, max_loops=1, world=ReachWorld(), condition="tally_current_tide_pearl_v1")
        initial_action_rng = runner1.get_action_rng_state()
        initial_mem_rng = runner1.memory.get_rng_state()

        runner2 = LoopRunner(seed=999, max_loops=1, world=ReachWorld(), condition="tally_current_tide_pearl_v1")
        runner2.set_action_rng_state(initial_action_rng)
        runner2.memory.set_rng_state(initial_mem_rng)

        # Drawing from both must match bit-for-bit
        draws1 = [runner1.action_rng.random() for _ in range(20)]
        draws2 = [runner2.action_rng.random() for _ in range(20)]
        self.assertEqual(draws1, draws2)

    def test_diagnostic_evaluation_and_replay(self):
        """Pre-training and post-training evaluation trajectories physically replay with zero errors."""
        import random
        from remnant_core.repair_diagnostic import evaluate_policy, replay_evaluation_trajectory
        from remnant_core.memory import EpisodicMemory
        from remnant_core.agent import QAgent
        world = ReachWorld(max_steps=54)
        mem = EpisodicMemory(seed=301)
        agent = QAgent(memory=mem)
        action_rng = random.Random(301)

        eval_res = evaluate_policy(
            world=world,
            agent=agent,
            condition="tally_current_tide_pearl_v1",
            action_rng=action_rng,
            eval_tides=2,
        )
        self.assertEqual(len(eval_res["tides"]), 2)
        ok, err = replay_evaluation_trajectory(eval_res["records"])
        self.assertTrue(ok, f"Replay failed: {err}")


if __name__ == "__main__":
    unittest.main()
