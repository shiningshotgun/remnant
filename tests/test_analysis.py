"""Tests for the Reach of Vesper analysis metrics and figure generation."""

import json
import os
import shutil
import tempfile
import unittest

from remnant_core.analysis import compute_study_outcomes, validate_records_integrity
from remnant_core.figures import render_all_figures


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        # Synthetic fixture:
        # Seed 1: escapes at loop 2 (1 stride on loop 2)
        # Seed 2: runs 3 loops, never escapes (censored at horizon 3)
        self.synthetic_payload = {
            "schema_version": "1.0.0",
            "protocol": {"seeds": [1, 2], "horizon_k_max": 3, "loop_limit": 2, "fixture_kind": "structural_miniature"},
            "records": [
                # Seed 1, loop 1: 2 strides (Wrecksmoor -> Needle Cut, picks pearls=False)
                {
                    "seed": 1, "loop": 1, "stride": 1,
                    "prev_obs": [[20, 0], []], "next_obs": [[20, 1], []],
                    "prev_node": [20, 0], "next_node": [20, 1], "action": 0, "reward": -0.01,
                    "pearls_picked": False, "key_picked": False, "terminal": False, "truncated": False,
                    "npc_states": {"Gleaner": {"activity": "tending the chapel"}},
                },
                {
                    "seed": 1, "loop": 1, "stride": 2,
                    "prev_obs": [[20, 1], []], "next_obs": [[20, 2], []],
                    "prev_node": [20, 1], "next_node": [20, 2], "action": 0, "reward": -0.01,
                    "pearls_picked": False, "key_picked": False, "terminal": False, "truncated": True,
                    "npc_states": {"Gleaner": {"activity": "tending the chapel"}},
                },
                # Seed 1, loop 2: 1 stride (escapes)
                {
                    "seed": 1, "loop": 2, "stride": 1,
                    "prev_obs": [[1, 14], [1]], "next_obs": [[0, 14], [1]],
                    "prev_node": [1, 14], "next_node": [0, 14], "action": 0, "reward": 9.99,
                    "pearls_picked": False, "key_picked": False, "terminal": True, "truncated": False,
                    "npc_states": {"Dov": {"activity": "inspecting the mechanism"}},
                },
                # Seed 2, loop 1: 1 stride (picks pearls at chapel)
                {
                    "seed": 2, "loop": 1, "stride": 1,
                    "prev_obs": [[11, 4], []], "next_obs": [[10, 4], []],
                    "prev_node": [11, 4], "next_node": [10, 4], "action": 0, "reward": 0.99,
                    "pearls_picked": True, "key_picked": False, "terminal": False, "truncated": True,
                    "npc_states": {"Gleaner": {"activity": "greeting Wick warmly"}},
                },
                # Seed 2, loop 2: 1 stride (no pearls, stays in place at Wrecksmoor)
                {
                    "seed": 2, "loop": 2, "stride": 1,
                    "prev_obs": [[20, 0], []], "next_obs": [[20, 0], []],
                    "prev_node": [20, 0], "next_node": [20, 0], "action": 3, "reward": -0.01,
                    "pearls_picked": False, "key_picked": False, "terminal": False, "truncated": True,
                    "npc_states": {"Gleaner": {"activity": "tending the chapel"}},
                },
                # Seed 2, loop 3: 1 stride (departing to Needle Cut)
                {
                    "seed": 2, "loop": 3, "stride": 1,
                    "prev_obs": [[20, 0], []], "next_obs": [[20, 1], []],
                    "prev_node": [20, 0], "next_node": [20, 1], "action": 0, "reward": -0.01,
                    "pearls_picked": False, "key_picked": False, "terminal": False, "truncated": True,
                    "npc_states": {"Gleaner": {"activity": "tending the chapel"}},
                },
            ],
        }

        # Explicit miniature episode contract: all censored loops have two rows.
        import copy
        padded = []
        for row in self.synthetic_payload["records"]:
            if row["truncated"] and row["stride"] == 1:
                row["truncated"] = False
                padded.append(row)
                final = copy.deepcopy(row)
                final.update(stride=2, truncated=True, pearls_picked=False,
                             key_picked=False, reward=-0.01, action=3,
                             prev_obs=copy.deepcopy(row["next_obs"]),
                             prev_node=copy.deepcopy(row["next_node"]))
                padded.append(final)
            else:
                padded.append(row)
        self.synthetic_payload["records"] = padded

    def test_synthetic_metrics_computation(self):
        """Verify exact string keys, denominators, escape rate, censoring, and pearls rate on synthetic fixture."""
        validate_records_integrity(self.synthetic_payload["records"], self.synthetic_payload["protocol"])
        outcomes = compute_study_outcomes(self.synthetic_payload)
        summary = outcomes["summary"]
        self.assertEqual(summary["total_seeds"], 2)
        self.assertEqual(summary["escaped_count"], 1)
        self.assertEqual(summary["censored_count"], 1)
        self.assertAlmostEqual(summary["escape_rate"], 0.5)
        self.assertEqual(summary["escape_distribution"], {"1": 2})

        # Loop 1: both seeds active (n=2)
        p1 = outcomes["per_loop"]["pearls_taken"]["1"]
        self.assertEqual(p1["active_seeds"], 2)
        self.assertEqual(p1["pearls_taken_count"], 1)
        self.assertAlmostEqual(p1["pearls_taken_rate"], 0.5)

        # Loop 2: both seeds active (n=2), 0 pearls
        p2 = outcomes["per_loop"]["pearls_taken"]["2"]
        self.assertEqual(p2["active_seeds"], 2)
        self.assertEqual(p2["pearls_taken_count"], 0)

        # Loop 3: Seed 1 escaped in loop 2! Active seeds must be 1 (Seed 2 only)
        p3 = outcomes["per_loop"]["pearls_taken"]["3"]
        self.assertEqual(p3["active_seeds"], 1)

        # Verify JSON round-trip invariance: outcomes == json.loads(json.dumps(outcomes))
        outcomes_json = json.loads(json.dumps(outcomes))
        self.assertEqual(outcomes, outcomes_json)

    def test_validation_rejects_corrupted_payloads(self):
        """validate_records_integrity rejects invalid seeds, malformed loops, early endings, and non-finite values."""
        import copy

        # 1. Boolean seed rejected
        bad_bool_seed = copy.deepcopy(self.synthetic_payload)
        bad_bool_seed["protocol"]["seeds"] = [True, 2]
        bad_bool_seed["records"][0]["seed"] = True
        with self.assertRaises(ValueError):
            validate_records_integrity(bad_bool_seed["records"], bad_bool_seed["protocol"])

        # 2. Duplicate seed rejected
        bad_dup_seed = copy.deepcopy(self.synthetic_payload)
        bad_dup_seed["protocol"]["seeds"] = [1, 1]
        with self.assertRaises(ValueError):
            validate_records_integrity(bad_dup_seed["records"], bad_dup_seed["protocol"])

        # 3. Foreign seed in records rejected
        bad_foreign = copy.deepcopy(self.synthetic_payload)
        bad_foreign["records"][0]["seed"] = 999
        with self.assertRaises(ValueError):
            validate_records_integrity(bad_foreign["records"], bad_foreign["protocol"])

        # 4. Non-finite reward rejected
        bad_nan_reward = copy.deepcopy(self.synthetic_payload)
        bad_nan_reward["records"][0]["reward"] = float("nan")
        with self.assertRaises(ValueError):
            validate_records_integrity(bad_nan_reward["records"], bad_nan_reward["protocol"])

        # 5. Premature terminal (stride ends with neither terminal nor truncated)
        bad_premature = copy.deepcopy(self.synthetic_payload)
        bad_premature["records"][1]["truncated"] = False
        with self.assertRaises(ValueError):
            validate_records_integrity(bad_premature["records"], bad_premature["protocol"])

        # 6. Censored run missing loops before horizon
        bad_incomplete = copy.deepcopy(self.synthetic_payload)
        # Drop loop 3 for seed 2
        bad_incomplete["records"] = [r for r in bad_incomplete["records"] if not (r["seed"] == 2 and r["loop"] == 3)]
        with self.assertRaises(ValueError):
            validate_records_integrity(bad_incomplete["records"], bad_incomplete["protocol"])

        # 7. Obs node mismatch
        bad_obs = copy.deepcopy(self.synthetic_payload)
        bad_obs["records"][0]["prev_obs"] = [[99, 99], []]
        with self.assertRaises(ValueError):
            validate_records_integrity(bad_obs["records"], bad_obs["protocol"])

    def test_figures_generation_and_image_decoding(self):
        """Generate all figures from synthetic data, assert files exist, are non-empty, and decode valid dimensions."""
        try:
            import matplotlib
            from PIL import Image
        except ImportError:
            self.skipTest("matplotlib or pillow not installed in system interpreter")

        temp_dir = tempfile.mkdtemp()
        try:
            records_file = os.path.join(temp_dir, "records.json")
            outcomes_file = os.path.join(temp_dir, "outcomes.json")
            with open(records_file, "w") as fp:
                json.dump(self.synthetic_payload, fp)
            outcomes = compute_study_outcomes(self.synthetic_payload)
            with open(outcomes_file, "w") as fp:
                json.dump(outcomes, fp)

            fig_dir = os.path.join(temp_dir, "figures")
            generated = render_all_figures(records_path=records_file, outcomes_path=outcomes_file, output_dir=fig_dir)

            expected_names = [
                "reach-map", "visit-frequency", "stride-economy",
                "escape-timing", "pearls-taken", "route-choices", "character-activity"
            ]
            for name in expected_names:
                self.assertIn(name, generated)
                path = generated[name]
                self.assertTrue(os.path.exists(path), f"Figure {path} missing")
                self.assertGreater(os.path.getsize(path), 1000, f"Figure {path} is empty or too small")

                # Decode image and check non-zero dimensions
                with Image.open(path) as img:
                    width, height = img.size
                    self.assertGreater(width, 100)
                    self.assertGreater(height, 100)
        finally:
            shutil.rmtree(temp_dir)



class TestStrictEvidence(unittest.TestCase):
    def setUp(self):
        from remnant_core.study import run_reach_seed
        result, records, _ = run_reach_seed(101, 1)
        self.payload = {'schema_version': '1.0.0',
                        'protocol': {'seeds': [101], 'horizon_k_max': 1, 'loop_limit': 54},
                        'total_records': len(records), 'records': records}
        self.summary = result

    def test_rejects_structural_corruptions(self):
        import copy
        changes = {
            'schema': lambda p: p.update(schema_version='999'),
            'missing_limit': lambda p: p['protocol'].pop('loop_limit'),
            'boolean_limit': lambda p: p['protocol'].update(loop_limit=True),
            'duplicate_seed': lambda p: p['protocol'].update(seeds=[101, 101]),
            'missing_seed': lambda p: p['protocol'].update(seeds=[101, 102]),
            'foreign_seed': lambda p: p['records'][0].update(seed=102),
            'boolean_reward': lambda p: p['records'][0].update(reward=True),
            'infinite_reward': lambda p: p['records'][0].update(reward=float('inf')),
            'boolean_pickup': lambda p: p['records'][0].update(pearls_picked=1),
            'count_mismatch': lambda p: p.update(total_records=53),
            'simultaneous_flags': lambda p: p['records'][-1].update(terminal=True),
            'early_terminal': lambda p: p['records'][0].update(terminal=True),
            'missing_cast': lambda p: p['records'][0]['npc_states'].pop('Dov'),
            'missing_snapshot_field': lambda p: p['records'][0]['npc_states']['Dov'].pop('local_state'),
            'invalid_inventory': lambda p: p['records'][0]['prev_obs'][1].append(9),
            'boolean_coordinate': lambda p: p['records'][0].update(prev_node=[True, 0], prev_obs=[[True, 0], []]),
            'duplicate_row': lambda p: (p['records'].append(copy.deepcopy(p['records'][-1])), p.update(total_records=55)),
            'partial_loop': lambda p: (p.update(records=p['records'][:1], total_records=1), p['records'][0].update(truncated=True)),
            'overlong_loop': lambda p: p['protocol'].update(loop_limit=53),
            'over_horizon': lambda p: p['records'][-1].update(loop=2),
            'observation_gap': lambda p: p['records'][1].update(prev_node=[99, 99], prev_obs=[[99, 99], []]),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                payload = copy.deepcopy(self.payload)
                change(payload)
                with self.assertRaises(ValueError):
                    compute_study_outcomes(payload)

    def test_every_comparable_summary_field_is_checked(self):
        import copy
        compute_study_outcomes(self.payload, [self.summary])
        changes = {'escaped': True, 'escape_loop': 1, 'status': 'escaped', 'max_loops': 2,
                   'total_loops_run': 2, 'total_real_interactions': 53, 'total_q_updates': 53,
                   'q_table_size': 0, 'memory_entries': 0, 'model_size': {}}
        for field, value in changes.items():
            summary = copy.deepcopy(self.summary)
            summary[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                compute_study_outcomes(self.payload, [summary])
        for field in ('cumulative_reward', 'q_table_size', 'memory_size', 'real_steps', 'escaped', 'loop'):
            summary = copy.deepcopy(self.summary)
            summary['loop_records'][0][field] = 'corrupted'
            with self.subTest(loop_field=field), self.assertRaises(ValueError):
                compute_study_outcomes(self.payload, [summary])

    def test_known_answer_forks_pickups_nulls_activity_and_overlap(self):
        # An explicitly miniature structural fixture, not a production Reach replay.
        # Each seed has an eight-row loop. Repeated junction departures count;
        # stays never count as departures; a pearl flag counts once per loop.
        def rows(seed, positions, pickup_indices):
            result = []
            for index, (before, after) in enumerate(zip(positions, positions[1:]), 1):
                result.append({'seed': seed, 'loop': 1, 'stride': index,
                    'prev_obs': [list(before), []], 'next_obs': [list(after), []],
                    'prev_node': list(before), 'next_node': list(after),
                    'action': 0, 'reward': -0.01, 'pearls_picked': index in pickup_indices,
                    'key_picked': False, 'terminal': False, 'truncated': index == 8,
                    'npc_states': {'Gleaner': {'activity': 'company' if index <= 2 else 'tending'}}})
            return result
        wm, needle = (20, 0), (20, 10)
        records = rows(1, [wm, (20, 1), wm, (19, 0), wm, wm, (20, 1), needle, (21, 10)], {3})
        records += rows(2, [needle, (20, 9), needle, (19, 10), needle, (21, 10), needle, needle, needle], set())
        payload = {'schema_version': '1.0.0', 'protocol': {'seeds': [1, 2], 'horizon_k_max': 1, 'loop_limit': 8, 'fixture_kind': 'structural_miniature'}, 'records': records}
        outcome = compute_study_outcomes(payload)
        per = outcome['per_loop']
        self.assertEqual(per['route_choices_wrecksmoor']['1'], {
            'loop': 1, 'active_seeds': 2, 'departures_to_needle_cut': 2,
            'departures_to_chapel': 1, 'total_departures': 3,
            'departures_per_active_seed': 1.5, 'share_to_needle_cut': 0.6667, 'share_to_chapel': 0.3333})
        self.assertEqual(per['route_choices_needle_cut']['1'], {
            'loop': 1, 'active_seeds': 2, 'departures_to_net_loft': 2,
            'departures_to_wrecksmoor': 1, 'departures_to_chapel': 1, 'total_departures': 4,
            'departures_per_active_seed': 2.0, 'share_to_net_loft': 0.5,
            'share_to_wrecksmoor': 0.25, 'share_to_chapel': 0.25})
        self.assertEqual(per['pearls_taken']['1']['pearls_taken_count'], 1)
        self.assertEqual(per['pearls_taken']['1']['pearls_taken_rate'], 0.5)
        self.assertEqual(per['character_telemetry']['1']['characters']['Gleaner'],
                         {'company': {'count': 4, 'fraction': 0.25}, 'tending': {'count': 12, 'fraction': 0.75}})
        self.assertEqual(outcome['visitation']['total_strides_early'], 16)
        self.assertEqual(outcome['visitation']['total_strides_late'], 16)
        self.assertEqual(outcome['visitation']['early_node_visits'], outcome['visitation']['late_node_visits'])
        # No departures: both shares null, never a numeric zero implied choice.
        still = rows(1, [wm] * 9, set())
        payload['protocol']['seeds'] = [1]
        payload['records'] = still
        zero = compute_study_outcomes(payload)
        self.assertEqual(zero['summary']['escaped_count'], 0)
        self.assertIsNone(zero['per_loop']['route_choices_wrecksmoor']['1']['share_to_chapel'])
        still[-1].update(terminal=True, truncated=False)
        all_escape = compute_study_outcomes(payload)
        self.assertEqual(all_escape['summary']['escape_rate'], 1.0)
        self.assertEqual(all_escape['summary']['censored_count'], 0)

    def test_schema_2_policy_state_validation(self):
        """Schema 2.0.0 validates prev_policy_state and next_policy_state while keeping external observations valid."""
        import copy
        from remnant_core.analysis import validate_records_integrity
        rec = {
            "seed": 1,
            "loop": 1,
            "stride": 1,
            "prev_obs": [[4, 0], []],
            "next_obs": [[4, 1], []],
            "prev_policy_state": [[4, 0], [], False],
            "next_policy_state": [[4, 1], [], False],
            "prev_node": [4, 0],
            "next_node": [4, 1],
            "action": 0,
            "reward": -0.01,
            "pearls_picked": False,
            "key_picked": False,
            "terminal": False,
            "truncated": True,
            "npc_states": {"dummy": "state"},
        }
        protocol = {
            "seeds": [1],
            "horizon_k_max": 1,
            "loop_limit": 1,
            "fixture_kind": "structural_miniature",
            "schema_version": "2.0.0",
        }
        # Valid schema 2 record passes
        validate_records_integrity([rec], protocol)

        # Missing policy state fails under schema 2.0.0
        rec_bad = copy.deepcopy(rec)
        del rec_bad["prev_policy_state"]
        with self.assertRaises(ValueError):
            validate_records_integrity([rec_bad], protocol)

        # Invalid tally marker type fails
        rec_bad_marker = copy.deepcopy(rec)
        rec_bad_marker["prev_policy_state"][2] = "invalid_bool"
        with self.assertRaises(ValueError):
            validate_records_integrity([rec_bad_marker], protocol)


if __name__ == "__main__":
    unittest.main()
