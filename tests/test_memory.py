"""Unit tests for EpisodicMemory: similarity calculation, softmax weighting, and sampling."""

import math
import unittest
from remnant_core.memory import EpisodicMemory


class TestEpisodicMemory(unittest.TestCase):
    def test_similarity_metric_fixture(self):
        """Hand-computed distance: Manhattan on position + symmetric diff on inventory.
        
        s1 = ((4, 0), frozenset())
        s2 = ((2, 1), frozenset({0}))
        dist = |4 - 2| + |0 - 1| + len(empty ^ {0}) = 2 + 1 + 1 = 4.
        sim = 1 / (1 + 4) = 0.2.
        """
        s1 = ((4, 0), frozenset())
        s2 = ((2, 1), frozenset({0}))
        sim = EpisodicMemory._sim(s1, s2)
        self.assertAlmostEqual(sim, 0.2, places=6)

    def test_softmax_value_lookup_fixture(self):
        """Hand-computed softmax weighting with tau=0.1.
        
        Query state: s_q = ((4, 0), empty), action = 1.
        Entry 1: s1 = ((4, 1), empty), action = 1, out = 0.5.
                 dist = |4-4| + |0-1| + 0 = 1 -> sim = 1/(1+1) = 0.5.
                 w1 = exp(0.5 / 0.1) = exp(5).
        Entry 2: s2 = ((2, 1), {0}), action = 1, out = -0.1.
                 dist = 4 -> sim = 0.2.
                 w2 = exp(0.2 / 0.1) = exp(2).
        Expected M = (w1 * 0.5 + w2 * -0.1) / (w1 + w2).
        """
        mem = EpisodicMemory(temperature=0.1)
        s_q = ((4, 0), frozenset())
        s1 = ((4, 1), frozenset())
        s2 = ((2, 1), frozenset({0}))

        mem.append(s1, action=1, outcome=0.5)
        mem.append(s2, action=1, outcome=-0.1)

        w1 = math.exp(0.5 / 0.1)
        w2 = math.exp(0.2 / 0.1)
        expected_val = (w1 * 0.5 + w2 * -0.1) / (w1 + w2)

        actual_val = mem.value(s_q, action=1)
        self.assertIsNotNone(actual_val)
        self.assertAlmostEqual(actual_val, expected_val, places=6)

    def test_entries_store_immediate_reward_without_origin(self):
        """Memory entries must be exactly 3-tuples (state, action, outcome)."""
        mem = EpisodicMemory()
        s0 = ((4, 0), frozenset())
        mem.append(s0, action=2, outcome=0.99)

        self.assertEqual(len(mem.entries), 1)
        entry = mem.entries[0]
        self.assertEqual(len(entry), 3)
        self.assertEqual(entry[0], s0)
        self.assertEqual(entry[1], 2)
        self.assertAlmostEqual(entry[2], 0.99)

    def test_reservoir_sampling_over_400_entries(self):
        """When entries exceed lookup_sample=400, a random subsample is used."""
        mem = EpisodicMemory(lookup_sample=400, seed=42)
        s_query = ((0, 0), frozenset())

        # Populate 500 entries
        for i in range(500):
            pos = (i % 5, (i * 2) % 5)
            inv = frozenset({0}) if i % 2 == 0 else frozenset()
            mem.append((pos, inv), action=0, outcome=float(i % 10))

        self.assertEqual(len(mem.entries), 500)
        # Verify lookup operates successfully over subsample
        val = mem.value(s_query, action=0)
        self.assertIsNotNone(val)
        self.assertIsInstance(val, float)

    def test_serialization_round_trip(self):
        """EpisodicMemory serializes and restores entries and RNG state."""
        mem = EpisodicMemory(temperature=0.05, lookup_sample=200, seed=7)
        s0 = ((4, 0), frozenset({1}))
        mem.append(s0, 3, 0.09)

        data = mem.to_dict()
        restored = EpisodicMemory.from_dict(data)

        self.assertEqual(restored.temperature, 0.05)
        self.assertEqual(restored.lookup_sample, 200)
        self.assertEqual(len(restored.entries), 1)
        self.assertEqual(restored.entries[0], (s0, 3, 0.09))

    def test_memory_similarity_marker_mismatch(self):
        """Memory similarity treats tally marker mismatch as different policy state."""
        s_base = ((1, 1), frozenset(), False)
        s_same = ((1, 1), frozenset(), False)
        s_diff_marker = ((1, 1), frozenset(), True)
        sim_same = EpisodicMemory._sim(s_base, s_same)
        sim_diff = EpisodicMemory._sim(s_base, s_diff_marker)
        self.assertGreater(sim_same, sim_diff)

    def test_memory_policy_state_with_marker_serialization_round_trip(self):
        """EpisodicMemory serializes and restores policy states with tally marker."""
        mem = EpisodicMemory(seed=42)
        s_marked = ((2, 3), frozenset({1}), True)
        mem.append(s_marked, 2, 1.5)

        data = mem.to_dict()
        restored = EpisodicMemory.from_dict(data)

        self.assertEqual(len(restored.entries), 1)
        self.assertEqual(restored.entries[0], (s_marked, 2, 1.5))


if __name__ == "__main__":
    unittest.main()
