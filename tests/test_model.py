"""Unit tests for WorldModel: transition counting, empirical rewards, and sampling."""

import random
import unittest
from remnant_core.model import WorldModel


class TestWorldModel(unittest.TestCase):
    def setUp(self):
        self.s0 = ((4, 0), frozenset())
        self.s1 = ((4, 1), frozenset())
        self.s2 = ((3, 0), frozenset())

    def test_counts_and_reward_accumulation(self):
        """WorldModel accurately counts transitions and accumulates rewards."""
        model = WorldModel()
        model.observe(self.s0, 3, -0.01, self.s1)
        model.observe(self.s0, 3, -0.01, self.s1)
        model.observe(self.s0, 3, 0.09, self.s1)

        self.assertEqual(model.counts[(self.s0, 3)][self.s1], 3)
        self.assertAlmostEqual(model.reward_sum[(self.s0, 3)], 0.07, places=6)

    def test_sample_next_and_predict(self):
        """sample_next samples proportionally; predict returns argmax and mean reward."""
        model = WorldModel()
        # 3 observations of s1 with reward 1.0, 1 observation of s2 with reward 2.0
        for _ in range(3):
            model.observe(self.s0, 0, 1.0, self.s1)
        model.observe(self.s0, 0, 2.0, self.s2)

        # Predict argmax
        pred_sn, pred_r = model.predict(self.s0, 0)
        self.assertEqual(pred_sn, self.s1)
        # Expected reward: (3*1.0 + 2.0) / 4 = 1.25
        self.assertAlmostEqual(pred_r, 1.25, places=6)

        # Unseen transition returns None, None
        unseen_sn, unseen_r = model.predict(self.s0, 2)
        self.assertIsNone(unseen_sn)
        self.assertIsNone(unseen_r)

        # sample_next returns valid state and mean reward
        rng = random.Random(42)
        sample_sn, sample_r = model.sample_next(self.s0, 0, rng)
        self.assertIn(sample_sn, [self.s1, self.s2])
        self.assertAlmostEqual(sample_r, 1.25, places=6)

    def test_serialization_round_trip(self):
        """WorldModel serializes and deserializes accurately."""
        model = WorldModel()
        model.observe(self.s0, 1, 0.5, self.s1)
        model.observe(self.s0, 1, 0.5, self.s2)

        data = model.to_dict()
        restored = WorldModel.from_dict(data)

        self.assertEqual(restored.counts[(self.s0, 1)][self.s1], 1)
        self.assertEqual(restored.counts[(self.s0, 1)][self.s2], 1)
        self.assertAlmostEqual(restored.reward_sum[(self.s0, 1)], 1.0)

    def test_world_model_keys_contain_external_observations_only(self):
        """WorldModel transitions and keys must contain external observations only (pos, inv)."""
        model = WorldModel()
        model.observe(self.s0, 0, 0.1, self.s1)
        for (s, a), succ in model.counts.items():
            self.assertEqual(len(s), 2, "WorldModel state key must be 2-tuple (pos, inv)")
            self.assertIsInstance(s[0], tuple)
            self.assertIsInstance(s[1], frozenset)
            for sn in succ:
                self.assertEqual(len(sn), 2, "Successor state must be 2-tuple (pos, inv)")


if __name__ == "__main__":
    unittest.main()
