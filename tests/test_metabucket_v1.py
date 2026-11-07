"""Unit tests for the corrected_metabucket_v1 experiment arm."""

import unittest

from remnant_core.runner import LoopRunner
from remnant_core.worlds.reach import ReachWorld


class TestMetabucketV1(unittest.TestCase):
    def test_initial_meta_state(self):
        """Fresh runner: nothing visited, no escape -> bucket 0, escape flag False."""
        r = LoopRunner(seed=1, world=ReachWorld(), condition="corrected_metabucket_v1")
        self.assertEqual(r._meta_state(), (0, False))

    def test_bucket_reflects_frontier_count(self):
        """A visited position with untried actions raises the bucket; fully tried does not."""
        r = LoopRunner(seed=1, world=ReachWorld(), condition="corrected_metabucket_v1")
        # Six visited positions with all actions untried -> bucket 2 (6-15).
        for i in range(6):
            r.node_novelty.mark((90, i))
        b, esc = r._meta_state()
        self.assertEqual((b, esc), (2, False))
        # Fill every action for those positions -> frontier empty -> bucket 0.
        for i in range(6):
            for a in range(4):
                for items in (frozenset(), frozenset({1})):
                    r.model.observe(((90, i), items), a, 0.0, ((90, i), items))
        b, esc = r._meta_state()
        self.assertEqual((b, esc), (0, False))

    def test_construction_smoke(self):
        """Both arms run two tides and record mode, map size and stride provenance."""
        for cond in ("corrected_metabucket_v1", "corrected_meta_random_v1"):
            r = LoopRunner(seed=1, world=ReachWorld(), condition=cond, max_loops=2)
            r.run()
            self.assertEqual(len(r.mode_log), 2)
            for e in r.mode_log:
                self.assertIn("mode", e)
                self.assertIn("map_edges", e)


if __name__ == "__main__":
    unittest.main()
