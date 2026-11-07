"""Unit tests for the corrected_metabucket_td_v1 experiment arm."""

import ast
import unittest

from remnant_core.runner import LoopRunner
from remnant_core.worlds.reach import ReachWorld


def _replay_bound_check(mode_log, alpha=0.3, gamma=0.95, tol=1.5e-3):
    """Per-step replay of the one-step TD update against the logged meta-Q tables.

    Each step recomputes from the PREVIOUS entry's logged table (rounded to 3
    decimals), so rounding cannot accumulate across steps. Error budget per step:
    ret is logged to 4 decimals (<=5e-5); old and max-next are logged to 3
    decimals (<=5e-4 each); delta error <= 5e-5 + 0.95*5e-4 + 5e-4 = 1.025e-3;
    the update scales it by alpha (0.3) and adds old's 5e-4 -> <=8.1e-4; the new
    value's own logging adds <=5e-4. Total <= 1.31e-3 < tol. Key sets must match
    exactly. Logged keys are str((meta_state, mode)); meta_state is itself a
    tuple, so ast.literal_eval(k) yields ((bucket, flag), mode).
    The comparison is against the unrounded recon: the 1.31e-3 budget already
    includes the new value's own 4- vs 3-decimal logging rounding; rounding
    recon again could add up to 5e-4 more. Unchanged entries are copied
    verbatim by the runner, so they must equal the previously logged values
    exactly.
    """
    prev = {}
    for e in mode_log:
        ms, mode = e["meta_state"], e["mode"]
        msn, ret = e["meta_state_next"], e["return"]
        old = prev.get((ms, mode), 0.0)
        max_next = max(prev.get((msn, m), 0.0) for m in range(3))
        recon = old + alpha * (ret + gamma * max_next - old)
        logged = dict(e["metaq"])
        expected_keys = set(prev) | {(ms, mode)}
        if set(logged) != {str(k) for k in expected_keys}:
            return False, ("key set mismatch", set(logged) ^ {str(k) for k in expected_keys})
        for k, v in prev.items():
            if k != (ms, mode) and logged[str(k)] != v:
                return False, ("unchanged entry drifted", k, v, logged[str(k)])
        if abs(logged[str((ms, mode))] - recon) > tol:
            return False, ("value mismatch", (ms, mode), logged[str((ms, mode))], recon)
        prev = {ast.literal_eval(k): v for k, v in logged.items()}
    return True, None


class TestMetabucketTdV1(unittest.TestCase):
    def test_initial_meta_state(self):
        """Fresh runner: bucket state routing - bucket 0, escape flag False."""
        r = LoopRunner(seed=1, world=ReachWorld(), condition="corrected_metabucket_td_v1")
        self.assertEqual(r._meta_state(), (0, False))

    def test_bucket_reflects_frontier_count(self):
        """A visited position with untried actions raises the bucket; fully tried does not."""
        r = LoopRunner(seed=1, world=ReachWorld(), condition="corrected_metabucket_td_v1")
        for i in range(6):
            r.node_novelty.mark((90, i))
        b, esc = r._meta_state()
        self.assertEqual((b, esc), (2, False))
        for i in range(6):
            for a in range(4):
                for items in (frozenset(), frozenset({1})):
                    r.model.observe(((90, i), items), a, 0.0, ((90, i), items))
        b, esc = r._meta_state()
        self.assertEqual((b, esc), (0, False))

    def test_trace_absence(self):
        """The TD arm never creates eligibility state or logs trace fields."""
        r = LoopRunner(seed=1, world=ReachWorld(), condition="corrected_metabucket_td_v1",
                       max_loops=2)
        r.run()
        self.assertEqual(r.meta_e, {})
        for e in r.mode_log:
            self.assertNotIn("meta_e_max", e)
            self.assertIn("metaq", e)

    def test_old_arm_still_traces(self):
        """corrected_metabucket_v1 keeps its Peng trace path after the tuple edits."""
        r = LoopRunner(seed=1, world=ReachWorld(), condition="corrected_metabucket_v1",
                       max_loops=2)
        r.run()
        self.assertTrue(any("meta_e_max" in e for e in r.mode_log))

    def test_one_step_td_recursion_replay(self):
        """Logged meta-Q tables match the one-step TD recursion, replayed per step."""
        r = LoopRunner(seed=1, world=ReachWorld(), condition="corrected_metabucket_td_v1",
                       max_loops=3)
        r.run()
        ok, info = _replay_bound_check(r.mode_log)
        self.assertTrue(ok, info)


if __name__ == "__main__":
    unittest.main()
