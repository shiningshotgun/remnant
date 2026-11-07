"""Structural acceptance tests for Remnant using the exact BFS solver and Markov chain."""

import unittest
from remnant_core.config import LOOP_LIMIT_L
from remnant_core.solver import solve_shortest_escape, compute_random_walk_escape_probabilities


class TestSolverAndStructuralAcceptance(unittest.TestCase):
    def test_shortest_escape_with_key_is_12_steps(self):
        """Proves optimal escape path with key available is exactly 12 steps."""
        dist, path, actions = solve_shortest_escape(has_key_allowed=True, visit_bait_first=False)
        self.assertIsNotNone(dist)
        self.assertEqual(dist, 12)
        # Expected optimal sequence: 4 East, 2 North, 2 West, 2 North, 2 East
        self.assertEqual(len(actions), 12)
        self.assertEqual(path[0], (4, 0))
        self.assertEqual(path[4], (4, 4))  # Key acquired
        self.assertEqual(path[11], (0, 3)) # Gate entered
        self.assertEqual(path[12], (0, 4)) # Exit reached

    def test_escape_without_key_is_unreachable(self):
        """Proves that without the key, no path to the exit exists (key is strictly necessary)."""
        dist, path, actions = solve_shortest_escape(has_key_allowed=False, visit_bait_first=False)
        self.assertIsNone(dist)
        self.assertIsNone(path)
        self.assertIsNone(actions)

    def test_escape_visiting_bait_first_is_20_steps(self):
        """Proves any path detouring to the bait requires at least 20 steps, exceeding L=18."""
        dist, path, actions = solve_shortest_escape(has_key_allowed=True, visit_bait_first=True)
        self.assertIsNotNone(dist)
        self.assertEqual(dist, 20)
        self.assertGreater(dist, LOOP_LIMIT_L)
        # Confirms bait is a fatal time trap within loop limit L=18

    def test_random_walk_escape_probability_recomputed(self):
        """Recomputes exact random walk Markov chain escape table and asserts P(18) < 1e-4."""
        probs = compute_random_walk_escape_probabilities(max_steps=LOOP_LIMIT_L)
        
        # Verify step 1 to 11 are zero (shortest path is 12)
        for s in range(1, 12):
            self.assertEqual(probs[s], 0.0)

        # Step 12 is positive but miniscule
        self.assertGreater(probs[12], 0.0)

        # Step 18 cumulative escape probability must be strictly less than 1e-4
        p_18 = probs[18]
        self.assertLess(p_18, 1e-4)

        # Output the exact recomputed table for reporting
        print("\n--- Exact Random Walk Escape Probabilities (Markov Chain) ---")
        for step in sorted(probs.keys()):
            print(f"Step {step:2d}: P(escape) = {probs[step]:.8e} ({probs[step] * 100:.6f}%)")
        print("-------------------------------------------------------------")


if __name__ == "__main__":
    unittest.main()
