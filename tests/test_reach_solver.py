"""Tests for the Reach of Vesper exact solver and Markov chain random walk."""

import unittest
from remnant_core.reach_solver import (
    solve_reach_shortest_escape,
    compute_reach_random_walk_escape_probabilities,
    propagate_random_walk_probabilities,
)
from remnant_core.worlds.reach import ReachWorld, physical_transition, START_POS


class TestReachSolver(unittest.TestCase):
    def test_shortest_escape_is_42_strides(self):
        """Shortest escape with key allowed is exactly 42 strides."""
        dist, path, actions = solve_reach_shortest_escape(has_key_allowed=True, collect_pearls_first=False)
        self.assertEqual(dist, 42)
        self.assertEqual(len(actions), 42)
        self.assertEqual(len(path), 43)

        # Replay the 42 actions in a live ReachWorld and assert escape on stride 42
        world = ReachWorld()
        world.reset()
        for i, a in enumerate(actions):
            obs, r, term, trunc = world.step(a)
            if i < 41:
                self.assertFalse(term)
                self.assertFalse(trunc)
            else:
                self.assertTrue(term)
                self.assertFalse(trunc)
                self.assertTrue(world.escaped)

    def test_escape_impossible_without_key(self):
        """Without the Lamp-Key, no escape path exists."""
        dist, path, actions = solve_reach_shortest_escape(has_key_allowed=False)
        self.assertIsNone(dist)
        self.assertIsNone(path)
        self.assertIsNone(actions)

    def test_pearls_first_escape_is_62_strides(self):
        """Collecting pearls before escape requires at least 62 strides (exceeding L=54).
        Validates the 62-action structural path link-by-link via physical_transition.
        """
        dist, path, actions = solve_reach_shortest_escape(has_key_allowed=True, collect_pearls_first=True)
        self.assertEqual(dist, 62)
        self.assertEqual(len(actions), 62)

        # Validate link by link using side-effect-free physical_transition
        curr_state = (START_POS, False, False)
        for i, a in enumerate(actions):
            next_state, r, term, inv_reason, pearls_picked, key_picked = physical_transition(curr_state, a)
            self.assertEqual(next_state[0], path[i + 1])
            curr_state = next_state

        self.assertTrue(curr_state[1])  # has_key
        self.assertTrue(curr_state[2])  # pearls_collected

    def test_random_walk_escape_probabilities(self):
        """Properties of the random walk probability distribution."""
        probs = compute_reach_random_walk_escape_probabilities(max_steps=54)

        prev_p = 0.0
        for step in range(1, 55):
            p = probs[step]
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
            self.assertGreaterEqual(p, prev_p, f"Non-monotonic CDF at step {step}")
            prev_p = p

        # Exactly zero escape probability before step 42
        for step in range(1, 42):
            self.assertEqual(probs[step], 0.0)

        # Positive probability at step 42
        self.assertGreater(probs[42], 0.0)

    def test_tiny_fixture_random_walk_probabilities_exact_formula(self):
        """Independently enumerable 2-state Markov chain evaluated via production propagate_random_walk_probabilities."""
        # 2 states: 0 (transient), 1 (absorbing).
        # Action count = 4: Action 0 transitions 0 -> 1; actions 1, 2, 3 stay in 0.
        # Transition prob: P(0 -> 1) = 1/4 = 0.25; P(0 -> 0) = 3/4 = 0.75.
        # Exact theoretical cumulative distribution: P(T <= k) = 1 - (0.75)^k
        def transition_fn(s, a):
            if s == 1:
                return 1
            return 1 if a == 0 else 0

        def is_terminal_fn(s):
            return s == 1

        probs = propagate_random_walk_probabilities(
            initial_state=0,
            transition_fn=transition_fn,
            is_terminal_fn=is_terminal_fn,
            max_steps=10,
            action_count=4,
        )

        for k in range(1, 11):
            expected = 1.0 - (0.75 ** k)
            self.assertAlmostEqual(probs[k], expected, places=7, msg=f"Mismatch at step {k}")



class TestProbabilityConservation(unittest.TestCase):
    def test_absorption_already_terminal_and_no_loss_of_mass(self):
        for terminal_states in ({0}, {1}, {0, 1}, set()):
            values = propagate_random_walk_probabilities(
                0, lambda state, action: 1 - state,
                lambda state: state in terminal_states, 4, 4)
            expected = 1.0 if terminal_states else 0.0
            self.assertEqual(values, {step: expected for step in range(1, 5)})


if __name__ == "__main__":
    unittest.main()
