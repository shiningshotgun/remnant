"""Unit tests for LoopWorld: topology, reset semantics, and physics."""

import unittest
from remnant_core.config import (
    START_POS,
    GATE_POS,
    EXIT_POS,
    BAIT_POS,
    KEY_POS,
    ITEM_BAIT,
    ITEM_KEY,
    WALLS,
    LOOP_LIMIT_L,
    STEP_COST,
    BAIT_REWARD,
    KEY_REWARD,
    EXIT_REWARD,
    NORTH,
    SOUTH,
    WEST,
    EAST,
)
from remnant_core.worlds.grid import LoopWorld


class TestLoopWorld(unittest.TestCase):
    def setUp(self):
        self.world = LoopWorld(max_steps=LOOP_LIMIT_L)

    def test_initial_state_and_exact_reset(self):
        """Reset returns world to ((4, 0), empty) with zero step count and no side signals."""
        s = self.world.reset()
        self.assertEqual(s, (START_POS, frozenset()))
        self.assertEqual(self.world.step_count, 0)
        self.assertFalse(self.world.escaped)

    def test_wall_collision_leaves_agent_in_place(self):
        """Moving into a wall leaves the agent in place and deducts step cost."""
        # Wall is at (2, 1). Move agent to (2, 0) then try EAST into (2, 1)
        self.world.pos = (2, 0)
        s, r, terminal, truncated = self.world.step(EAST)
        self.assertEqual(s[0], (2, 0))  # Stayed in place
        self.assertAlmostEqual(r, STEP_COST)
        self.assertFalse(terminal)
        self.assertFalse(truncated)

    def test_boundary_collision_leaves_agent_in_place(self):
        """Moving out of bounds leaves the agent in place and deducts step cost."""
        self.world.reset()  # pos=(4, 0)
        s, r, terminal, truncated = self.world.step(SOUTH)  # Off grid
        self.assertEqual(s[0], (4, 0))
        self.assertAlmostEqual(r, STEP_COST)

        s, r, terminal, truncated = self.world.step(WEST)  # Off grid
        self.assertEqual(s[0], (4, 0))
        self.assertAlmostEqual(r, STEP_COST)

    def test_locked_gate_leaves_agent_in_place_without_key(self):
        """Gate at (0, 3) blocks entry if key is not held."""
        self.world.pos = (0, 2)
        self.world.inventory = frozenset()
        s, r, terminal, truncated = self.world.step(EAST)  # Attempt into (0, 3)
        self.assertEqual(s[0], (0, 2))  # Blocked
        self.assertAlmostEqual(r, STEP_COST)
        self.assertNotIn(ITEM_KEY, s[1])

    def test_gate_allows_passage_with_key(self):
        """Gate at (0, 3) allows entry when key is in inventory."""
        self.world.pos = (0, 2)
        self.world.inventory = frozenset({ITEM_KEY})
        s, r, terminal, truncated = self.world.step(EAST)
        self.assertEqual(s[0], GATE_POS)  # Entered (0, 3)
        self.assertAlmostEqual(r, STEP_COST)

    def test_bait_collection_and_single_loop_exhaustion(self):
        """Entering bait cell yields net +0.99 and adds bait; subsequent visits yield step cost."""
        self.world.pos = (1, 0)
        self.world.inventory = frozenset()
        s, r, terminal, truncated = self.world.step(NORTH)  # Into (0, 0)
        self.assertEqual(s[0], BAIT_POS)
        self.assertIn(ITEM_BAIT, s[1])
        self.assertAlmostEqual(r, STEP_COST + BAIT_REWARD)  # -0.01 + 1.0 = +0.99

        # Step south and back north
        self.world.step(SOUTH)
        s2, r2, _, _ = self.world.step(NORTH)
        self.assertEqual(s2[0], BAIT_POS)
        self.assertAlmostEqual(r2, STEP_COST)  # Only -0.01

    def test_key_collection(self):
        """Entering key cell yields net +0.09 and adds key to inventory."""
        self.world.pos = (4, 3)
        self.world.inventory = frozenset()
        s, r, terminal, truncated = self.world.step(EAST)  # Into (4, 4)
        self.assertEqual(s[0], KEY_POS)
        self.assertIn(ITEM_KEY, s[1])
        self.assertAlmostEqual(r, STEP_COST + KEY_REWARD)  # -0.01 + 0.1 = +0.09

    def test_exit_portal_terminal_reward(self):
        """Stepping onto exit portal yields net +9.99 and sets terminal=True."""
        self.world.pos = (0, 3)
        self.world.inventory = frozenset({ITEM_KEY})
        s, r, terminal, truncated = self.world.step(EAST)  # Into (0, 4)
        self.assertEqual(s[0], EXIT_POS)
        self.assertAlmostEqual(r, STEP_COST + EXIT_REWARD)  # -0.01 + 10.0 = +9.99
        self.assertTrue(terminal)
        self.assertFalse(truncated)
        self.assertTrue(self.world.escaped)

    def test_truncation_at_loop_limit(self):
        """When step count reaches L=18 without escape, truncated=True, terminal=False."""
        self.world.reset()
        for t in range(LOOP_LIMIT_L - 1):
            s, r, terminal, truncated = self.world.step(NORTH)
            self.assertFalse(terminal)
            self.assertFalse(truncated)
            self.assertEqual(self.world.step_count, t + 1)

        # 18th step
        s, r, terminal, truncated = self.world.step(NORTH)
        self.assertEqual(self.world.step_count, LOOP_LIMIT_L)
        self.assertFalse(terminal)
        self.assertTrue(truncated)

    def test_items_rearm_on_reset(self):
        """Bait and key re-arm when world is reset."""
        # Collect bait
        self.world.pos = (1, 0)
        self.world.step(NORTH)
        self.assertIn(ITEM_BAIT, self.world.inventory)

        # Reset world
        self.world.reset()
        self.assertEqual(self.world.inventory, frozenset())

        # Move back to bait
        self.world.pos = (1, 0)
        s, r, _, _ = self.world.step(NORTH)
        self.assertAlmostEqual(r, STEP_COST + BAIT_REWARD)  # Re-armed!


if __name__ == "__main__":
    unittest.main()
