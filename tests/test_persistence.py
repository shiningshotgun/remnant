"""Unit tests verifying component persistence across time-loop resets."""

import random
import unittest
from remnant_core.agent import QAgent
from remnant_core.config import START_POS, LOOP_LIMIT_L
from remnant_core.memory import EpisodicMemory
from remnant_core.model import WorldModel
from remnant_core.worlds.grid import LoopWorld


class TestComponentPersistenceAcrossResets(unittest.TestCase):
    def test_persistence_across_multiple_loops(self):
        """World alone resets to s0 and re-arms; Q-table, memory, and world model persist."""
        world = LoopWorld(max_steps=LOOP_LIMIT_L)
        mem = EpisodicMemory()
        agent = QAgent(n_actions=4, lr=0.3, gamma=0.95, eps=0.2, memory=mem)
        model = WorldModel()
        rng = random.Random(101)

        total_steps_all_loops = 0

        for loop_idx in range(1, 4):
            # World resets physically
            s = world.reset()
            self.assertEqual(s, (START_POS, frozenset()))
            self.assertEqual(world.step_count, 0)
            self.assertFalse(world.escaped)

            loop_step = 0
            while True:
                a = agent.act(s, rng)
                s_next, r, terminal, truncated = world.step(a)

                # Real environmental transition updates agent and model
                agent.memory.append(s, a, r)
                agent.learn(s, a, r, s_next, terminal=terminal)
                model.observe(s, a, r, s_next)

                total_steps_all_loops += 1
                loop_step += 1
                s = s_next

                if terminal or truncated:
                    break

            # At the end of the loop, verify persistence
            self.assertEqual(len(agent.memory.entries), total_steps_all_loops)
            self.assertGreater(len(agent.q), 0)
            self.assertGreater(len(model.counts), 0)

        # After 3 loops of 18 steps each (total 54 steps)
        self.assertEqual(total_steps_all_loops, 3 * LOOP_LIMIT_L)
        self.assertEqual(len(agent.memory.entries), 54)
        # World model and Q-table have accumulated experience across all loops
        self.assertGreater(len(agent.q), 5)
        self.assertGreater(len(model.counts), 5)


if __name__ == "__main__":
    unittest.main()
