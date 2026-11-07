"""Unit tests for QAgent: Bellman update fixtures, terminal flag, truncation, and blending."""

import random
import unittest
from remnant_core.agent import QAgent
from remnant_core.memory import EpisodicMemory

State = tuple


class TestQAgent(unittest.TestCase):
    def setUp(self):
        self.s0 = ((4, 0), frozenset())
        self.s1 = ((4, 1), frozenset())
        self.s_exit = ((0, 4), frozenset({1}))

    def test_non_terminal_q_update_fixture(self):
        """Hand-computed Bellman update on non-terminal transition.
        
        Q(s, a) = 0.0, r = -0.01, gamma = 0.95, alpha = 0.3.
        Suppose max_a' V(s_next, a') = 2.0.
        target = -0.01 + 0.95 * 2.0 = 1.89.
        new_q = 0.0 + 0.3 * (1.89 - 0.0) = 0.567.
        """
        agent = QAgent(n_actions=4, lr=0.3, gamma=0.95, eps=0.0)
        # Seed s1 with Q(s1, 0) = 2.0 so next_max = 2.0
        agent.q[(self.s1, 0)] = 2.0

        agent.learn(self.s0, 3, -0.01, self.s1, terminal=False)
        self.assertAlmostEqual(agent.q[(self.s0, 3)], 0.567, places=6)

    def test_terminal_q_update_fixture(self):
        """Hand-computed Bellman update on terminal escape transition (target = r).
        
        Q(s, a) = 0.5, r = 9.99, alpha = 0.3, terminal = True.
        target = 9.99 (future value is 0; no gamma * next_max bootstrap).
        new_q = 0.5 + 0.3 * (9.99 - 0.5) = 0.5 + 0.3 * 9.49 = 3.347.
        """
        agent = QAgent(n_actions=4, lr=0.3, gamma=0.95, eps=0.0)
        agent.q[(self.s0, 3)] = 0.5
        # Even if s_exit has high Q-values in table, terminal=True must ignore them
        agent.q[(self.s_exit, 0)] = 100.0

        agent.learn(self.s0, 3, 9.99, self.s_exit, terminal=True)
        self.assertAlmostEqual(agent.q[(self.s0, 3)], 3.347, places=6)

    def test_truncation_boundary_behavior(self):
        """At step L, the final transition updates using real state s_L with terminal=False;
        no transition is updated across the environmental reset boundary to s_0.
        """
        agent = QAgent(n_actions=4, lr=0.3, gamma=0.95, eps=0.0)
        s_L_minus_1 = ((3, 0), frozenset())
        s_L = ((2, 0), frozenset())
        agent.q[(s_L, 0)] = 1.0

        # Update final in-loop step L-1 -> L
        agent.learn(s_L_minus_1, 0, -0.01, s_L, terminal=False)
        expected_target = -0.01 + 0.95 * 1.0  # 0.94
        expected_q = 0.3 * 0.94               # 0.282
        self.assertAlmostEqual(agent.q[(s_L_minus_1, 0)], expected_q, places=6)

        # Confirm that the agent's Q-table only contains the setup entry and the step L-1 update;
        # no pseudo-transition from s_L across the reset boundary to s0 was added or learned.
        self.assertEqual(len(agent.q), 2)
        self.assertIn((s_L_minus_1, 0), agent.q)
        self.assertIn((s_L, 0), agent.q)

    def test_blended_action_value_computation(self):
        """V(s, a) = (1 - beta) * Q(s, a) + beta * M(s, a) when M(s, a) exists."""
        mem = EpisodicMemory(temperature=0.1)
        agent = QAgent(n_actions=4, memory=mem, memory_blend=0.5)

        # Without memory entry, V = Q
        agent.q[(self.s0, 1)] = 0.4
        self.assertAlmostEqual(agent.qval(self.s0, 1), 0.4)

        # Add exact matching memory entry with outcome 1.0
        mem.append(self.s0, action=1, outcome=1.0)
        # V = 0.5 * 0.4 + 0.5 * 1.0 = 0.2 + 0.5 = 0.7
        self.assertAlmostEqual(agent.qval(self.s0, 1), 0.7)

    def test_act_greedy_and_random(self):
        """Epsilon=0 acts greedily; epsilon=1 acts purely randomly."""
        agent_greedy = QAgent(n_actions=4, eps=0.0)
        agent_greedy.q[(self.s0, 2)] = 5.0
        rng = random.Random(42)
        self.assertEqual(agent_greedy.act(self.s0, rng), 2)

        agent_random = QAgent(n_actions=4, eps=1.0)
        actions = {agent_random.act(self.s0, rng) for _ in range(50)}
        self.assertEqual(actions, {0, 1, 2, 3})

    def test_serialization_round_trip(self):
        """Agent serializes and deserializes accurately with Q-table and memory state."""
        mem = EpisodicMemory()
        mem.append(self.s0, 1, 0.99)
        agent = QAgent(n_actions=4, lr=0.25, gamma=0.9, eps=0.1, memory=mem, memory_blend=0.6)
        agent.q[(self.s0, 1)] = 1.234
        agent.q[(self.s1, 2)] = -0.567

        data = agent.to_dict()
        restored = QAgent.from_dict(data)

        self.assertEqual(restored.lr, 0.25)
        self.assertEqual(restored.gamma, 0.9)
        self.assertEqual(restored.eps, 0.1)
        self.assertEqual(restored.memory_blend, 0.6)
        self.assertAlmostEqual(restored.q[(self.s0, 1)], 1.234)
        self.assertAlmostEqual(restored.q[(self.s1, 2)], -0.567)
        self.assertIsNotNone(restored.memory)
        self.assertEqual(len(restored.memory.entries), 1)

    def test_agent_policy_state_with_tally_marker_round_trip(self):
        """QAgent serializes and restores policy states containing the Route B tally marker."""
        agent = QAgent()
        s_with_marker = ((1, 2), frozenset({1}), True)
        s_without_marker = ((1, 2), frozenset({1}), False)
        agent.q[(s_with_marker, 0)] = 3.14
        agent.q[(s_without_marker, 0)] = -1.5

        data = agent.to_dict()
        restored = QAgent.from_dict(data)

        self.assertIn((s_with_marker, 0), restored.q)
        self.assertIn((s_without_marker, 0), restored.q)
        self.assertAlmostEqual(restored.q[(s_with_marker, 0)], 3.14)
        self.assertAlmostEqual(restored.q[(s_without_marker, 0)], -1.5)


if __name__ == "__main__":
    unittest.main()
