"""Tests for the Reach of Vesper world environment."""

from collections import deque
import unittest
from remnant_core.worlds.reach import (
    ReachWorld,
    JUNCTIONS,
    EDGES,
    GRAPH_CHAINS,
    GRAPH_ARCS,
    ALL_NODES,
    START_POS,
    CHAPEL_POS,
    NET_LOFT_POS,
    HARBOR_LOCK_POS,
    TIDEGATE_POS,
    LOCK_BOUNDARY,
    REACH_LOOP_LIMIT,
    STRIDE_COST,
    PEARLS_REWARD,
    LAMP_KEY_REWARD,
    TIDEGATE_REWARD,
    ITEM_LAMP_KEY,
    ITEM_PEARLS,
    generate_chain,
    physical_transition,
)
from remnant_core.reach_solver import solve_reach_shortest_escape


class TestReachWorld(unittest.TestCase):
    def setUp(self):
        self.world = ReachWorld()

    def test_junction_coordinates_match_table(self):
        """Fixed coordinates of the six named junctions."""
        self.assertEqual(JUNCTIONS["Tidegate"], (0, 14))
        self.assertEqual(JUNCTIONS["Harbor Lock"], (6, 14))
        self.assertEqual(JUNCTIONS["Drowned Chapel"], (10, 4))
        self.assertEqual(JUNCTIONS["Wrecksmoor Landing"], (20, 0))
        self.assertEqual(JUNCTIONS["Needle Cut"], (20, 10))
        self.assertEqual(JUNCTIONS["Net Loft"], (24, 14))

    def test_all_72_nodes_unique_and_edge_lengths(self):
        """Exactly 72 unique nodes, 72 unit-stride links, verified edge chain lengths and bounding boxes."""
        self.assertEqual(len(ALL_NODES), 72)
        total_links = 0
        interior_nodes = set()
        for eid, a_name, b_name, strides, path_order in EDGES:
            chain = GRAPH_CHAINS[eid]
            total_links += (len(chain) - 1)
            self.assertEqual(len(chain), strides + 1)
            self.assertEqual(chain[0], JUNCTIONS[a_name])
            self.assertEqual(chain[-1], JUNCTIONS[b_name])

            # Unit-step lattice adjacency along chain
            for i in range(len(chain) - 1):
                r1, c1 = chain[i]
                r2, c2 = chain[i + 1]
                dist = abs(r1 - r2) + abs(c1 - c2)
                self.assertEqual(dist, 1, f"Non-unit step on edge {eid}: {chain[i]} -> {chain[i+1]}")

            # Every chain node inside endpoints' bounding box
            min_r = min(JUNCTIONS[a_name][0], JUNCTIONS[b_name][0])
            max_r = max(JUNCTIONS[a_name][0], JUNCTIONS[b_name][0])
            min_c = min(JUNCTIONS[a_name][1], JUNCTIONS[b_name][1])
            max_c = max(JUNCTIONS[a_name][1], JUNCTIONS[b_name][1])
            for r, c in chain:
                self.assertTrue(min_r <= r <= max_r and min_c <= c <= max_c)

            interior_nodes.update(chain[1:-1])

        self.assertEqual(total_links, 72)
        self.assertEqual(len(interior_nodes), 66)

    def test_independent_route_traversal_both_directions(self):
        """Independently construct expected chains and arc ordering from the six-route table.
        Assert exact action index and neighbor, both ways, actually invoking physical_transition.
        Verify no invented coordinate-adjacency shortcuts exist.
        """
        # Independently derive chains from route table
        written_junctions = {
            "Wrecksmoor Landing": (20, 0), "Needle Cut": (20, 10),
            "Drowned Chapel": (10, 4), "Net Loft": (24, 14),
            "Harbor Lock": (6, 14), "Tidegate": (0, 14),
        }
        written_routes = [
            (0, "Wrecksmoor Landing", "Needle Cut", 10, "col-first"),
            (1, "Wrecksmoor Landing", "Drowned Chapel", 14, "row-first"),
            (2, "Drowned Chapel", "Needle Cut", 16, "col-first"),
            (3, "Needle Cut", "Net Loft", 8, "row-first"),
            (4, "Net Loft", "Harbor Lock", 18, "row-first"),
            (5, "Harbor Lock", "Tidegate", 6, "row-first"),
        ]
        expected_edge_chains = {}
        for eid, a_name, b_name, strides, path_order in written_routes:
            coord_a = written_junctions[a_name]
            coord_b = written_junctions[b_name]
            nodes = [coord_a]
            curr_r, curr_c = coord_a
            r_b, c_b = coord_b
            if path_order == "row-first":
                step_r = 1 if r_b > curr_r else (-1 if r_b < curr_r else 0)
                while curr_r != r_b:
                    curr_r += step_r
                    nodes.append((curr_r, curr_c))
                step_c = 1 if c_b > curr_c else (-1 if c_b < curr_c else 0)
                while curr_c != c_b:
                    curr_c += step_c
                    nodes.append((curr_r, curr_c))
            elif path_order == "col-first":
                step_c = 1 if c_b > curr_c else (-1 if c_b < curr_c else 0)
                while curr_c != c_b:
                    curr_c += step_c
                    nodes.append((curr_r, curr_c))
                step_r = 1 if r_b > curr_r else (-1 if r_b < curr_r else 0)
                while curr_r != r_b:
                    curr_r += step_r
                    nodes.append((curr_r, curr_c))
            expected_edge_chains[eid] = nodes

        # Independently derive incident arcs per node: sorted by edge_id ascending, toward_A before toward_B
        expected_arcs = {}
        for eid, chain in expected_edge_chains.items():
            for i, u in enumerate(chain):
                if u not in expected_arcs:
                    expected_arcs[u] = []
                if i > 0:
                    expected_arcs[u].append((eid, "toward_A", chain[i - 1]))
                if i < len(chain) - 1:
                    expected_arcs[u].append((eid, "toward_B", chain[i + 1]))

        for u in expected_arcs:
            expected_arcs[u].sort(key=lambda x: (x[0], 0 if x[1] == "toward_A" else 1))

        self.assertEqual(set(GRAPH_ARCS), set(expected_arcs))
        self.assertEqual(GRAPH_ARCS, expected_arcs)  # Exact set, no invented shortcuts.

        # Check traversal forward and backward for all routes
        for eid, chain in expected_edge_chains.items():
            # Forward A -> B
            for i in range(len(chain) - 1):
                curr = chain[i]
                nxt = chain[i + 1]
                # Expected action index from independent expected_arcs
                arcs = expected_arcs[curr]
                action_idx = [a for a, (e, d, neigh) in enumerate(arcs) if e == eid and d == "toward_B" and neigh == nxt][0]

                # Invoke live physical_transition
                state = (curr, True, True)  # Key and pearls held so lock permits passage
                next_state, r, term, inv_reason, pp, kp = physical_transition(state, action_idx)
                self.assertEqual(next_state[0], nxt, f"Failed forward step on edge {eid} from {curr} to {nxt}")
                self.assertIsNone(inv_reason)
                self.assertEqual(next_state[1:], (True, True))
                self.assertAlmostEqual(r, -0.01 + (10.0 if nxt == (0, 14) else 0.0))
                self.assertEqual(term, nxt == (0, 14))
                self.assertFalse(pp)
                self.assertFalse(kp)
                live = ReachWorld()
                live.pos, live.has_key, live.pearls_collected = curr, True, True
                observed, reward, terminal, truncated = live.step(action_idx)
                self.assertEqual(observed, (nxt, frozenset({1})))
                self.assertEqual((reward, terminal, truncated), (r, term, False))

            # Backward B -> A
            for i in range(len(chain) - 1, 0, -1):
                curr = chain[i]
                nxt = chain[i - 1]
                arcs = expected_arcs[curr]
                action_idx = [a for a, (e, d, neigh) in enumerate(arcs) if e == eid and d == "toward_A" and neigh == nxt][0]

                state = (curr, True, True)
                next_state, r, term, inv_reason, pp, kp = physical_transition(state, action_idx)
                self.assertEqual(next_state[0], nxt, f"Failed backward step on edge {eid} from {curr} to {nxt}")
                self.assertIsNone(inv_reason)
                self.assertEqual(next_state[1:], (True, True))
                self.assertAlmostEqual(r, -0.01 + (10.0 if nxt == (0, 14) else 0.0))
                self.assertEqual(term, nxt == (0, 14))
                self.assertFalse(pp)
                self.assertFalse(kp)
                live = ReachWorld()
                live.pos, live.has_key, live.pearls_collected = curr, True, True
                observed, reward, terminal, truncated = live.step(action_idx)
                self.assertEqual(observed, (nxt, frozenset({1})))
                self.assertEqual((reward, terminal, truncated), (r, term, False))

    def test_bfs_shortest_distances_between_all_junction_pairs(self):
        """Verify BFS graph distances match stride specifications and route topology."""
        adj = {}
        for u, arcs in GRAPH_ARCS.items():
            adj[u] = [neigh for _, _, neigh in arcs]

        def bfs_dist(start, goal):
            q = deque([(start, 0)])
            seen = {start}
            while q:
                curr, d = q.popleft()
                if curr == goal:
                    return d
                for nxt in adj[curr]:
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append((nxt, d + 1))
            return None

        # Verify exact edge stride counts via BFS distance
        self.assertEqual(bfs_dist(JUNCTIONS["Wrecksmoor Landing"], JUNCTIONS["Needle Cut"]), 10)
        self.assertEqual(bfs_dist(JUNCTIONS["Wrecksmoor Landing"], JUNCTIONS["Drowned Chapel"]), 14)
        self.assertEqual(bfs_dist(JUNCTIONS["Drowned Chapel"], JUNCTIONS["Needle Cut"]), 16)
        self.assertEqual(bfs_dist(JUNCTIONS["Needle Cut"], JUNCTIONS["Net Loft"]), 8)
        self.assertEqual(bfs_dist(JUNCTIONS["Net Loft"], JUNCTIONS["Harbor Lock"]), 18)
        self.assertEqual(bfs_dist(JUNCTIONS["Harbor Lock"], JUNCTIONS["Tidegate"]), 6)

    def test_real_14_stride_pearl_road_to_chapel_and_rearm(self):
        """Traverse the actual 14-stride Pearl Road from dawn to Chapel.
        Assert exact net +0.99, actual pickup event, internal flag, hidden pearl observation,
        and re-arming upon reset.
        """
        obs = self.world.reset()
        self.assertEqual(obs, (START_POS, frozenset()))
        self.assertFalse(self.world.pearls_collected)

        chain = GRAPH_CHAINS[1]
        for step_idx in range(len(chain) - 1):
            curr_pos = chain[step_idx]
            next_pos = chain[step_idx + 1]
            arcs = GRAPH_ARCS[curr_pos]
            action = [a for a, (e, d, neigh) in enumerate(arcs) if e == 1 and neigh == next_pos][0]

            obs, r, term, trunc = self.world.step(action)
            tel = self.world.last_telemetry

            if step_idx < len(chain) - 2:
                self.assertAlmostEqual(r, STRIDE_COST)
                self.assertFalse(self.world.pearls_collected)
                self.assertFalse(tel["pearls_picked"])
            else:
                self.assertEqual(obs[0], CHAPEL_POS)
                self.assertAlmostEqual(r, STRIDE_COST + PEARLS_REWARD)
                self.assertTrue(self.world.pearls_collected)
                self.assertTrue(tel["pearls_picked"])
                self.assertNotIn(ITEM_PEARLS, obs[1])
                self.assertNotIn(ITEM_PEARLS, self.world.state()[1])

        # Step away and back: no second reward
        step_away_action = 0
        obs_away, r_away, _, _ = self.world.step(step_away_action)
        self.assertAlmostEqual(r_away, STRIDE_COST)

        # Reset re-arms pearls
        self.world.reset()
        self.assertFalse(self.world.pearls_collected)

    def test_harbor_lock_boundary_requires_key(self):
        """Edge 5 is impassable from Harbor Lock without key; passable with key in both directions."""
        self.world.reset()
        self.world.pos = HARBOR_LOCK_POS
        self.world.has_key = False

        edge5_interior = GRAPH_CHAINS[5][1]
        arcs = GRAPH_ARCS[HARBOR_LOCK_POS]
        action_gate = [a for a, (e, d, neigh) in enumerate(arcs) if neigh == edge5_interior][0]

        # Blocked without key
        obs, r, term, trunc = self.world.step(action_gate)
        self.assertEqual(obs[0], HARBOR_LOCK_POS)
        self.assertAlmostEqual(r, STRIDE_COST)

        # Passable with key
        self.world.has_key = True
        obs, r, term, trunc = self.world.step(action_gate)
        self.assertEqual(obs[0], edge5_interior)
        self.assertAlmostEqual(r, STRIDE_COST)

        # Passable in reverse direction
        arcs_int = GRAPH_ARCS[edge5_interior]
        action_back = [a for a, (e, d, neigh) in enumerate(arcs_int) if neigh == HARBOR_LOCK_POS][0]
        obs_back, r_back, _, _ = self.world.step(action_back)
        self.assertEqual(obs_back[0], HARBOR_LOCK_POS)

    def test_tidegate_escape_terminal_precedence(self):
        """Tidegate ends loop as terminal escape with +10 reward."""
        self.world.reset()
        self.world.pos = GRAPH_CHAINS[5][-2]
        self.world.has_key = True
        self.world.stride_count = 53
        arcs = GRAPH_ARCS[self.world.pos]
        action_exit = [a for a, (e, d, neigh) in enumerate(arcs) if neigh == TIDEGATE_POS][0]

        obs, r, term, trunc = self.world.step(action_exit)
        self.assertEqual(obs[0], TIDEGATE_POS)
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertAlmostEqual(r, STRIDE_COST + TIDEGATE_REWARD)
        self.assertTrue(self.world.escaped)

        with self.assertRaises(RuntimeError):
            self.world.step(0)

    def test_truncation_at_54_strides(self):
        """Stride 54 truncates if not terminal; post-truncation step is rejected."""
        self.world.reset()
        for i in range(53):
            obs, r, term, trunc = self.world.step(3)
            self.assertFalse(term)
            self.assertFalse(trunc)

        obs, r, term, trunc = self.world.step(3)
        self.assertFalse(term)
        self.assertTrue(trunc)

        with self.assertRaises(RuntimeError):
            self.world.step(0)

    def test_invalid_actions_and_stays(self):
        """Action index without arc stays in place at step cost."""
        self.world.reset()
        obs, r, term, trunc = self.world.step(3)
        self.assertEqual(obs[0], START_POS)
        self.assertAlmostEqual(r, STRIDE_COST)
        self.assertEqual(self.world.last_telemetry["invalid_reason"], "no_incident_arc")

        with self.assertRaises(ValueError):
            self.world.step(-1)
        with self.assertRaises(ValueError):
            self.world.step(4)


if __name__ == "__main__":
    unittest.main()
