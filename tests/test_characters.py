"""Tests for autonomous characters on the Reach of Vesper."""

import unittest
from remnant_core.characters import Gleaner, Brannoch, Dov, CharacterManager
from remnant_core.worlds.reach import (
    ReachWorld,
    CHAPEL_POS,
    NET_LOFT_POS,
    HARBOR_LOCK_POS,
    START_POS,
    GRAPH_CHAINS,
    GRAPH_ARCS,
    STRIDE_COST,
    PEARLS_REWARD,
    LAMP_KEY_REWARD,
    ITEM_LAMP_KEY,
    ITEM_PEARLS,
)


class TestCharacters(unittest.TestCase):
    def test_gleaner_arrival_stay_absence(self):
        """Gleaner greets warmly on arrival, invites to stay while present, tends when absent."""
        g = Gleaner(home_pos=(10, 4))
        self.assertEqual(g.current_activity, "tending the chapel")

        # Absent
        act = g.step(wick_pos=(20, 0))
        self.assertEqual(act, "tending the chapel")

        # Arrival
        act = g.step(wick_pos=(10, 4))
        self.assertEqual(act, "greeting Wick warmly")

        # Continued presence
        act = g.step(wick_pos=(10, 4))
        self.assertEqual(act, "inviting Wick to stay")

        # Departure
        act = g.step(wick_pos=(10, 5))
        self.assertEqual(act, "tending the chapel")

    def test_brannoch_key_blindness_and_work_counter(self):
        """Brannoch tells story when Wick is present; mends net and increments counter when absent.
        Changing key flag alone does not change decision.
        """
        b = Brannoch(home_pos=(24, 14))
        self.assertEqual(b.work_counter, 0)

        # Absent step 1
        b.step(wick_pos=(20, 10))
        self.assertEqual(b.work_counter, 1)
        self.assertEqual(b.current_activity, "mending the unfinished net")

        # Absent step 2
        b.step(wick_pos=(20, 11))
        self.assertEqual(b.work_counter, 2)

        # Wick arrives at Net Loft
        act = b.step(wick_pos=(24, 14))
        self.assertEqual(act, "telling a recurring story")
        self.assertEqual(b.work_counter, 2)

    def test_dov_procedural_requirements(self):
        """Dov inspects mechanism when absent; demands key if Wick has none; acknowledges key if held."""
        d = Dov(home_pos=(6, 14))
        self.assertEqual(d.current_activity, "inspecting the mechanism")

        # Absent
        act = d.step(wick_pos=(20, 0), wick_has_key=False)
        self.assertEqual(act, "inspecting the mechanism")

        # Wick at lock without key
        act = d.step(wick_pos=(6, 14), wick_has_key=False)
        self.assertEqual(act, "requiring the Lamp-Key")

        # Wick at lock with key
        act = d.step(wick_pos=(6, 14), wick_has_key=True)
        self.assertEqual(act, "acknowledging the Lamp-Key and procedure")

    def test_clock_independence_and_counterfactual_agency(self):
        """Same stride, different Wick position produces different choices.
        Same position, different stride produces identical choices.
        """
        # Hold percepts and local states fixed, change only world clock.
        for pos in (CHAPEL_POS, HARBOR_LOCK_POS, NET_LOFT_POS, START_POS):
            first, second = ReachWorld(), ReachWorld()
            for world in (first, second):
                world.pos = pos
                world.has_key = True
                world.characters.gleaner.wick_present_prior = True
                world.characters.brannoch.work_counter = 7
            first.stride_count, second.stride_count = 0, 40
            first.step(3)
            second.step(3)
            self.assertEqual(first.npc_states(), second.npc_states())
        cm = CharacterManager()
        cm.step(wick_pos=CHAPEL_POS, wick_has_key=False)
        snap1 = cm.npc_states()
        self.assertEqual(snap1["Gleaner"]["activity"], "greeting Wick warmly")
        self.assertEqual(snap1["Dov"]["activity"], "inspecting the mechanism")

        cm.reset()
        cm.step(wick_pos=HARBOR_LOCK_POS, wick_has_key=False)
        snap2 = cm.npc_states()
        self.assertEqual(snap2["Gleaner"]["activity"], "tending the chapel")
        self.assertEqual(snap2["Dov"]["activity"], "requiring the Lamp-Key")

    def test_characters_exert_no_physical_force(self):
        """Physical lock passage is distinct from terminal escape six strides later."""
        from remnant_core.worlds.reach import TIDEGATE_POS, TIDEGATE_REWARD

        # Independently specified route 5: Harbor Lock (6,14) to Tidegate (0,14).
        # At Harbor Lock route 4 toward_A is action 0, route 5 toward_B is action 1.
        world = ReachWorld()
        world.pos = HARBOR_LOCK_POS
        obs, reward, terminal, truncated = world.step(1)
        self.assertEqual(obs, (HARBOR_LOCK_POS, frozenset()))
        self.assertAlmostEqual(reward, -0.01)
        self.assertFalse(terminal)
        self.assertFalse(truncated)
        self.assertEqual(world.last_telemetry["invalid_reason"], "locked_gate_no_key")
        self.assertEqual(world.characters.dov.current_activity, "requiring the Lamp-Key")

        # Approach on route 4 ends at the lock, not at the exit, with either inventory.
        # Route 4 is row-first from (24,14) to (6,14): (7,14) -> (6,14) action 1.
        for held in (False, True):
            arrival = ReachWorld()
            arrival.pos, arrival.has_key = (7, 14), held
            arrived, reward, terminal, truncated = arrival.step(1)
            self.assertEqual(arrived, (HARBOR_LOCK_POS, frozenset({1}) if held else frozenset()))
            self.assertAlmostEqual(reward, -0.01)
            self.assertFalse(terminal)
            self.assertFalse(truncated)
            self.assertEqual(arrival.characters.dov.current_activity,
                "acknowledging the Lamp-Key and procedure" if held else "requiring the Lamp-Key")

        world.has_key = True  # Test-local physical fixture, not a production policy.
        for offset in range(1, 7):
            obs, reward, terminal, truncated = world.step(1)
            self.assertEqual(obs, ((6 - offset, 14), frozenset({ITEM_LAMP_KEY})))
            self.assertTrue(world.has_key)
            self.assertEqual(terminal, offset == 6)
            self.assertFalse(truncated)
            self.assertAlmostEqual(reward, -0.01 + (10.0 if offset == 6 else 0.0))
            self.assertEqual(world.characters.dov.current_activity, "inspecting the mechanism")
        self.assertEqual(obs[0], TIDEGATE_POS)
        self.assertAlmostEqual(TIDEGATE_REWARD, 10.0)
        with self.assertRaises(RuntimeError):
            world.step(0)

        # Re-entering the lock from route 5 is not terminal; Dov acknowledges key.
        reverse = ReachWorld()
        reverse.pos = (5, 14)
        reverse.has_key = True
        obs, reward, terminal, truncated = reverse.step(0)
        self.assertEqual(obs, (HARBOR_LOCK_POS, frozenset({ITEM_LAMP_KEY})))
        self.assertAlmostEqual(reward, -0.01)
        self.assertFalse(terminal)
        self.assertFalse(truncated)
        self.assertEqual(reverse.characters.dov.current_activity,
                         "acknowledging the Lamp-Key and procedure")

        # Pearl collection is triggered by entry, once per loop, hidden from observation.
        pearls = ReachWorld()
        pearls.pos = (10, 3)
        for action, position, expected_reward, picked in [
            (1, (10, 4), 0.99, True),
            (0, (10, 3), -0.01, False),
            (1, (10, 4), -0.01, False),
        ]:
            obs, reward, terminal, truncated = pearls.step(action)
            self.assertEqual(obs, (position, frozenset()))
            self.assertAlmostEqual(reward, expected_reward)
            self.assertEqual(pearls.last_telemetry["pearls_picked"], picked)
            self.assertFalse(terminal)
            self.assertFalse(truncated)
        pearls.reset()
        pearls.pos = (10, 3)
        self.assertAlmostEqual(pearls.step(1)[1], 0.99)

        # Net Loft entry collects key once, irrespective of Brannoch's work counter.
        key = ReachWorld()
        key.pos = (24, 13)
        key.characters.brannoch.work_counter = 37
        obs, reward, terminal, truncated = key.step(1)
        self.assertEqual(obs, (NET_LOFT_POS, frozenset({ITEM_LAMP_KEY})))
        self.assertAlmostEqual(reward, 0.09)
        self.assertTrue(key.last_telemetry["key_picked"])
        self.assertFalse(terminal)
        self.assertFalse(truncated)

        # All directed links under each item state: changing only NPC state cannot
        # change movement, inventory, scalar reward or termination flags.
        for position, arcs in GRAPH_ARCS.items():
            for action in range(4):
                for has_key in (False, True):
                    for has_pearls in (False, True):
                        with self.subTest(position=position, action=action,
                                          key=has_key, pearls=has_pearls):
                            first, second = ReachWorld(), ReachWorld()
                            for candidate in (first, second):
                                candidate.pos = position
                                candidate.has_key = has_key
                                candidate.pearls_collected = has_pearls
                            second.characters.gleaner.wick_present_prior = True
                            second.characters.brannoch.work_counter = 53
                            second.characters.dov.current_activity = "test-local prior activity"
                            self.assertEqual(first.step(action), second.step(action))
                            for field in ("pearls_picked", "key_picked", "invalid_reason"):
                                self.assertEqual(first.last_telemetry[field],
                                                 second.last_telemetry[field])

        # Deep isolation must mutate actual nested fields, not an unused extra key.
        snapshot = pearls.npc_states()
        snapshot["Gleaner"]["local_state"]["wick_present_prior"] = False
        self.assertTrue(pearls.npc_states()["Gleaner"]["local_state"]["wick_present_prior"])
        telemetry = pearls.last_telemetry
        telemetry["npc_states"]["Gleaner"]["activity"] = "MUTATED"
        self.assertNotEqual(pearls.last_telemetry["npc_states"]["Gleaner"]["activity"], "MUTATED")


if __name__ == "__main__":
    unittest.main()
