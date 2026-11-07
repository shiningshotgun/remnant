"""Tests for the Return Tally, the episodic runner, and the return_tally_v1 driver."""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from remnant_core.return_tally import ReturnTally, UNKNOWN_VALUE
from remnant_core.episodic_runner import EpisodicRunner, ARMS
from remnant_core.worlds.reach import TIDEGATE_POS, HARBOR_LOCK_POS


class TestReturnTally(unittest.TestCase):
    def test_discounted_return_to_go_ends_at_tide(self):
        t = ReturnTally(gamma=0.5)
        t.ingest_tide([("s0", 0, 1.0), ("s1", 1, 2.0), ("s2", 2, 4.0)])
        self.assertAlmostEqual(t.value("s2", 2), 4.0)
        self.assertAlmostEqual(t.value("s1", 1), 2.0 + 0.5 * 4.0)
        self.assertAlmostEqual(t.value("s0", 0), 1.0 + 0.5 * 4.0)

    def test_keeps_best_lived_return_and_counts_visits(self):
        t = ReturnTally(gamma=1.0)
        t.ingest_tide([("s", 0, 5.0)])
        t.ingest_tide([("s", 0, 1.0)])
        self.assertEqual(t.value("s", 0), 5.0)
        self.assertEqual(t.visits[("s", 0)], 2)

    def test_nothing_crosses_tides(self):
        t = ReturnTally(gamma=1.0)
        t.ingest_tide([("a", 0, -1.0)])
        t.ingest_tide([("b", 0, 10.0)])
        self.assertEqual(t.value("a", 0), -1.0)

    def test_greedy_prefers_best_and_unknown_default(self):
        t = ReturnTally()
        t.best[("s", 2)] = 0.5
        t.best[("s", 0)] = -0.2
        rng = random.Random(0)
        self.assertEqual(t.act("s", rng, eps=0.0), 2)
        t.best[("s", 2)] = -0.5
        picks = {t.act("s", random.Random(i), eps=0.0) for i in range(30)}
        self.assertEqual(picks, {1, 3})  # unknown (0.0) beats lived -0.2 and -0.5
        self.assertEqual(UNKNOWN_VALUE, 0.0)


class TestEpisodicRunner(unittest.TestCase):
    def test_arms(self):
        self.assertEqual(set(ARMS), {"ec_curious", "q_curious", "ec_amnesic", "ec_only"})
        with self.assertRaises(ValueError):
            EpisodicRunner(1, "nope")

    def test_frozen_eval_never_uses_curiosity_and_never_learns(self):
        r = EpisodicRunner(3, "ec_curious")
        r.run(5)
        before = (dict(r.tally.best), dict(r.tally.visits), len(r.model.counts))
        def boom(*a, **k):
            raise AssertionError("curiosity used in frozen eval")
        r.frontier_action = boom
        r.evaluate(2, 11)
        self.assertEqual(before, (dict(r.tally.best), dict(r.tally.visits), len(r.model.counts)))

    def test_q_curious_eval_does_not_learn(self):
        r = EpisodicRunner(3, "q_curious")
        r.run(3)
        q = dict(r.agent.q)
        n = len(r.memory.entries)
        r.evaluate(2, 5)
        self.assertEqual(q, r.agent.q)
        self.assertEqual(n, len(r.memory.entries))

    def test_amnesic_wipes_at_dawn(self):
        r = EpisodicRunner(4, "ec_amnesic")
        r.run(3)
        rec = r.train_tide()
        self.assertEqual(sum(r.tally.visits.values()), rec["strides"])

    def test_matched_tide_coin(self):
        kinds = []
        for arm in ("ec_curious", "q_curious", "ec_amnesic"):
            r = EpisodicRunner(5, arm)
            kinds.append([t["kind"] for t in r.run(10)])
        self.assertEqual(kinds[0], kinds[1])
        self.assertEqual(kinds[0], kinds[2])
        self.assertTrue(all(t["kind"] == "policy" for t in EpisodicRunner(5, "ec_only").run(10)))

    def test_frontier_excludes_escape_place(self):
        r = EpisodicRunner(6, "ec_curious")
        tg = (TIDEGATE_POS, frozenset({1}))
        r.terminal_places.add(tg)
        # Build a map where the only untried place is Tidegate itself.
        hl = (HARBOR_LOCK_POS, frozenset({1}))
        for a in range(4):
            r.model.observe(hl, a, -0.01, hl if a != 3 else tg)
        self.assertIsNone(r.frontier_action(hl))

    def test_frontier_takes_untried_action_here(self):
        r = EpisodicRunner(6, "ec_curious")
        p = ((20, 0), frozenset())
        r.model.observe(p, 0, -0.01, ((20, 1), frozenset()))
        self.assertEqual(r.frontier_action(p), 1)

    def test_determinism(self):
        a = EpisodicRunner(8, "ec_curious").run(15)
        b = EpisodicRunner(8, "ec_curious").run(15)
        self.assertEqual(a, b)

    def test_prov_accounts_every_stride(self):
        for t in EpisodicRunner(9, "ec_curious").run(10):
            self.assertEqual(sum(t["prov"].values()), t["strides"])


class TestDriverStats(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
        import run_return_tally_v1 as d
        self.d = d

    def test_fisher(self):
        # 3/3 vs 0/3: p = 1/20
        self.assertAlmostEqual(self.d.fisher_one_sided_greater(3, 3, 0, 3), 0.05)
        self.assertAlmostEqual(self.d.fisher_one_sided_greater(0, 3, 0, 3), 1.0)

    def test_wilson(self):
        lo, hi = self.d.wilson(30, 30)
        self.assertAlmostEqual(lo, 0.8865, places=3)
        self.assertAlmostEqual(hi, 1.0)

    def test_config_frozen(self):
        c = self.d.CONFIG
        self.assertEqual(c["seeds"], list(range(7001, 7031)))
        self.assertEqual(c["tides"], 100)
        self.assertEqual(c["P2_min_seeds"], 24)
        self.assertEqual(set(c["arms"]), {"ec_curious", "q_curious", "ec_amnesic", "ec_only", "ec_curious_p25"})


if __name__ == "__main__":
    unittest.main()
