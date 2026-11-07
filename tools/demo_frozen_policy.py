"""Small demo: train one seed of the primary arm, then deploy its frozen learned policy.

Training prerequisite (disclosed): no trained state is shipped, so this first trains
seed 7001 for 100 tides with the primary arm's exact configuration (about a second).
It then runs one frozen evaluation tide in which:
  - curiosity/frontier walking is disabled (calling it raises),
  - no store is updated (the Return Tally and map are compared before and after),
  - every stride is chosen by the learned policy with eps = 0.
Finally it checks the route against the accepted result file (read-only).

Usage: python3 tools/demo_frozen_policy.py [--seed N]   (N in 7001..7030, default 7001)
"""
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from remnant_core.episodic_runner import EpisodicRunner  # noqa: E402
from remnant_core.worlds.reach import JUNCTIONS  # noqa: E402


def main():
    args = sys.argv[1:]
    seed = 7001
    if args[:1] == ["--seed"] and len(args) == 2 and args[1].isdigit():
        seed = int(args[1])
    elif args:
        print("usage: demo_frozen_policy.py [--seed N]", file=sys.stderr)
        sys.exit(2)
    r = EpisodicRunner(seed, "ec_curious", curiosity_p=0.5)
    pre = r.evaluate(5, seed * 7 + 1)  # same call order as the accepted driver
    r.run(100)
    lived = sum(t["escaped"] for t in r.tide_log)
    print(f"trained seed {seed}: 100 tides lived, {lived} escapes lived during training, "
          f"pre-training frozen eval escapes {pre['escapes']}/5")

    def forbidden(*_a, **_k):
        raise AssertionError("curiosity planner called during frozen deployment")
    r.frontier_action = forbidden
    tally_before = copy.deepcopy(r.tally.best)
    edges_before = r.map_edges()
    ev = r.evaluate(5, seed * 7 + 2)
    assert r.tally.best == tally_before and r.map_edges() == edges_before, "store changed"
    names = {v: k for k, v in JUNCTIONS.items()}
    t0 = ev["tides"][0]
    stops = [names[tuple(p)] for p in t0["path"] if tuple(p) in names]
    dedup = [s for i, s in enumerate(stops) if i == 0 or stops[i - 1] != s]
    print(f"frozen deployment: {ev['escapes']}/5 tides escaped; first tide {t0['strides']} strides, "
          f"return {t0['ret']}")
    print("route through junctions:", " -> ".join(dedup))
    with open(os.path.join(ROOT, "results", "return_tally_v1.json")) as f:
        acc = {row["seed"]: row for row in json.load(f)["results"]["ec_curious"]}
    if seed in acc:
        same = [t["path"] for t in ev["tides"]] == acc[seed]["post_paths"]
        print("matches accepted results/return_tally_v1.json for this seed:", same)
        if not same:
            sys.exit(1)


if __name__ == "__main__":
    main()
