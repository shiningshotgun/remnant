"""Derived, post-hoc analysis of results/return_tally_v1.json (read-only).

NOT part of the pre-registered analysis. The pre-registered test (P3) was an unpaired
one-sided Fisher exact test per control; it is recomputed here unchanged as the record.
Because every arm uses the same 30 seeds (7001-7030), a paired comparison is also
appropriate: for each seed, compare the primary arm's seed-level endpoint with the
control's. Discordant pairs: primary succeeds & control fails ("wins"), the reverse
("losses"). Exact one-sided sign test (equivalently exact McNemar) on the discordant
pairs: p = P(Binomial(wins + losses, 1/2) >= wins).

The statistical unit is the seed (n = 30 per arm). The 5 frozen evaluation tides per
seed are repeated deployment checks in one deterministic world, not independent samples.

Usage: python3 analysis/paired_return_tally_v1.py [--write]
  prints the analysis; --write also writes analysis/paired_return_tally_v1.json.
"""
import hashlib
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "return_tally_v1.json")
OUT = os.path.join(ROOT, "analysis", "paired_return_tally_v1.json")
PRIMARY = "ec_curious"
CONTROLS = ("q_curious", "ec_amnesic", "ec_only")
ALPHA_EACH = 0.01 / 3  # pre-specified Bonferroni threshold (unchanged)


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fisher_one_sided_greater(a, n1, c, n2):
    k, N = a + c, n1 + n2
    return sum(math.comb(n1, x) * math.comb(n2, k - x) / math.comb(N, k)
               for x in range(a, min(n1, k) + 1))


def sign_test_one_sided(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def analyse(data):
    rows = data["results"]
    seeds = [r["seed"] for r in rows[PRIMARY]]
    out = {"source": "results/return_tally_v1.json", "unit": "seed", "seeds": [seeds[0], seeds[-1]],
           "n_seeds_per_arm": len(seeds), "eval_tides_per_seed": 5,
           "endpoint": "frozen policy escapes in all 5 post-training evaluation tides",
           "arms": {}, "preregistered_fisher_unpaired": {}, "posthoc_paired_sign_test": {}}
    for arm, rs in rows.items():
        assert [r["seed"] for r in rs] == seeds, arm
        k = sum(bool(r["endpoint"]) for r in rs)
        lo, hi = wilson(k, len(rs))
        out["arms"][arm] = {"endpoint_k": k, "n": len(rs), "wilson95": [round(lo, 4), round(hi, 4)],
                            "eval_tide_escapes": sum(r["post_escapes"] for r in rs),
                            "pre_training_eval_tide_escapes": sum(r["pre_escapes"] for r in rs)}
    pk = out["arms"][PRIMARY]["endpoint_k"]
    for c in CONTROLS:
        ck = out["arms"][c]["endpoint_k"]
        out["preregistered_fisher_unpaired"][c] = fisher_one_sided_greater(pk, 30, ck, 30)
        wins = sum(1 for p, q in zip(rows[PRIMARY], rows[c]) if p["endpoint"] and not q["endpoint"])
        losses = sum(1 for p, q in zip(rows[PRIMARY], rows[c]) if q["endpoint"] and not p["endpoint"])
        both = sum(1 for p, q in zip(rows[PRIMARY], rows[c]) if p["endpoint"] and q["endpoint"])
        neither = 30 - wins - losses - both
        pv = sign_test_one_sided(wins, losses)
        out["posthoc_paired_sign_test"][c] = {"wins": wins, "losses": losses, "both_succeed": both,
                                              "neither": neither, "p_one_sided": pv,
                                              "below_bonferroni_0.01_over_3": pv < ALPHA_EACH}
    return out


def main():
    raw = open(RESULTS, "rb").read()
    res = analyse(json.loads(raw))
    res["source_sha256"] = hashlib.sha256(raw).hexdigest()
    text = json.dumps(res, indent=1, sort_keys=True)
    print(text)
    if sys.argv[1:] == ["--write"]:
        with open(OUT, "w") as f:
            f.write(text + "\n")
    elif sys.argv[1:]:
        print("usage: paired_return_tally_v1.py [--write]", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
