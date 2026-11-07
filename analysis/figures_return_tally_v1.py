"""Derived figures for the return_tally_v1 study (read-only on results).

Requires the optional plotting dependency matplotlib (see requirements-plot.txt).
Writes PNGs to figures/return_tally_v1/ (or --out DIR). Never writes to results/.

  fig1_endpoint_by_arm.png   seed-level endpoint per arm, Wilson 95% intervals (n = 30 seeds)
  fig2_discovery_and_learning.png
                             (a) share of the 30 seeds that have lived at least one escape,
                                 by training tide; (b) escape rate of policy-only training
                                 tides relative to each seed's first lived escape
  fig3_frozen_route.png      the Vesper graph and the frozen policy's escape route

Usage: python3 analysis/figures_return_tally_v1.py [--out DIR]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.paired_return_tally_v1 import wilson  # noqa: E402
from remnant_core.worlds.reach import GRAPH_CHAINS, JUNCTIONS  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "return_tally_v1.json")
LABELS = {
    "ec_curious": "Return Tally\n+ curiosity\n(primary)",
    "q_curious": "Legacy learner\n+ curiosity\n(control)",
    "ec_amnesic": "Amnesic\n(wiped each dawn)\n(control)",
    "ec_only": "No curiosity\ntides\n(control)",
    "ec_curious_p25": "Curiosity\np = 0.25\n(secondary)",
}
ORDER = ["ec_curious", "q_curious", "ec_amnesic", "ec_only", "ec_curious_p25"]
C_PRIMARY, C_CONTROL, C_SECONDARY = "#1b6ca8", "#8a8a8a", "#d08c2b"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def color(arm):
    return C_PRIMARY if arm == "ec_curious" else (C_SECONDARY if arm == "ec_curious_p25" else C_CONTROL)


def fig1(d, out):
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    for i, arm in enumerate(ORDER):
        rows = d["results"][arm]
        k, n = sum(r["endpoint"] for r in rows), len(rows)
        lo, hi = wilson(k, n)
        x = i + (0.4 if arm == "ec_curious_p25" else 0)
        ax.bar(x, k / n, width=0.6, color=color(arm), alpha=0.85)
        ax.errorbar(x, k / n, yerr=[[max(0.0, k / n - lo)], [max(0.0, hi - k / n)]], fmt="none", ecolor="black", capsize=5, lw=1.2)
        ax.text(x, hi + 0.03, f"{k}/{n}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axvline(3.7, color="#bbbbbb", lw=1, ls="--")
    ax.set_xticks([i + (0.4 if a == "ec_curious_p25" else 0) for i, a in enumerate(ORDER)])
    ax.set_xticklabels([LABELS[a] for a in ORDER], fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Share of seeds (n = 30 per arm)")
    ax.set_title("Frozen learned policy escapes on all 5 evaluation tides\n"
                 "after 100 training tides; whiskers = Wilson 95% interval", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_endpoint_by_arm.png"))
    plt.close(fig)


def fig2(d, out):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 8.4), dpi=150)
    for arm in ORDER:
        rows = d["results"][arm]
        firsts = [r["first_escape_tide"] for r in rows]
        ys = [sum(1 for f in firsts if f is not None and f <= t) / len(rows) for t in range(1, 101)]
        style = "-" if arm in ("ec_curious", "q_curious", "ec_curious_p25") else ":"
        lab = LABELS[arm].replace("\n", " ")
        a1.step(range(1, 101), ys, where="post", color=color(arm), ls=style,
                lw=2.2 if arm == "ec_curious" else 1.4, label=lab)
    a1.set_xlabel("Training tide")
    a1.set_ylabel("Share of seeds with ≥1 lived escape (n = 30)")
    a1.set_title("(a) Discovery: first lived escape", fontsize=11)
    a1.set_ylim(-0.02, 1.05)
    a1.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(0.0, 0.5), frameon=False)
    a1.text(98, 0.04, "Primary and legacy-learner arms\nhave similar, not identical\ndiscovery times",
            ha="right", fontsize=7.5, color="#555555")

    bins = list(range(-40, 61, 10))
    for arm in ("ec_curious", "q_curious"):
        esc, tot = [0] * (len(bins) - 1), [0] * (len(bins) - 1)
        for r in d["results"][arm]:
            f = r["first_escape_tide"]
            if f is None:
                continue
            for t in r["tide_log"]:
                if set(t["prov"]) != {"policy"}:
                    continue
                rel = t["tide"] - f
                for b in range(len(bins) - 1):
                    if bins[b] <= rel < bins[b + 1]:
                        tot[b] += 1
                        esc[b] += t["escaped"]
        xs = [(bins[b] + bins[b + 1]) / 2 for b in range(len(bins) - 1) if tot[b]]
        ys = [esc[b] / tot[b] for b in range(len(bins) - 1) if tot[b]]
        a2.plot(xs, ys, marker="o", color=color(arm), lw=2, label=LABELS[arm].replace("\n", " "))
    a2.axvline(0, color="#bbbbbb", ls="--", lw=1)
    a2.set_xlabel("Tides relative to the seed's first lived escape (10-tide bins)")
    a2.set_ylabel("Escape rate of policy-only training tides")
    a2.set_title("(b) Learning: does the policy repeat it?", fontsize=11)
    a2.set_ylim(-0.02, 1.0)
    a2.legend(fontsize=7.5, loc="upper left", frameon=False)
    a2.text(58, 0.9, "Training uses ε = 0.2, so training\nrates sit below frozen performance.\n"
            "Tides pooled over seeds; not independent.", ha="right", va="top", fontsize=7.5, color="#555555")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_discovery_and_learning.png"))
    plt.close(fig)


def fig3(d, out):
    paths = {json.dumps(p) for r in d["results"]["ec_curious"] for p in r["post_paths"]}
    assert len(paths) == 1, "expected one shared frozen route"
    route = json.loads(paths.pop())
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    for chain in GRAPH_CHAINS.values():
        ax.plot([c for _r, c in chain], [-r for r, _c in chain], color="#c9c9c9", lw=3, zorder=1)
        ax.scatter([c for _r, c in chain], [-r for r, _c in chain], s=8, color="#9a9a9a", zorder=2)
    ax.plot([c for _r, c in route], [-r for r, _c in route], color=C_PRIMARY, lw=2.5, zorder=3,
            label=f"Frozen policy route ({len(route) - 1} strides)")
    offs = {"Tidegate": (-3.6, 0), "Harbor Lock": (-4.2, 0), "Drowned Chapel": (0, 1.2),
            "Wrecksmoor Landing": (0, -1.8), "Needle Cut": (1.6, 0.9), "Net Loft": (3.6, -0.3)}
    notes = {"Drowned Chapel": "pearls (+1)", "Net Loft": "Lamp-Key (+0.1)", "Tidegate": "escape (+10)",
             "Wrecksmoor Landing": "dawn start", "Harbor Lock": "locked without key"}
    for name, (r, c) in JUNCTIONS.items():
        ax.scatter([c], [-r], s=60, color="black", zorder=4)
        dx, dy = offs[name]
        txt = name + ("\n" + notes[name] if name in notes else "")
        ax.text(c + dx, -r + dy, txt, ha="center", va="center", fontsize=8.5)
    ax.set_aspect("equal")
    ax.set_xlim(-2.5, 19)
    ax.set_ylim(-27.5, 2.5)
    ax.axis("off")
    ax.set_title("Vesper (72 nodes). One route, identical in all 150 frozen evaluation\n"
                 "tides of the primary arm (30 seeds × 5 tides, one deterministic world)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_frozen_route.png"))
    plt.close(fig)


def main():
    args = sys.argv[1:]
    out = os.path.join(ROOT, "figures", "return_tally_v1")
    if args[:1] == ["--out"] and len(args) == 2:
        out = args[1]
    elif args:
        print("usage: figures_return_tally_v1.py [--out DIR]", file=sys.stderr)
        sys.exit(2)
    if os.path.abspath(out).startswith(os.path.join(ROOT, "results")):
        sys.exit("refusing to write figures into results/")
    os.makedirs(out, exist_ok=True)
    with open(RESULTS) as f:
        d = json.load(f)
    fig1(d, out)
    fig2(d, out)
    fig3(d, out)
    print("wrote", sorted(p for p in os.listdir(out) if p.endswith(".png")))


if __name__ == "__main__":
    main()
