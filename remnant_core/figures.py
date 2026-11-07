"""Figure rendering module for the Reach of Vesper.

Reads saved records and outcomes from disk and generates 7 evidence pack PNGs:
1. reach-map.png
2. visit-frequency.png
3. stride-economy.png
4. escape-timing.png
5. pearls-taken.png
6. route-choices.png
7. character-activity.png
"""

import json
import os
from typing import Dict, Any

from remnant_core.worlds.reach import (
    JUNCTIONS,
    EDGES,
    GRAPH_CHAINS,
    GRAPH_ARCS,
    START_POS,
    CHAPEL_POS,
    NET_LOFT_POS,
    HARBOR_LOCK_POS,
    TIDEGATE_POS,
)


def render_all_figures(
    records_path: str = "results/records.json",
    outcomes_path: str = "results/outcomes.json",
    output_dir: str = "figures",
) -> Dict[str, str]:
    """Generate all 7 figures into output_dir and return their file paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    with open(records_path, "r", encoding="utf-8") as fp:
        records_payload = json.load(fp)

    with open(outcomes_path, "r", encoding="utf-8") as fp:
        outcomes_payload = json.load(fp)

    records = records_payload["records"]
    protocol = outcomes_payload.get("protocol", {})
    horizon = protocol.get("horizon_k_max", 100)
    summary = outcomes_payload["summary"]
    per_loop = outcomes_payload["per_loop"]
    visitation = outcomes_payload.get("visitation", {})

    generated_paths = {}

    # -------------------------------------------------------------
    # 1. reach-map.png
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))
    for eid, a_name, b_name, strides, path_order in EDGES:
        chain = GRAPH_CHAINS[eid]
        xs = [c for r, c in chain]
        ys = [-r for r, c in chain]
        ax.plot(xs, ys, color="gray", linewidth=2, zorder=1)
        mid_idx = len(chain) // 2
        mid_x, mid_y = chain[mid_idx][1], -chain[mid_idx][0]
        ax.text(mid_x + 0.3, mid_y + 0.3, f"{strides} st", fontsize=9, color="darkblue", weight="bold")

    all_r = [r for r, c in GRAPH_ARCS.keys()]
    all_c = [c for r, c in GRAPH_ARCS.keys()]
    ax.scatter(all_c, [-r for r in all_r], color="lightgray", edgecolors="black", s=40, zorder=2)

    landmarks = [
        (START_POS, "Start (Wrecksmoor)", "blue", "s"),
        (CHAPEL_POS, "Pearls (Chapel)", "gold", "o"),
        (NET_LOFT_POS, "Lamp-Key (Net Loft)", "green", "^"),
        (HARBOR_LOCK_POS, "Harbor Lock", "darkred", "X"),
        (TIDEGATE_POS, "Exit (Tidegate)", "purple", "*"),
    ]
    for pos, label, col, marker in landmarks:
        ax.scatter([pos[1]], [-pos[0]], color=col, marker=marker, s=160, edgecolors="black", zorder=3, label=label)
        ax.annotate(f" {label}", (pos[1], -pos[0]), fontsize=10, weight="bold", zorder=4)

    nc_pos = JUNCTIONS["Needle Cut"]
    ax.scatter([nc_pos[1]], [-nc_pos[0]], color="orange", marker="D", s=100, edgecolors="black", zorder=3)
    ax.annotate(" Needle Cut", (nc_pos[1], -nc_pos[0]), fontsize=10, weight="bold")

    ax.set_title("Reach of Vesper - Waypoint Graph Layout (72 Nodes, 6 Routes)", fontsize=13, weight="bold")
    ax.set_xlabel("Lattice Column (East ->)")
    ax.set_ylabel("Lattice Row (North is Up; decreasing row)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()
    map_path = os.path.join(output_dir, "reach-map.png")
    fig.savefig(map_path, dpi=150)
    plt.close(fig)
    generated_paths["reach-map"] = map_path

    # -------------------------------------------------------------
    # 2. visit-frequency.png
    # -------------------------------------------------------------
    fig, (ax_e, ax_l) = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

    ev = visitation.get("early_node_visits", {})
    lv = visitation.get("late_node_visits", {})
    tot_e = visitation.get("total_strides_early", 1)
    tot_l = visitation.get("total_strides_late", 1)

    max_frac = 0.05
    for v_dict, tot in [(ev, tot_e), (lv, tot_l)]:
        for k_str, count in v_dict.items():
            frac = count / tot if tot > 0 else 0
            if frac > max_frac:
                max_frac = frac

    for ax_panel, v_dict, tot, title in [
        (ax_e, ev, tot_e, f"Early Loops (1..10 Pooled, {tot_e} strides)"),
        (ax_l, lv, tot_l, f"Late Loops (Final 10 Pooled, {tot_l} strides)"),
    ]:
        for eid in GRAPH_CHAINS:
            chain = GRAPH_CHAINS[eid]
            ax_panel.plot([c for r, c in chain], [-r for r, c in chain], color="lightgray", zorder=1)

        xs, ys, colors, sizes = [], [], [], []
        for r, c in GRAPH_ARCS.keys():
            k_str = f"{r},{c}"
            cnt = v_dict.get(k_str, 0)
            frac = cnt / tot if tot > 0 else 0.0
            xs.append(c)
            ys.append(-r)
            colors.append(frac)
            sizes.append(30 + 300 * (frac / max_frac if max_frac > 0 else 0))

        sc = ax_panel.scatter(xs, ys, c=colors, cmap="viridis", s=sizes, vmin=0, vmax=max_frac, edgecolors="black", zorder=2)
        ax_panel.set_title(title, fontsize=11, weight="bold")
        ax_panel.set_xlabel("Lattice Column")
        ax_panel.grid(True, linestyle=":", alpha=0.4)

    ax_e.set_ylabel("Lattice Row (-row, North up)")
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label="Normalized Node Visit Fraction")
    fig.suptitle("Reach of Vesper: Spatial Visitation Density (Early vs Late Loops - Descriptive Exposure)", fontsize=14, weight="bold")
    visit_path = os.path.join(output_dir, "visit-frequency.png")
    fig.savefig(visit_path, dpi=150)
    plt.close(fig)
    generated_paths["visit-frequency"] = visit_path

    # -------------------------------------------------------------
    # 3. stride-economy.png
    # -------------------------------------------------------------
    fig, (ax, ax_cov) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    se_data = per_loop.get("stride_economy", {})
    loops_sorted = sorted([int(k) for k in se_data.keys()])
    means = [se_data[str(k)]["mean_strides"] for k in loops_sorted]
    sds = [se_data[str(k)]["pop_sd_strides"] for k in loops_sorted]
    active_ns = [se_data[str(k)]["active_seeds"] for k in loops_sorted]

    upper = [m + s for m, s in zip(means, sds)]
    lower = [max(0, m - s) for m, s in zip(means, sds)]

    ax.plot(loops_sorted, means, color="teal", linewidth=2, label="Mean Strides per Loop (Active Seeds)")
    ax.fill_between(loops_sorted, lower, upper, color="teal", alpha=0.2, label="+/- 1 Population SD")
    ax.axhline(54, color="crimson", linestyle="--", label="Loop Limit L = 54")
    ax.axhline(42, color="green", linestyle=":", label="Shortest Escape = 42")

    ax.set_title("Stride Economy Across Tides (Active-Seed Denominator)", fontsize=12, weight="bold")
    ax.set_ylabel("Strides Attempted")
    ax.set_ylim(0, 60)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left")

    # Lower panel showing active seeds
    ax_cov.plot(loops_sorted, active_ns, color="navy", linewidth=1.5, marker=".", label="Active Seeds (Denominator n)")
    ax_cov.set_ylabel("Active n")
    ax_cov.set_xlabel("Loop Index (Tide)")
    ax_cov.grid(True, linestyle=":", alpha=0.5)
    ax_cov.legend(loc="upper right")

    plt.tight_layout()
    se_path = os.path.join(output_dir, "stride-economy.png")
    fig.savefig(se_path, dpi=150)
    plt.close(fig)
    generated_paths["stride-economy"] = se_path

    # -------------------------------------------------------------
    # 4. escape-timing.png
    # -------------------------------------------------------------
    # Escape time is numeric; censoring is a separate categorical outcome.
    fig, (ax, ax_censor) = plt.subplots(
        1, 2, figsize=(9, 5), gridspec_kw={"width_ratios": [3, 1]}, sharey=True)
    esc_dist = summary.get("escape_distribution", {})
    esc_loops = list(esc_dist.values())
    censored_count = summary.get("censored_count", 0)

    if esc_loops:
        bin_width = max(1, horizon // 20)
        bins = list(range(1, horizon + 2, bin_width))
        if bins[-1] < horizon + 1:
            bins.append(horizon + 1)
        ax.hist(esc_loops, bins=bins, color="royalblue", edgecolor="black")
    else:
        ax.text(0.5, 0.5, "No escapes observed", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, style="italic")
    ax.set_xlim(0.5, horizon + 0.5)
    ticks = sorted({1, horizon, max(1, horizon // 4),
                    max(1, horizon // 2), max(1, 3 * horizon // 4)})
    ax.set_xticks(ticks)
    ax.set_xlabel("Loop index of first escape")
    ax.set_ylabel("Number of seeds")
    ax.set_title(f"Observed escapes (n={len(esc_loops)})")
    ax.grid(True, linestyle=":", alpha=0.5)

    ax_censor.bar(["Censored"], [censored_count], width=0.55,
                  color="lightcoral", edgecolor="black")
    ax_censor.set_title(f"At horizon {horizon}")
    ax_censor.set_xlabel("Outcome category")
    ax_censor.set_ylim(0, max(1, censored_count, len(esc_loops)) * 1.15)
    ax_censor.text(0, censored_count + max(1, censored_count) * 0.03,
                   str(censored_count), ha="center")
    ax_censor.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.suptitle(f"First escape and censoring (protocol horizon {horizon})",
                 fontsize=12, weight="bold")
    plt.tight_layout()
    esc_path = os.path.join(output_dir, "escape-timing.png")
    fig.savefig(esc_path, dpi=150)
    plt.close(fig)
    generated_paths["escape-timing"] = esc_path

    # -------------------------------------------------------------
    # 5. pearls-taken.png
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    pt_data = per_loop.get("pearls_taken", {})
    p_loops = sorted([int(k) for k in pt_data.keys()])
    p_rates = [pt_data[str(k)]["pearls_taken_rate"] * 100 for k in p_loops]

    ax.plot(p_loops, p_rates, color="darkgoldenrod", linewidth=2, marker="o", markersize=3, label="Pearls Collection Rate (%)")
    ax.set_title("Pearls Collection Rate per Tide (Active-Seed Denominator)", fontsize=12, weight="bold")
    ax.set_xlabel("Loop Index (Tide)")
    ax.set_ylabel("Active Seeds Collecting Pearls (%)")
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    pearls_path = os.path.join(output_dir, "pearls-taken.png")
    fig.savefig(pearls_path, dpi=150)
    plt.close(fig)
    generated_paths["pearls-taken"] = pearls_path

    # -------------------------------------------------------------
    # 6. route-choices.png (Gaps for null departure shares)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    wm_data = per_loop.get("route_choices_wrecksmoor", {})
    nc_data = per_loop.get("route_choices_needle_cut", {})

    r_loops = sorted([int(k) for k in wm_data.keys()])

    # Wrecksmoor choices
    wm_needle_share = [wm_data[str(k)]["share_to_needle_cut"] * 100 if wm_data[str(k)]["share_to_needle_cut"] is not None else float("nan") for k in r_loops]
    wm_chapel_share = [wm_data[str(k)]["share_to_chapel"] * 100 if wm_data[str(k)]["share_to_chapel"] is not None else float("nan") for k in r_loops]

    ax1.plot(r_loops, wm_needle_share, color="blue", marker="o", markersize=3, label="To Needle Cut (Route 0)")
    ax1.plot(r_loops, wm_chapel_share, color="gold", marker="o", markersize=3, label="To Chapel / Pearls (Route 1)")
    ax1.set_title("Route Choice Shares: Wrecksmoor Landing Departures (Gaps indicate no departures)", fontsize=11, weight="bold")
    ax1.set_ylabel("Departure Share (%)")
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right")

    # Needle Cut choices
    nc_net_share = [nc_data[str(k)]["share_to_net_loft"] * 100 if nc_data[str(k)]["share_to_net_loft"] is not None else float("nan") for k in r_loops]
    nc_wm_share = [nc_data[str(k)]["share_to_wrecksmoor"] * 100 if nc_data[str(k)]["share_to_wrecksmoor"] is not None else float("nan") for k in r_loops]
    nc_ch_share = [nc_data[str(k)]["share_to_chapel"] * 100 if nc_data[str(k)]["share_to_chapel"] is not None else float("nan") for k in r_loops]

    ax2.plot(r_loops, nc_net_share, color="green", marker="o", markersize=3, label="To Net Loft / Key (Route 3)")
    ax2.plot(r_loops, nc_wm_share, color="purple", marker="o", markersize=3, label="To Wrecksmoor (Route 0)")
    ax2.plot(r_loops, nc_ch_share, color="darkorange", marker="o", markersize=3, label="To Chapel (Route 2)")
    ax2.set_title("Route Choice Shares: Needle Cut Departures (Gaps indicate no departures)", fontsize=11, weight="bold")
    ax2.set_xlabel("Loop Index (Tide)")
    ax2.set_ylabel("Departure Share (%)")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    rc_path = os.path.join(output_dir, "route-choices.png")
    fig.savefig(rc_path, dpi=150)
    plt.close(fig)
    generated_paths["route-choices"] = rc_path

    # -------------------------------------------------------------
    # 7. character-activity.png
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    lowest_seed = min(protocol.get("seeds", [101])) if protocol.get("seeds") else min(r["seed"] for r in records)
    seed_l1 = [r for r in records if r["seed"] == lowest_seed and r["loop"] == 1]
    strides = [r["stride"] for r in seed_l1]

    characters = ["Gleaner", "Brannoch", "Dov", "Ysolde"]
    char_y = {"Ysolde": 1, "Dov": 2, "Brannoch": 3, "Gleaner": 4}

    # Distinct categorical colors for activities
    act_colors = {
        "tending the chapel": "teal",
        "greeting Wick warmly": "coral",
        "inviting Wick to stay": "crimson",
        "mending the unfinished net": "steelblue",
        "telling a recurring story": "gold",
        "inspecting the mechanism": "darkcyan",
        "requiring the Lamp-Key": "firebrick",
        "acknowledging the Lamp-Key and procedure": "limegreen",
        "watching from the painted portrait": "mediumpurple",
    }

    # Track seen activities for legend
    seen_activities = {}

    for char in characters:
        y_val = char_y[char]
        activities = [r["npc_states"].get(char, {}).get("activity", "none") for r in seed_l1]
        ax.plot(strides, [y_val] * len(strides), color="lightgray", linestyle=":", alpha=0.7, zorder=1)

        # Initial activity label at dawn (stride 1)
        if activities:
            init_act = activities[0]
            ax.annotate(f"Dawn: {init_act}", (1, y_val + 0.18), fontsize=8, color="#333333", weight="semibold")

        prev_act = None
        for st, act in zip(strides, activities):
            col = act_colors.get(act, "gray")
            sc = ax.scatter([st], [y_val], color=col, s=35, edgecolors="black", linewidths=0.5, zorder=3)
            seen_activities[act] = col
            if act != prev_act and prev_act is not None:
                ax.annotate(act, (st, y_val + 0.15), fontsize=8, rotation=20, weight="bold", color=col)
            prev_act = act

    # Add categorical legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markeredgecolor="black", markersize=8, label=act)
        for act, col in seen_activities.items()
    ]
    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.0, 1.22), ncol=2, fontsize=8, framealpha=0.9)

    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["Ysolde (Portrait)", "Dov (Harbor Lock)", "Brannoch (Net Loft)", "Gleaner (Chapel)"], fontsize=10, weight="bold")
    ax.set_ylim(0.5, 5.0)
    ax.set_xlabel(f"Stride Number within Loop 1 (Seed {lowest_seed}, L=54 strides)", fontsize=10)
    ax.set_title(f"WORLD TELEMETRY: Character Activity Timeline (Seed {lowest_seed}, Loop 1)", fontsize=12, weight="bold", pad=25)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    char_path = os.path.join(output_dir, "character-activity.png")
    fig.savefig(char_path, dpi=150)
    plt.close(fig)
    generated_paths["character-activity"] = char_path

    return generated_paths
