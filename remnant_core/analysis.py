"""Study analysis module for the Reach of Vesper.

Computes descriptive empirical outcomes strictly from recorded data:
1. Escape rate and loops-to-first-escape distribution (censoring handled explicitly).
2. Pearls taken per loop index (active-seed denominator).
3. Stride economy (mean and population SD over active seeds).
4. Route choices at Wrecksmoor Landing and Needle Cut (departure denominator).
5. Character world telemetry fractions.
"""

from collections import defaultdict
import math
from typing import Dict, List, Optional, Any

from remnant_core.worlds.reach import (
    JUNCTIONS,
    GRAPH_CHAINS,
)

WRECKSMOOR_POS = list(JUNCTIONS["Wrecksmoor Landing"])
NEEDLE_CUT_POS = list(JUNCTIONS["Needle Cut"])

# Neighbor positions along edges from Wrecksmoor Landing:
WRECKSMOOR_TO_NEEDLE = list(GRAPH_CHAINS[0][1])
WRECKSMOOR_TO_CHAPEL = list(GRAPH_CHAINS[1][1])

# Neighbor positions along edges from Needle Cut:
NEEDLE_TO_WRECKSMOOR = list(GRAPH_CHAINS[0][-2])
NEEDLE_TO_CHAPEL = list(GRAPH_CHAINS[2][-2])
NEEDLE_TO_NET_LOFT = list(GRAPH_CHAINS[3][1])

SUPPORTED_SCHEMAS = {"1.0.0", "2.0.0"}


def validate_records_integrity(records: List[Dict[str, Any]], protocol: Dict[str, Any], total_records: Optional[int] = None) -> None:
    """Validate completeness, contiguity, and schema integrity of recorded study strides."""
    if not isinstance(records, list):
        raise ValueError("Records payload must be a list of stride records")

    if total_records is not None and (type(total_records) is not int or len(records) != total_records):
        raise ValueError(f"total_records {total_records} does not match len(records) {len(records)}")

    if not isinstance(protocol, dict):
        raise ValueError("Protocol must be a dictionary")

    schema_version = protocol.get("schema_version", "1.0.0")
    if schema_version not in SUPPORTED_SCHEMAS:
        raise ValueError(f"Unsupported schema version: {schema_version}")

    is_schema_2 = (schema_version == "2.0.0") or (bool(records) and "prev_policy_state" in records[0])

    expected_seeds = protocol.get("seeds", [])
    if not isinstance(expected_seeds, list) or len(expected_seeds) == 0:
        raise ValueError("Protocol must declare a non-empty list of seeds")

    # Seeds must be unique positive integers (not bools)
    seen_seeds_set = set()
    for s in expected_seeds:
        if isinstance(s, bool) or not isinstance(s, int):
            raise ValueError(f"Seed {s} is not a valid integer")
        if s in seen_seeds_set:
            raise ValueError(f"Duplicate seed {s} declared in protocol")
        seen_seeds_set.add(s)

    horizon = protocol.get("horizon_k_max")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError(f"Protocol horizon_k_max must be a positive integer, got {horizon}")

    loop_limit = protocol.get("loop_limit")
    if isinstance(loop_limit, bool) or not isinstance(loop_limit, int) or loop_limit <= 0:
        raise ValueError(f"Protocol loop_limit must be a positive integer, got {loop_limit}")

    if loop_limit != 54 and protocol.get("fixture_kind") != "structural_miniature":
        raise ValueError("Non-Reach limits require explicit structural_miniature fixture_kind")

    # Check records membership and structure
    seen_records_keys = set()
    seen_seeds_in_records = set()
    seed_loop_strides: Dict[int, Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    required_fields = [
        "seed", "loop", "stride", "prev_obs", "next_obs", "prev_node", "next_node",
        "action", "reward", "pearls_picked", "key_picked", "terminal", "truncated", "npc_states"
    ]
    if is_schema_2:
        required_fields.extend(["prev_policy_state", "next_policy_state"])

    for idx, r in enumerate(records):
        if not isinstance(r, dict):
            raise ValueError(f"Record {idx} is not a dictionary")

        for f in required_fields:
            if f not in r:
                raise ValueError(f"Record {idx} missing required field '{f}'")

        s = r["seed"]
        l = r["loop"]
        st = r["stride"]

        if isinstance(s, bool) or not isinstance(s, int):
            raise ValueError(f"Record {idx} seed {s} is not an integer")
        if isinstance(l, bool) or not isinstance(l, int) or l <= 0:
            raise ValueError(f"Record {idx} loop {l} must be a positive integer")
        if isinstance(st, bool) or not isinstance(st, int) or st <= 0:
            raise ValueError(f"Record {idx} stride {st} must be a positive integer")

        if s not in seen_seeds_set:
            raise ValueError(f"Foreign seed {s} found in records at index {idx} (expected one of {expected_seeds})")

        rec_key = (s, l, st)
        if rec_key in seen_records_keys:
            raise ValueError(f"Duplicate row for seed {s} loop {l} stride {st}")
        seen_records_keys.add(rec_key)

        # Finite numeric reward
        rew = r["reward"]
        if isinstance(rew, bool) or not isinstance(rew, (int, float)) or not math.isfinite(rew):
            raise ValueError(f"Record {idx} non-finite reward {rew}")

        # Action must be int
        act = r["action"]
        if isinstance(act, bool) or not isinstance(act, int) or not (0 <= act <= 3):
            raise ValueError(f"Record {idx} invalid action {act}")

        # Boolean flags
        term = r["terminal"]
        trunc = r["truncated"]
        if not isinstance(term, bool) or not isinstance(trunc, bool):
            raise ValueError(f"Record {idx} terminal/truncated must be booleans")

        if term and trunc:
            raise ValueError(f"Record {idx} cannot have both terminal=True and truncated=True")

        # Observation / node agreement
        prev_obs = r["prev_obs"]
        next_obs = r["next_obs"]
        prev_node = r["prev_node"]
        next_node = r["next_node"]

        if not isinstance(prev_obs, list) or len(prev_obs) != 2 or prev_obs[0] != prev_node:
            raise ValueError(f"Record {idx} prev_obs node does not match prev_node")
        if not isinstance(next_obs, list) or len(next_obs) != 2 or next_obs[0] != next_node:
            raise ValueError(f"Record {idx} next_obs node does not match next_node")

        for field, coordinate in (("prev_node", prev_node), ("next_node", next_node)):
            if (not isinstance(coordinate, list) or len(coordinate) != 2
                    or any(isinstance(v, bool) or not isinstance(v, int) for v in coordinate)):
                raise ValueError(f"Record {idx} invalid {field}")
        for flag in ("pearls_picked", "key_picked"):
            if not isinstance(r[flag], bool):
                raise ValueError(f"Record {idx} {flag} must be boolean")
        # Inventory check
        if not isinstance(prev_obs[1], list) or not isinstance(next_obs[1], list):
            raise ValueError(f"Record {idx} observation inventory must be a list")

        for inventory in (prev_obs[1], next_obs[1]):
            if inventory not in ([], [1]) or any(type(v) is not int for v in inventory):
                raise ValueError(f"Record {idx} inventory must be [] or [1]")

        # Policy state check under schema 2.0.0
        if is_schema_2:
            prev_ps = r["prev_policy_state"]
            next_ps = r["next_policy_state"]
            if not isinstance(prev_ps, list) or len(prev_ps) != 3:
                raise ValueError(f"Record {idx} prev_policy_state must be [pos, inv, marker]")
            if not isinstance(next_ps, list) or len(next_ps) != 3:
                raise ValueError(f"Record {idx} next_policy_state must be [pos, inv, marker]")
            if prev_ps[0] != prev_node or prev_ps[1] != prev_obs[1]:
                raise ValueError(f"Record {idx} prev_policy_state pos/inv must match prev_obs")
            if next_ps[0] != next_node or next_ps[1] != next_obs[1]:
                raise ValueError(f"Record {idx} next_policy_state pos/inv must match next_obs")
            if not isinstance(prev_ps[2], bool) or not isinstance(next_ps[2], bool):
                raise ValueError(f"Record {idx} policy state tally marker must be boolean")
            if r["pearls_picked"]:
                if not next_ps[2]:
                    raise ValueError(f"Record {idx} experienced pearls_picked but next_policy_state marker is False")
            else:
                if next_ps[2] != prev_ps[2]:
                    raise ValueError(f"Record {idx} marker changed without pearls_picked")
        # NPC states check
        npc_st = r["npc_states"]
        if not isinstance(npc_st, dict):
            raise ValueError(f"Record {idx} npc_states must be a dict")
        # In production Reach world (loop_limit == 54), require all 4 cast members
        if loop_limit == 54:
            for char_name in ["Gleaner", "Brannoch", "Dov", "Ysolde"]:
                if char_name not in npc_st or not isinstance(npc_st[char_name], dict) or not isinstance(npc_st[char_name].get("activity"), str):
                    raise ValueError(f"Record {idx} missing required NPC state for '{char_name}'")
                snapshot = npc_st[char_name]
                if (not isinstance(snapshot.get("name"), str)
                        or not isinstance(snapshot.get("goal"), str)
                        or not isinstance(snapshot.get("local_state"), dict)
                        or not isinstance(snapshot.get("home"), (list, tuple))
                        or len(snapshot["home"]) != 2
                        or any(type(v) is not int for v in snapshot["home"])):
                    raise ValueError(f"Record {idx} malformed NPC snapshot for {char_name}")

        seen_seeds_in_records.add(s)
        seed_loop_strides[s][l].append(r)

    # Check for missing seeds
    for s in expected_seeds:
        if s not in seen_seeds_in_records:
            raise ValueError(f"Expected protocol seed {s} missing from records")

    # Validate loop and stride contiguity per seed
    for s in expected_seeds:
        loops = sorted(seed_loop_strides[s].keys())
        if not loops:
            raise ValueError(f"Seed {s} has no loops recorded")

        expected_loop = 1
        has_escaped = False

        for l in loops:
            if has_escaped:
                raise ValueError(f"Seed {s} recorded records in loop {l} after escaping")
            if l != expected_loop:
                raise ValueError(f"Seed {s} non-contiguous loop {l} (expected {expected_loop})")
            if l > horizon:
                raise ValueError(f"Seed {s} loop {l} exceeds declared horizon {horizon}")
            expected_loop += 1

            recs = seed_loop_strides[s][l]
            for before, after in zip(recs, recs[1:]):
                if before["next_obs"] != after["prev_obs"]:
                    raise ValueError(f"Seed {s} loop {l} discontinuous observations")
            # Strides must be contiguous 1..N
            for stride_idx, r in enumerate(recs, 1):
                if r["stride"] != stride_idx:
                    raise ValueError(f"Seed {s} loop {l} non-contiguous stride {r['stride']} (expected {stride_idx})")

                # Only the final stride may be terminal or truncated
                if stride_idx < len(recs):
                    if r["terminal"] or r["truncated"]:
                        raise ValueError(f"Seed {s} loop {l} stride {stride_idx} had premature terminal/truncated flag")

            final_r = recs[-1]
            if not (final_r["terminal"] or final_r["truncated"]):
                raise ValueError(f"Seed {s} loop {l} final stride neither terminal nor truncated")

            # Check loop limit consistency
            if loop_limit is not None:
                if len(recs) > loop_limit:
                    raise ValueError(f"Seed {s} loop {l} length {len(recs)} exceeds loop limit {loop_limit}")
                if final_r["truncated"] and len(recs) != loop_limit:
                    raise ValueError(f"Seed {s} loop {l} truncated at stride {len(recs)} instead of declared limit {loop_limit}")

            if final_r["terminal"]:
                has_escaped = True

        # If not escaped, must have completed horizon exactly
        if not has_escaped and len(loops) != horizon:
            raise ValueError(f"Censored seed {s} ran {len(loops)} loops instead of declared horizon {horizon}")


def compute_study_outcomes(
    records_payload: Dict[str, Any],
    run_summaries: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute all analytical metrics from records and protocol."""
    if not isinstance(records_payload, dict):
        raise ValueError("records_payload must be a dictionary")

    schema_ver = records_payload.get("schema_version")
    if schema_ver not in SUPPORTED_SCHEMAS:
        raise ValueError(f"Unsupported schema version: {schema_ver} (supported: {SUPPORTED_SCHEMAS})")

    records = records_payload.get("records", [])
    protocol = records_payload.get("protocol", {})
    total_records = records_payload.get("total_records")

    # Validate records integrity strictly
    validate_records_integrity(records, protocol, total_records=total_records)

    seeds = protocol["seeds"]
    horizon = protocol["horizon_k_max"]
    total_seeds = len(seeds)

    # Group records by seed and loop
    seed_loop_records: Dict[int, Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        seed_loop_records[r["seed"]][r["loop"]].append(r)

    # Derive per-seed outcomes strictly from records
    seed_escape_loop: Dict[int, Optional[int]] = {}
    seed_total_loops: Dict[int, int] = {}
    seed_total_real: Dict[int, int] = {}
    seed_total_q: Dict[int, int] = {}

    for s in seeds:
        s_loops = sorted(seed_loop_records[s].keys())
        seed_total_loops[s] = max(s_loops) if s_loops else 0
        esc_loop = None
        tot_real = 0
        for l in s_loops:
            l_recs = seed_loop_records[s][l]
            tot_real += len(l_recs)
            if any(r["terminal"] for r in l_recs):
                esc_loop = l
                break
        seed_escape_loop[s] = esc_loop
        seed_total_real[s] = tot_real
        seed_total_q[s] = tot_real

    # Validate optional run_summaries if provided
    if run_summaries is not None:
        if not isinstance(run_summaries, list):
            raise ValueError("run_summaries must be a list of dictionaries")
        seen_summary_seeds = set()
        for s_summary in run_summaries:
            if not isinstance(s_summary, dict):
                raise ValueError("Each run summary must be a dictionary")
            s = s_summary.get("seed")
            if type(s) is not int:
                raise ValueError("Run summary seed must be integer")
            if s in seen_summary_seeds:
                raise ValueError(f"Duplicate seed {s} in run_summaries")
            seen_summary_seeds.add(s)
            if s not in seed_escape_loop:
                raise ValueError(f"Foreign seed {s} in run_summaries")

            if s_summary.get("escaped") is not (seed_escape_loop[s] is not None):
                raise ValueError(f"Discrepancy in escaped for seed {s}")
            expected_status = "escaped" if seed_escape_loop[s] is not None else "loop_cap_reached"
            if s_summary.get("status") != expected_status or s_summary.get("max_loops") != horizon:
                raise ValueError(f"Discrepancy in status/max_loops for seed {s}")
            rows_for_seed = [row for loop_rows in seed_loop_records[s].values() for row in loop_rows]
            q_state_actions = set()
            model_state_actions = set()
            model_transitions = set()
            cumulative = 0
            for loop in sorted(seed_loop_records[s]):
                rows = seed_loop_records[s][loop]
                for row in rows:
                    before = (tuple(row["prev_node"]), tuple(row["prev_obs"][1]))
                    after = (tuple(row["next_node"]), tuple(row["next_obs"][1]))
                    model_state_actions.add((before, row["action"]))
                    model_transitions.add((before, row["action"], after))
                    if "prev_policy_state" in row:
                        q_s = (tuple(row["prev_policy_state"][0]), tuple(row["prev_policy_state"][1]), bool(row["prev_policy_state"][2]))
                    else:
                        q_s = before
                    q_state_actions.add((q_s, row["action"]))
                cumulative += len(rows)
                supplied = s_summary.get("loop_records", [])
                if len(supplied) < loop or not isinstance(supplied[loop - 1], dict):
                    raise ValueError(f"Missing loop summary for seed {s}")
                if (supplied[loop - 1].get("q_table_size") != len(q_state_actions)
                        or supplied[loop - 1].get("memory_size") != cumulative):
                    raise ValueError(f"Discrepancy in loop store sizes for seed {s}")
            expected_sizes = {"state_actions": len(model_state_actions), "transitions": len(model_transitions)}
            if (s_summary.get("model_size") != expected_sizes
                    or s_summary.get("q_table_size") != len(q_state_actions)
                    or s_summary.get("memory_entries") != len(rows_for_seed)):
                raise ValueError(f"Discrepancy in store sizes for seed {s}")
            expected_loops = []
            for loop in sorted(seed_loop_records[s]):
                rows = seed_loop_records[s][loop]
                expected_loops.append({"loop": loop, "real_steps": len(rows),
                    "cumulative_reward": round(sum(row["reward"] for row in rows), 4),
                    "escaped": rows[-1]["terminal"]})
            supplied_loops = s_summary.get("loop_records")
            if not isinstance(supplied_loops, list) or len(supplied_loops) != len(expected_loops):
                raise ValueError(f"Discrepancy in loop_records for seed {s}")
            for expected, supplied in zip(expected_loops, supplied_loops):
                if any(supplied.get(k) != v for k, v in expected.items()):
                    raise ValueError(f"Discrepancy in loop summary for seed {s}")
            if seed_escape_loop[s] != s_summary.get("escape_loop"):
                raise ValueError(f"Discrepancy in escape_loop for seed {s}: records={seed_escape_loop[s]} vs summary={s_summary.get('escape_loop')}")
            if seed_total_loops[s] != s_summary.get("total_loops_run"):
                raise ValueError(f"Discrepancy in total_loops_run for seed {s}: records={seed_total_loops[s]} vs summary={s_summary.get('total_loops_run')}")
            if seed_total_real[s] != s_summary.get("total_real_interactions"):
                raise ValueError(f"Discrepancy in total_real_interactions for seed {s}: records={seed_total_real[s]} vs summary={s_summary.get('total_real_interactions')}")
            if seed_total_q[s] != s_summary.get("total_q_updates"):
                raise ValueError(f"Discrepancy in total_q_updates for seed {s}: records={seed_total_q[s]} vs summary={s_summary.get('total_q_updates')}")

        if seen_summary_seeds != set(seeds):
            raise ValueError("run_summaries seeds do not match protocol seeds exactly")

    escaped_seeds = [s for s in seeds if seed_escape_loop[s] is not None]
    censored_seeds = [s for s in seeds if seed_escape_loop[s] is None]

    escape_rate = len(escaped_seeds) / total_seeds if total_seeds > 0 else 0.0

    # Escape distribution with string keys
    escape_distribution = {
        str(s): seed_escape_loop[s] for s in escaped_seeds
    }
    escape_loops_list = [seed_escape_loop[s] for s in escaped_seeds]
    mean_escape_loops = (sum(escape_loops_list) / len(escape_loops_list)) if escape_loops_list else None
    min_escape_loop = min(escape_loops_list) if escape_loops_list else None
    max_escape_loop = max(escape_loops_list) if escape_loops_list else None

    all_loops_run = [seed_total_loops[s] for s in seeds]
    max_observed_loop = max(all_loops_run) if all_loops_run else 0

    # Per-loop metrics using string keys
    pearls_taken_per_loop: Dict[str, Dict[str, Any]] = {}
    stride_economy_per_loop: Dict[str, Dict[str, Any]] = {}
    route_choices_wrecksmoor: Dict[str, Dict[str, Any]] = {}
    route_choices_needle_cut: Dict[str, Dict[str, Any]] = {}
    character_telemetry_per_loop: Dict[str, Dict[str, Any]] = {}

    for k in range(1, max_observed_loop + 1):
        k_str = str(k)
        active_seeds_k = [s for s in seeds if seed_total_loops[s] >= k]
        n_active = len(active_seeds_k)

        pearl_pick_count = 0
        strides_k: List[int] = []

        wm_to_needle = 0
        wm_to_chapel = 0

        nc_to_netloft = 0
        nc_to_wrecksmoor = 0
        nc_to_chapel = 0

        char_activity_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        total_strides_in_loop_k = 0

        for s in active_seeds_k:
            recs = seed_loop_records[s].get(k, [])
            strides_k.append(len(recs))
            total_strides_in_loop_k += len(recs)

            if any(r["pearls_picked"] for r in recs):
                pearl_pick_count += 1

            for r in recs:
                prev_n = r["prev_node"]
                next_n = r["next_node"]

                if prev_n != next_n:
                    if prev_n == WRECKSMOOR_POS:
                        if next_n == WRECKSMOOR_TO_NEEDLE:
                            wm_to_needle += 1
                        elif next_n == WRECKSMOOR_TO_CHAPEL:
                            wm_to_chapel += 1

                    elif prev_n == NEEDLE_CUT_POS:
                        if next_n == NEEDLE_TO_NET_LOFT:
                            nc_to_netloft += 1
                        elif next_n == NEEDLE_TO_WRECKSMOOR:
                            nc_to_wrecksmoor += 1
                        elif next_n == NEEDLE_TO_CHAPEL:
                            nc_to_chapel += 1

                npc_st = r.get("npc_states", {})
                for char_name, c_info in npc_st.items():
                    act = c_info.get("activity", "unknown")
                    char_activity_counts[char_name][act] += 1

        pearls_taken_per_loop[k_str] = {
            "loop": k,
            "active_seeds": n_active,
            "pearls_taken_count": pearl_pick_count,
            "pearls_taken_rate": round(pearl_pick_count / n_active, 4) if n_active > 0 else 0.0,
        }

        if n_active > 0 and strides_k:
            mean_strides = sum(strides_k) / n_active
            variance = sum((x - mean_strides) ** 2 for x in strides_k) / n_active
            pop_sd_strides = math.sqrt(variance)
        else:
            mean_strides = 0.0
            pop_sd_strides = 0.0

        stride_economy_per_loop[k_str] = {
            "loop": k,
            "active_seeds": n_active,
            "mean_strides": round(mean_strides, 4),
            "pop_sd_strides": round(pop_sd_strides, 4),
        }

        wm_total_dep = wm_to_needle + wm_to_chapel
        route_choices_wrecksmoor[k_str] = {
            "loop": k,
            "active_seeds": n_active,
            "departures_to_needle_cut": wm_to_needle,
            "departures_to_chapel": wm_to_chapel,
            "total_departures": wm_total_dep,
            "departures_per_active_seed": round(wm_total_dep / n_active, 4) if n_active > 0 else 0.0,
            "share_to_needle_cut": round(wm_to_needle / wm_total_dep, 4) if wm_total_dep > 0 else None,
            "share_to_chapel": round(wm_to_chapel / wm_total_dep, 4) if wm_total_dep > 0 else None,
        }

        nc_total_dep = nc_to_netloft + nc_to_wrecksmoor + nc_to_chapel
        route_choices_needle_cut[k_str] = {
            "loop": k,
            "active_seeds": n_active,
            "departures_to_net_loft": nc_to_netloft,
            "departures_to_wrecksmoor": nc_to_wrecksmoor,
            "departures_to_chapel": nc_to_chapel,
            "total_departures": nc_total_dep,
            "departures_per_active_seed": round(nc_total_dep / n_active, 4) if n_active > 0 else 0.0,
            "share_to_net_loft": round(nc_to_netloft / nc_total_dep, 4) if nc_total_dep > 0 else None,
            "share_to_wrecksmoor": round(nc_to_wrecksmoor / nc_total_dep, 4) if nc_total_dep > 0 else None,
            "share_to_chapel": round(nc_to_chapel / nc_total_dep, 4) if nc_total_dep > 0 else None,
        }

        telemetry_k: Dict[str, Dict[str, Any]] = {}
        for char_name, acts in char_activity_counts.items():
            telemetry_k[char_name] = {
                act: {
                    "count": cnt,
                    "fraction": round(cnt / total_strides_in_loop_k, 4) if total_strides_in_loop_k > 0 else 0.0,
                }
                for act, cnt in acts.items()
            }
        character_telemetry_per_loop[k_str] = {
            "loop": k,
            "active_seeds": n_active,
            "total_strides": total_strides_in_loop_k,
            "characters": telemetry_k,
        }

    node_visits_early: Dict[str, int] = defaultdict(int)
    node_visits_late: Dict[str, int] = defaultdict(int)
    total_strides_early = 0
    total_strides_late = 0

    for s in seeds:
        s_total = seed_total_loops[s]
        early_loops = set(range(1, min(10, s_total) + 1))
        late_loops = set(range(max(1, s_total - 9), s_total + 1))

        for l in sorted(seed_loop_records[s].keys()):
            recs = seed_loop_records[s][l]
            for r in recs:
                node_str = f"{r['next_node'][0]},{r['next_node'][1]}"
                if l in early_loops:
                    node_visits_early[node_str] += 1
                    total_strides_early += 1
                if l in late_loops:
                    node_visits_late[node_str] += 1
                    total_strides_late += 1

    return {
        "schema_version": schema_ver,
        "protocol": protocol,
        "summary": {
            "total_seeds": total_seeds,
            "escaped_count": len(escaped_seeds),
            "censored_count": len(censored_seeds),
            "escape_rate": round(escape_rate, 4),
            "mean_escape_loops": round(mean_escape_loops, 4) if mean_escape_loops is not None else None,
            "min_escape_loop": min_escape_loop,
            "max_escape_loop": max_escape_loop,
            "escaped_seeds": sorted(escaped_seeds),
            "censored_seeds": sorted(censored_seeds),
            "escape_distribution": escape_distribution,
        },
        "per_loop": {
            "pearls_taken": pearls_taken_per_loop,
            "stride_economy": stride_economy_per_loop,
            "route_choices_wrecksmoor": route_choices_wrecksmoor,
            "route_choices_needle_cut": route_choices_needle_cut,
            "character_telemetry": character_telemetry_per_loop,
        },
        "visitation": {
            "total_strides_early": total_strides_early,
            "total_strides_late": total_strides_late,
            "early_node_visits": dict(node_visits_early),
            "late_node_visits": dict(node_visits_late),
        },
    }
