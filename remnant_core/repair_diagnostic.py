"""Resource-measured matched diagnostic and operational learned-escape screen.

Compares:
- Baseline arm: baseline_hidden_pearl_v1 -> results_ctem_baseline_v1/
- Route B control arm: tally_current_tide_pearl_v1 -> results_ctem_tally_control_v1/
- Exploration arm: cross_tide_exploration_memory_v1 -> results_cross_tide_exploration_v1/

Performs pre-training control (epsilon=0, eval_tides, no learning, no exploration bonus)
and post-training frozen-policy evaluation (epsilon=0, eval_tides, no learning, no exploration bonus,
restored RNG) for each seed. Saves full evaluation trajectories into evaluation_trajectories.json
and evaluation summaries into evaluations.json.
Replays all evaluation trajectories against physical_transition.
Publishes atomically using safe publication.
"""

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import resource
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from remnant_core.agent import QAgent
from remnant_core.memory import EpisodicMemory
from remnant_core.model import WorldModel
from remnant_core.exploration_memory import CrossTideExplorationMemory
from remnant_core.worlds.reach import (
    ReachWorld,
    REACH_LOOP_LIMIT,
    STRIDE_COST,
    PEARLS_REWARD,
    LAMP_KEY_REWARD,
    TIDEGATE_REWARD,
    ITEM_LAMP_KEY,
    ITEM_PEARLS,
    physical_transition,
)
from remnant_core.study import (
    run_reach_seed,
    write_study_manifest,
    _publish_new_directory,
    _validate_run_parameters,
    compute_file_sha256,
    get_current_git_commit,
)
from remnant_core.analysis import compute_study_outcomes, validate_records_integrity


def get_peak_rss_bytes() -> int:
    """Return peak resident memory in bytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On macOS, ru_maxrss is in bytes; on Linux, in kilobytes.
    if sys.platform != "darwin":
        rss *= 1024
    return int(rss)


def evaluate_policy(
    world: ReachWorld,
    agent: QAgent,
    condition: str,
    action_rng: random.Random,
    eval_tides: int = 2,
    max_strides_per_tide: int = REACH_LOOP_LIMIT,
) -> Dict[str, Any]:
    """Evaluate a frozen policy without learning from dawn for eval_tides (exploration bonus excluded)."""
    use_tally = (condition in ("tally_current_tide_pearl_v1", "cross_tide_exploration_memory_v1"))
    tide_results = []
    all_eval_records = []

    for tide_idx in range(1, eval_tides + 1):
        obs = world.reset()
        tally_marker = False
        tide_steps = 0
        tide_reward = 0.0
        escaped = False

        while tide_steps < max_strides_per_tide:
            if use_tally:
                policy_state = (obs[0], obs[1], tally_marker)
            else:
                policy_state = obs

            # Greedy evaluation: epsilon=0 and training=False (exploration bonus excluded)
            old_eps = agent.eps
            agent.eps = 0.0
            a = agent.act(policy_state, action_rng, training=False)
            agent.eps = old_eps

            obs_next, r, terminal, truncated = world.step(a)
            tide_steps += 1
            tide_reward += r

            tel = getattr(world, "last_telemetry", None)
            pearls_picked = bool(tel.get("pearls_picked", False)) if tel is not None else False
            next_tally = True if pearls_picked else tally_marker

            if use_tally:
                next_policy_state = (obs_next[0], obs_next[1], next_tally)
            else:
                next_policy_state = obs_next

            rec = {
                "tide": tide_idx,
                "stride": tide_steps,
                "prev_obs": [list(obs[0]), sorted(list(obs[1]))],
                "next_obs": [list(obs_next[0]), sorted(list(obs_next[1]))],
                "prev_node": list(obs[0]),
                "next_node": list(obs_next[0]),
                "action": a,
                "reward": round(r, 4),
                "pearls_picked": pearls_picked,
                "key_picked": bool(tel.get("key_picked", False)) if tel is not None else False,
                "terminal": terminal,
                "truncated": truncated,
                "npc_states": copy.deepcopy(tel["npc_states"]) if tel is not None else {},
            }
            if use_tally:
                rec["prev_policy_state"] = [list(policy_state[0]), sorted(list(policy_state[1])), bool(policy_state[2])]
                rec["next_policy_state"] = [list(next_policy_state[0]), sorted(list(next_policy_state[1])), bool(next_policy_state[2])]

            all_eval_records.append(rec)

            obs = obs_next
            tally_marker = next_tally

            if terminal:
                escaped = True
                break
            if truncated:
                break

        tide_results.append({
            "tide": tide_idx,
            "strides": tide_steps,
            "cumulative_reward": round(tide_reward, 4),
            "escaped": escaped,
        })

    return {
        "tides": tide_results,
        "escaped_all_tides": all(t["escaped"] for t in tide_results),
        "records": all_eval_records,
    }


def replay_evaluation_trajectory(records: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Physically replay an evaluation trajectory against physical_transition to verify legality."""
    from remnant_core.worlds.reach import START_POS
    curr_pos = START_POS
    has_key = False
    pearls_collected = False

    for idx, r in enumerate(records):
        st = r["stride"]
        if st == 1:
            curr_pos = START_POS
            has_key = False
            pearls_collected = False

        a = r["action"]
        expected_inv = frozenset({1}) if has_key else frozenset()
        expected_s = (curr_pos, expected_inv)
        actual_prev_obs = (tuple(r["prev_obs"][0]), frozenset(r["prev_obs"][1]))
        if actual_prev_obs != expected_s:
            return False, f"Record {idx} prev_obs mismatch: expected {expected_s}, got {actual_prev_obs}"

        phys_state = (curr_pos, has_key, pearls_collected)
        next_phys, rew, term, invalid_reason, pearls_picked, key_picked = physical_transition(
            phys_state, a
        )

        next_pos, next_has_key, next_pearls = next_phys
        expected_next_inv = frozenset({1}) if next_has_key else frozenset()
        expected_next_obs = (next_pos, expected_next_inv)
        actual_next_obs = (tuple(r["next_obs"][0]), frozenset(r["next_obs"][1]))
        if actual_next_obs != expected_next_obs:
            return False, f"Record {idx} next_obs mismatch: expected {expected_next_obs}, got {actual_next_obs}"

        if round(rew, 4) != round(r["reward"], 4):
            return False, f"Record {idx} reward mismatch: expected {rew}, got {r['reward']}"

        if term != r["terminal"]:
            return False, f"Record {idx} terminal mismatch: expected {term}, got {r['terminal']}"

        if pearls_picked != r["pearls_picked"]:
            return False, f"Record {idx} pearls_picked mismatch"

        if key_picked != r["key_picked"]:
            return False, f"Record {idx} key_picked mismatch"

        curr_pos = next_pos
        has_key = next_has_key
        pearls_collected = next_pearls

    return True, None


class DiagnosticArmContext:
    """Manages execution and staged publication for one diagnostic arm."""

    def __init__(
        self,
        output_dir: str,
        condition: str,
        seeds: List[int],
        horizon: int,
        eval_tides: int,
        command_str: str,
        exploration_coefficient: float = 1.0,
    ):
        self.output_dir = output_dir
        self.condition = condition
        self.seeds = seeds
        self.horizon = horizon
        self.eval_tides = eval_tides
        self.command_str = command_str
        self.exploration_coefficient = exploration_coefficient

        if os.path.lexists(output_dir):
            raise FileExistsError(f"Destination directory already exists: '{output_dir}'")

        parent_dir = os.path.dirname(os.path.abspath(output_dir))
        self.staging_dir = os.path.join(parent_dir, f".staging_{os.path.basename(output_dir)}_{uuid.uuid4().hex[:8]}")
        os.makedirs(self.staging_dir, exist_ok=False)

        self.all_records: List[Dict[str, Any]] = []
        self.run_summaries: List[Dict[str, Any]] = []
        self.learners: Dict[str, Any] = {}
        self.evaluations: Dict[str, Any] = {}
        self.evaluation_trajectories: Dict[str, Any] = {}
        self.completed_seeds: List[int] = []
        self.schema_ver = "2.0.0" if condition in ("tally_current_tide_pearl_v1", "cross_tide_exploration_memory_v1") else "1.0.0"
        self.start_time = time.perf_counter()
        self.wall_clock = 0.0

    def run_seed(self, s: int) -> None:
        if s in self.completed_seeds:
            return

        # Pre-training control:
        world_pre = ReachWorld(max_steps=REACH_LOOP_LIMIT)
        mem_pre = EpisodicMemory(seed=s * 1000 + 6)
        agent_pre = QAgent(memory=mem_pre)
        rng_pre = random.Random(s * 1000 + 1)

        pre_eval = evaluate_policy(
            world=world_pre,
            agent=agent_pre,
            condition=self.condition,
            action_rng=rng_pre,
            eval_tides=self.eval_tides,
        )
        pre_replay_ok, pre_replay_err = replay_evaluation_trajectory(pre_eval["records"])
        if not pre_replay_ok:
            raise RuntimeError(f"Seed {s} pre-training trajectory replay failed: {pre_replay_err}")

        # Training:
        res, seed_records, learner_state = run_reach_seed(
            seed=s,
            max_loops=self.horizon,
            condition=self.condition,
            exploration_coefficient=self.exploration_coefficient,
        )
        self.all_records.extend(seed_records)
        self.run_summaries.append(res)
        self.learners[str(s)] = learner_state

        # Post-training evaluation:
        restored_agent = QAgent.from_dict(learner_state["agent"])
        action_rng_post = random.Random()
        st_act = learner_state["final_action_rng_state"]
        action_rng_post.setstate((st_act[0], tuple(st_act[1]), st_act[2]))

        world_post = ReachWorld(max_steps=REACH_LOOP_LIMIT)
        post_eval = evaluate_policy(
            world=world_post,
            agent=restored_agent,
            condition=self.condition,
            action_rng=action_rng_post,
            eval_tides=self.eval_tides,
        )
        post_replay_ok, post_replay_err = replay_evaluation_trajectory(post_eval["records"])
        if not post_replay_ok:
            raise RuntimeError(f"Seed {s} post-training trajectory replay failed: {post_replay_err}")

        # Screen outcome for this seed:
        post_escaped_both = post_eval["escaped_all_tides"]
        pre_escaped_any = any(t["escaped"] for t in pre_eval["tides"])
        seed_screen_positive = post_escaped_both and not pre_escaped_any

        self.evaluations[str(s)] = {
            "seed": s,
            "pre_training_control": {
                "tides": pre_eval["tides"],
                "records_count": len(pre_eval["records"]),
                "replay_verified": pre_replay_ok,
            },
            "post_training_test": {
                "tides": post_eval["tides"],
                "records_count": len(post_eval["records"]),
                "replay_verified": post_replay_ok,
            },
            "post_escaped_both": post_escaped_both,
            "pre_escaped_any": pre_escaped_any,
            "screen_positive": seed_screen_positive,
        }

        self.evaluation_trajectories[str(s)] = {
            "seed": s,
            "pre_training_control": {
                "tides": pre_eval["tides"],
                "records": pre_eval["records"],
            },
            "post_training_test": {
                "tides": post_eval["tides"],
                "records": post_eval["records"],
            },
        }

        self.completed_seeds.append(s)

    def finalize_staging(self) -> None:
        self.wall_clock = time.perf_counter() - self.start_time
        protocol = {
            "seeds": self.completed_seeds,
            "horizon_k_max": self.horizon,
            "loop_limit": REACH_LOOP_LIMIT,
            "condition": self.condition,
            "schema_version": self.schema_ver,
            "eval_tides": self.eval_tides,
        }
        if self.condition == "cross_tide_exploration_memory_v1":
            protocol["exploration_coefficient"] = self.exploration_coefficient

        records_payload = {
            "schema_version": self.schema_ver,
            "condition": self.condition,
            "protocol": protocol,
            "total_records": len(self.all_records),
            "records": self.all_records,
        }

        learners_payload = {
            "schema_version": self.schema_ver,
            "condition": self.condition,
            "seeds": self.completed_seeds,
            "learners": self.learners,
        }

        outcomes_payload = compute_study_outcomes(records_payload, self.run_summaries)

        evaluations_payload = {
            "schema_version": self.schema_ver,
            "condition": self.condition,
            "protocol": protocol,
            "evaluations": self.evaluations,
            "all_seeds_screen_positive": all(ev["screen_positive"] for ev in self.evaluations.values()) if self.evaluations else False,
        }

        with open(os.path.join(self.staging_dir, "records.json"), "w", encoding="utf-8") as fp:
            json.dump(records_payload, fp, indent=2)

        with open(os.path.join(self.staging_dir, "learners.json"), "w", encoding="utf-8") as fp:
            json.dump(learners_payload, fp, indent=2)

        with open(os.path.join(self.staging_dir, "outcomes.json"), "w", encoding="utf-8") as fp:
            json.dump(outcomes_payload, fp, indent=2)

        with open(os.path.join(self.staging_dir, "evaluations.json"), "w", encoding="utf-8") as fp:
            json.dump(evaluations_payload, fp, indent=2)

        with open(os.path.join(self.staging_dir, "evaluation_trajectories.json"), "w", encoding="utf-8") as fp:
            json.dump(self.evaluation_trajectories, fp, indent=2)

        write_study_manifest(
            output_dir=self.staging_dir,
            protocol=protocol,
            wall_clock_seconds=self.wall_clock,
            command_str=self.command_str,
        )

        expected_files = [
            "records.json",
            "outcomes.json",
            "learners.json",
            "evaluations.json",
            "evaluation_trajectories.json",
            "manifest.json",
        ]
        for fname in expected_files:
            fpath = os.path.join(self.staging_dir, fname)
            if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
                raise RuntimeError(f"Missing artifact in staging: {fname}")

        # Replay every saved evaluation trajectory directly from disk
        with open(os.path.join(self.staging_dir, "evaluation_trajectories.json"), "r", encoding="utf-8") as fp:
            saved_traj = json.load(fp)
        for s_str, s_data in saved_traj.items():
            ok_pre, err_pre = replay_evaluation_trajectory(s_data["pre_training_control"]["records"])
            if not ok_pre:
                raise RuntimeError(f"Disk trajectory replay failed for seed {s_str} pre-training: {err_pre}")
            ok_post, err_post = replay_evaluation_trajectory(s_data["post_training_test"]["records"])
            if not ok_post:
                raise RuntimeError(f"Disk trajectory replay failed for seed {s_str} post-training: {err_post}")

    def publish(self) -> Dict[str, Any]:
        if os.path.lexists(self.output_dir):
            raise FileExistsError(f"Target destination appeared unexpectedly: '{self.output_dir}'")
        _publish_new_directory(self.staging_dir, self.output_dir)

        expected_files = [
            "records.json",
            "outcomes.json",
            "learners.json",
            "evaluations.json",
            "evaluation_trajectories.json",
            "manifest.json",
        ]
        artifact_sizes = {
            fname: os.path.getsize(os.path.join(self.output_dir, fname))
            for fname in expected_files
        }
        artifact_hashes = {
            fname: compute_file_sha256(os.path.join(self.output_dir, fname))
            for fname in expected_files
        }
        peak_rss = get_peak_rss_bytes()

        return {
            "condition": self.condition,
            "schema_version": self.schema_ver,
            "output_dir": self.output_dir,
            "seeds": self.completed_seeds,
            "wall_clock_seconds": self.wall_clock,
            "peak_rss_bytes": peak_rss,
            "artifact_sizes": artifact_sizes,
            "artifact_hashes": artifact_hashes,
            "evaluations": self.evaluations,
            "all_seeds_screen_positive": all(ev["screen_positive"] for ev in self.evaluations.values()) if self.evaluations else False,
        }

    def cleanup(self) -> None:
        if os.path.exists(self.staging_dir):
            shutil.rmtree(self.staging_dir, ignore_errors=True)


def run_diagnostic_arm(
    seeds: List[int],
    horizon: int,
    condition: str,
    eval_tides: int,
    output_dir: str,
    command_str: str,
    exploration_coefficient: float = 1.0,
    measure_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run one experimental arm with training, pre/post evaluation, and safe atomic publication."""
    _validate_run_parameters(seeds, horizon)
    ctx = DiagnosticArmContext(
        output_dir=output_dir,
        condition=condition,
        seeds=seeds,
        horizon=horizon,
        eval_tides=eval_tides,
        command_str=command_str,
        exploration_coefficient=exploration_coefficient,
    )
    try:
        for idx, s in enumerate(seeds):
            ctx.run_seed(s)
            if measure_callback is not None:
                measure_callback(idx + 1, len(seeds))
        ctx.finalize_staging()
        return ctx.publish()
    except Exception:
        ctx.cleanup()
        raise


def parse_diagnostic_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resource-measured matched diagnostic for Remnant.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[301, 302, 303], help="Seeds to evaluate")
    parser.add_argument("--horizon", type=int, default=100, help="Training horizon in tides")
    parser.add_argument("--eval-tides", type=int, default=2, help="Evaluation tides per policy")
    parser.add_argument("--baseline-output-dir", type=str, default="results_ctem_baseline_v1", help="Baseline output directory")
    parser.add_argument("--route-b-output-dir", type=str, default="results_ctem_tally_control_v1", help="Route B control output directory")
    parser.add_argument("--exploration-output-dir", type=str, default="results_cross_tide_exploration_v1", help="Exploration arm output directory")
    parser.add_argument("--exploration-coefficient", type=float, default=1.0, help="Exploration bonus coefficient")
    parser.add_argument("--exploration-only", action="store_true", default=False, help="Run only the exploration arm")
    return parser.parse_args(args)


if __name__ == "__main__":
    parsed = parse_diagnostic_args(sys.argv[1:])
    cmd = "python -m remnant_core.repair_diagnostic " + " ".join(sys.argv[1:])

    # Determine arm configurations: (output_dir, condition, exploration_coefficient)
    if parsed.exploration_only:
        arm_configs = [
            (parsed.exploration_output_dir, "cross_tide_exploration_memory_v1", parsed.exploration_coefficient),
        ]
    else:
        arm_configs = [
            (parsed.baseline_output_dir, "baseline_hidden_pearl_v1", 0.0),
            (parsed.route_b_output_dir, "tally_current_tide_pearl_v1", 0.0),
            (parsed.exploration_output_dir, "cross_tide_exploration_memory_v1", parsed.exploration_coefficient),
        ]

    # Verify destination directories do not exist
    for out, _, _ in arm_configs:
        if os.path.lexists(out):
            raise FileExistsError(f"Destination directory already exists: '{out}'")

    # Step 1: Preflight calibration check on seed 301 in temporary staging
    first_seed = parsed.seeds[0]
    remaining_seeds = parsed.seeds[1:]
    print(f"Preflight calibration check on seed {first_seed} across {len(arm_configs)} arm(s)...")

    contexts: List[DiagnosticArmContext] = []
    try:
        t0 = time.perf_counter()
        free_disk_before = shutil.disk_usage(Path.cwd()).free

        for out_dir, cond, coeff in arm_configs:
            ctx = DiagnosticArmContext(
                output_dir=out_dir,
                condition=cond,
                seeds=parsed.seeds,
                horizon=parsed.horizon,
                eval_tides=parsed.eval_tides,
                command_str=cmd,
                exploration_coefficient=coeff,
            )
            contexts.append(ctx)
            # Run seed 301 into staging
            ctx.run_seed(first_seed)

        calib_time = time.perf_counter() - t0
        calib_rss = get_peak_rss_bytes()
        calib_bytes = sum(
            sum(os.path.getsize(os.path.join(c.staging_dir, f)) for f in os.listdir(c.staging_dir) if os.path.isfile(os.path.join(c.staging_dir, f)))
            for c in contexts
        )
        free_disk_after = shutil.disk_usage(Path.cwd()).free

        print(f"Preflight results (seed {first_seed}, horizon {parsed.horizon}):")
        print(f"  Wall time: {calib_time:.2f}s")
        print(f"  Peak RSS: {calib_rss / (1024 * 1024):.2f} MB")
        print(f"  Artifact bytes (seed {first_seed}): {calib_bytes / (1024 * 1024):.2f} MB")
        print(f"  Free disk: {free_disk_after / (1024 * 1024 * 1024):.2f} GB")

        # Headroom decision:
        projected_total_bytes = calib_bytes * len(parsed.seeds) * 2
        if free_disk_after < 500 * 1024 * 1024 or free_disk_after < projected_total_bytes:
            raise RuntimeError(f"Resource bound: insufficient disk headroom ({free_disk_after} bytes free)")
        if calib_time > 300.0:
            raise RuntimeError(f"Resource bound: preflight exceeded time bound ({calib_time:.2f}s)")

        print("HEADROOM DECISION: Sufficient headroom confirmed. Continuing remaining seeds in staging.")

        # Step 2: Continue remaining seeds directly into the same staging roots without rerunning seed 301
        for s in remaining_seeds:
            print(f"Running seed {s} across {len(contexts)} arm(s)...")
            for ctx in contexts:
                ctx.run_seed(s)

        # Step 3: Finalize staging and validate all arms
        print("Finalizing and validating staging directories...")
        for ctx in contexts:
            ctx.finalize_staging()

        # Step 4: Publish all arms atomically once validation passes
        results = []
        for ctx in contexts:
            res = ctx.publish()
            results.append(res)
            print(f"Published {res['condition']} -> {res['output_dir']}")

        print("\n=== Matched Diagnostic Execution Complete ===")
        for res in results:
            print(f"Arm: {res['condition']} ({res['output_dir']})")
            print(f"  Wall clock: {res['wall_clock_seconds']:.2f}s, Peak RSS: {res['peak_rss_bytes']} bytes")
            print(f"  Operational screen positive: {res['all_seeds_screen_positive']}")

    except Exception:
        for ctx in contexts:
            ctx.cleanup()
        raise
