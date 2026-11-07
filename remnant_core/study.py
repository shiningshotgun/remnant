"""Study runner for the Reach of Vesper.

Executes the 30-seed empirical study using LoopRunner on ReachWorld.
Captures per-stride telemetry via on_learn_callback without perturbing learning.
Saves:
- results/records.json
- results/outcomes.json
- results/learners.json
- results/manifest.json
"""

import argparse
import ctypes
import importlib.metadata
from pathlib import Path
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any

from remnant_core.runner import LoopRunner
from remnant_core.worlds.reach import (ReachWorld, REACH_LOOP_LIMIT, STRIDE_COST,
    PEARLS_REWARD, LAMP_KEY_REWARD, TIDEGATE_REWARD)
from remnant_core.study_config import (
    STUDY_SEEDS,
    STUDY_HORIZON_K_MAX,
    SCHEMA_VERSION,
    STUDY_LR,
    STUDY_GAMMA,
    STUDY_EPS,
    STUDY_MEMORY_BLEND,
    STUDY_MEMORY_TEMPERATURE,
    STUDY_MEMORY_LOOKUP_SAMPLE,
)


def _validate_run_parameters(seeds, max_loops):
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be a nonempty list of unique integers")
    if type(max_loops) is not int or not 1 <= max_loops <= STUDY_HORIZON_K_MAX:
        raise ValueError(f"max_loops must be an integer in 1..{STUDY_HORIZON_K_MAX}")


def _publish_new_directory(staging_dir, output_dir):
    """Atomic exclusive rename, or fail closed. Never fall back to os.rename.

    Linux renameat2(RENAME_NOREPLACE); macOS renamex_np(RENAME_EXCL).
    Unsupported kernels/filesystems/platforms raise without replacing evidence.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    source, destination = os.fsencode(staging_dir), os.fsencode(output_dir)
    if sys.platform == "linux" and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                           ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, source, -100, destination, 1)
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source, destination, 4)
    else:
        raise RuntimeError("Exclusive atomic directory publication is unsupported")
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), output_dir)


def run_reach_seed(
    seed: int,
    max_loops: int = STUDY_HORIZON_K_MAX,
    runner_ref_collector: Optional[List[LoopRunner]] = None,
    condition: str = "tally_current_tide_pearl_v1",
    exploration_coefficient: float = 1.0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Execute a single seed run on ReachWorld and record full per-stride telemetry.
    
    Preserves 3-result public API: (run_result, stride_records, learner_state).
    Optional runner_ref_collector allows test fixtures to inspect runner without leaking to schemas.
    """
    _validate_run_parameters([seed], max_loops)
    world = ReachWorld(max_steps=REACH_LOOP_LIMIT)
    stride_records: List[Dict[str, Any]] = []

    def on_learn(s, a, r, s_next, terminal, loop_idx, step_idx, policy_state=None, next_policy_state=None):
        tel = world.last_telemetry
        if tel is None:
            raise RuntimeError(f"Telemetry missing at seed {seed} loop {loop_idx} stride {step_idx}")

        rec = {
            "seed": seed,
            "loop": loop_idx,
            "stride": step_idx,
            "prev_obs": [list(s[0]), sorted(list(s[1]))],
            "next_obs": [list(s_next[0]), sorted(list(s_next[1]))],
            "prev_node": list(s[0]),
            "next_node": list(s_next[0]),
            "action": a,
            "reward": round(r, 4),
            "pearls_picked": tel["pearls_picked"],
            "key_picked": tel["key_picked"],
            "terminal": terminal,
            "truncated": tel["truncated"],
            "npc_states": copy.deepcopy(tel["npc_states"]),
        }
        if condition in ("tally_current_tide_pearl_v1", "cross_tide_exploration_memory_v1"):
            if policy_state is not None and next_policy_state is not None:
                rec["prev_policy_state"] = [list(policy_state[0]), sorted(list(policy_state[1])), bool(policy_state[2])]
                rec["next_policy_state"] = [list(next_policy_state[0]), sorted(list(next_policy_state[1])), bool(next_policy_state[2])]
            else:
                rec["prev_policy_state"] = [list(s[0]), sorted(list(s[1])), False]
                rec["next_policy_state"] = [list(s_next[0]), sorted(list(s_next[1])), bool(tel["pearls_picked"])]

        stride_records.append(rec)

    runner = LoopRunner(
        seed=seed,
        max_loops=max_loops,
        on_learn_callback=on_learn,
        world=world,
        condition=condition,
        exploration_coefficient=exploration_coefficient,
    )

    if runner_ref_collector is not None:
        runner_ref_collector.append(runner)

    initial_action_rng = runner.get_action_rng_state()
    initial_memory_rng = runner.memory.get_rng_state()

    res = runner.run()

    # Capture final learner stores with RNG states and condition
    learner_state = {
        "seed": seed,
        "condition": condition,
        "schema_version": "2.0.0" if condition in ("tally_current_tide_pearl_v1", "cross_tide_exploration_memory_v1") else "1.0.0",
        "initial_action_rng_state": initial_action_rng,
        "final_action_rng_state": runner.get_action_rng_state(),
        "initial_memory_rng_state": initial_memory_rng,
        "final_memory_rng_state": runner.memory.get_rng_state(),
        "agent": runner.agent.to_dict(),
        "memory": runner.memory.to_dict(),
        "model": runner.model.to_dict(),
    }

    return res, stride_records, learner_state


def get_current_git_commit() -> str:
    """Return HEAD commit hash."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent), stderr=subprocess.DEVNULL).decode().strip()
        return commit
    except Exception:
        return "unknown"


def compute_file_sha256(filepath: str) -> str:
    """Compute sha256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as fp:
        while chunk := fp.read(8192):
            h.update(chunk)
    return h.hexdigest()


def write_study_manifest(
    output_dir: str,
    protocol: Dict[str, Any],
    wall_clock_seconds: float,
    command_str: str,
    figure_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Generate and write manifest.json recording commit, environment, checksums, and metadata."""
    records_file = os.path.join(output_dir, "records.json")
    outcomes_file = os.path.join(output_dir, "outcomes.json")
    learners_file = os.path.join(output_dir, "learners.json")

    file_hashes = {}
    for name, fpath in [("records.json", records_file), ("outcomes.json", outcomes_file), ("learners.json", learners_file)]:
        if os.path.exists(fpath):
            file_hashes[name] = compute_file_sha256(fpath)

    if figure_paths:
        for f_name, f_path in figure_paths.items():
            if os.path.exists(f_path):
                file_hashes[os.path.basename(f_path)] = compute_file_sha256(f_path)

    schema_version = protocol.get("schema_version", SCHEMA_VERSION)
    manifest_payload = {
        "schema_version": schema_version,
        "condition": protocol.get("condition", "tally_current_tide_pearl_v1"),
        "source_commit": get_current_git_commit(),
        "source_provenance": "git HEAD when available; unknown means unverifiable",
        "source_hashes_sha256": {
            str(path.relative_to(Path(__file__).parent)):
                compute_file_sha256(str(path))
            for path in sorted(Path(__file__).parent.rglob("*.py"))
        },
        "protocol": protocol,
        "configuration": {
            "rewards": {"stride": STRIDE_COST, "pearls": PEARLS_REWARD,
                        "lamp_key": LAMP_KEY_REWARD, "tidegate": TIDEGATE_REWARD},
            "learning_rate": STUDY_LR,
            "gamma": STUDY_GAMMA,
            "epsilon": STUDY_EPS,
            "memory_blend": STUDY_MEMORY_BLEND,
            "memory_temperature": STUDY_MEMORY_TEMPERATURE,
            "memory_lookup_sample": STUDY_MEMORY_LOOKUP_SAMPLE,
        },
        "environment": {
            "installed_packages": {
                distribution.metadata["Name"]: distribution.version
                for distribution in importlib.metadata.distributions()
                if distribution.metadata.get("Name")
            },
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "execution": {
            "command": command_str,
            "wall_clock_seconds": round(wall_clock_seconds, 4),
        },
        "artifact_hashes_sha256": file_hashes,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest_payload, fp, indent=2)

    return manifest_payload


def run_reach_study(
    seeds: Optional[List[int]] = None,
    max_loops: int = STUDY_HORIZON_K_MAX,
    output_dir: str = "results_new",
    command_str: str = "run_reach_study()",
    condition: str = "tally_current_tide_pearl_v1",
) -> Dict[str, Any]:
    """Execute Reach study and publish atomically to a new destination directory."""
    if seeds is None:
        seeds = list(STUDY_SEEDS)

    _validate_run_parameters(seeds, max_loops)

    # Destination directory must not exist - safe publication contract
    if os.path.lexists(output_dir):
        raise FileExistsError(
            f"Destination directory already exists: '{output_dir}'. "
            "Safe publication requires a non-existent destination directory to prevent overwriting existing results."
        )

    # Use unique staging directory in same parent directory on same filesystem for atomic rename
    parent_dir = os.path.dirname(os.path.abspath(output_dir))
    staging_dir = os.path.join(parent_dir, f".staging_{os.path.basename(output_dir)}_{uuid.uuid4().hex[:8]}")
    os.makedirs(staging_dir, exist_ok=False)

    try:
        all_records: List[Dict[str, Any]] = []
        run_summaries: List[Dict[str, Any]] = []
        learners: Dict[str, Any] = {}

        start_wall_clock = time.perf_counter()

        for s in seeds:
            res, seed_records, learner_state = run_reach_seed(s, max_loops=max_loops, condition=condition)
            all_records.extend(seed_records)
            run_summaries.append(res)
            learners[str(s)] = learner_state

        total_wall_clock = time.perf_counter() - start_wall_clock

        schema_ver = "2.0.0" if condition == "tally_current_tide_pearl_v1" else "1.0.0"

        protocol = {
            "seeds": seeds,
            "horizon_k_max": max_loops,
            "loop_limit": REACH_LOOP_LIMIT,
            "condition": condition,
            "schema_version": schema_ver,
        }

        records_payload = {
            "schema_version": schema_ver,
            "condition": condition,
            "protocol": protocol,
            "total_records": len(all_records),
            "records": all_records,
        }

        learners_payload = {
            "schema_version": schema_ver,
            "condition": condition,
            "seeds": seeds,
            "learners": learners,
        }

        with open(os.path.join(staging_dir, "records.json"), "w", encoding="utf-8") as fp:
            json.dump(records_payload, fp, indent=2)

        with open(os.path.join(staging_dir, "learners.json"), "w", encoding="utf-8") as fp:
            json.dump(learners_payload, fp, indent=2)

        from remnant_core.analysis import compute_study_outcomes
        outcomes_payload = compute_study_outcomes(records_payload, run_summaries)

        with open(os.path.join(staging_dir, "outcomes.json"), "w", encoding="utf-8") as fp:
            json.dump(outcomes_payload, fp, indent=2)

        # Write manifest in staging
        write_study_manifest(
            output_dir=staging_dir,
            protocol=protocol,
            wall_clock_seconds=total_wall_clock,
            command_str=command_str,
        )

        # Validate that all expected files were written in staging
        expected_files = ["records.json", "outcomes.json", "learners.json", "manifest.json"]
        for fname in expected_files:
            fpath = os.path.join(staging_dir, fname)
            if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
                raise RuntimeError(f"Missing or empty expected artifact in staging: {fname}")

        # Double check destination does not exist right before atomic rename
        if os.path.lexists(output_dir):
            raise FileExistsError(f"Target destination appeared unexpectedly: '{output_dir}'")

        # Atomic publication: rename staging directory to target output_dir
        _publish_new_directory(staging_dir, output_dir)

    except Exception:
        # Clean up staging directory on failure
        if os.path.exists(staging_dir):
            import shutil
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return {
        "seeds": seeds,
        "total_seeds": len(seeds),
        "total_records": len(all_records),
        "wall_clock_seconds": total_wall_clock,
        "escaped_count": sum(1 for r in run_summaries if r["escaped"]),
        "censored_count": sum(1 for r in run_summaries if not r["escaped"]),
    }


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for Reach study."""
    parser = argparse.ArgumentParser(
        description="Run Reach of Vesper empirical study with safe destination publication."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        required=True,
        help="Path to new, non-existent directory for publishing study artifacts.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(STUDY_SEEDS),
                        help="Selected integer seeds, for example --seeds 101")
    parser.add_argument("--max-loops", type=int, default=STUDY_HORIZON_K_MAX,
                        help="Bounded loop cap, 1..100")
    parser.add_argument(
        "--condition",
        type=str,
        default="tally_current_tide_pearl_v1",
        choices=["tally_current_tide_pearl_v1", "baseline_hidden_pearl_v1"],
        help="Study condition (tally_current_tide_pearl_v1 or baseline_hidden_pearl_v1)",
    )
    parsed = parser.parse_args(args)
    try:
        _validate_run_parameters(parsed.seeds, parsed.max_loops)
    except ValueError as error:
        parser.error(str(error))
    return parsed


if __name__ == "__main__":
    parsed = parse_args(sys.argv[1:])
    cmd = "python -m remnant_core.study " + " ".join(sys.argv[1:])
    print(f"Starting Reach study ({len(parsed.seeds)} seeds, cap {parsed.max_loops}, condition {parsed.condition}) -> {parsed.output_dir}")
    res = run_reach_study(seeds=parsed.seeds, max_loops=parsed.max_loops, output_dir=parsed.output_dir, command_str=cmd, condition=parsed.condition)
    print("Study completed successfully:")
    print(f"  Total seeds: {res['total_seeds']}")
    print(f"  Escaped seeds: {res['escaped_count']}")
    print(f"  Censored seeds: {res['censored_count']}")
    print(f"  Total records: {res['total_records']}")
    print(f"  Wall-clock time: {res['wall_clock_seconds']:.2f}s")
