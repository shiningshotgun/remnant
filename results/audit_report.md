# Remnant: Reach of Vesper Integrity Audit Report

This report summarizes the integrity audit of the completed Reach of Vesper experimental study.

## 1. Study Overview and Empirical Results

The Reach of Vesper study is an empirical investigation of tabular Q-learning with episodic memory on a 72-node waypoint graph:
- **Protocol**: 30 seeds (101–130), horizon $K_{\max} = 100$ tides per seed, $L = 54$ strides per tide.
- **Escape Rate**: 0.0% (0 / 30 seeds escaped; all 30 censored at 100 tides).
- **Physical Strides**: Exactly 162,000 strides executed across the 3,000 tides (54 strides per tide without exception).
- **Pearls Attractor**: Pearl collection at the Drowned Chapel rose from 0% (loops 1 and 10) to 26.7% (loop 25), 53.3% (loop 50), 90.0% (loop 75), and 93.3% (loop 100).
- **Fluctuation / Non-monotonicity**: Across the 100 tides, the pearl collection rate dropped 34 times between adjacent loops, demonstrating an overall increasing trend with exploratory fluctuations rather than monotonic growth.
- **Censoring Interpretation**: Censoring at loop 100 reflects the prespecified horizon bound; it does not demonstrate that the agent is fundamentally incapable of escaping given a larger horizon.

## 2. Integrity Verification and Replay Evidence

All four primary study result files in `results/` are byte-frozen and verified against their historical SHA-256 preimages:
- `results/records.json`: `7f8f0c7c9c3e5dd7e0a1ebf9f697f95571c50293634898c3e68ee56825a40498`
- `results/learners.json`: `3bd4e1f067e0d0ad0e908ce497c291dab631c47f5554ec872f11b8825e316a39`
- `results/outcomes.json`: `07d35e7fdd2f61785a3903cee09751bd5a8fdd9f51d52b4829ff259c5990b43b`
- `results/manifest.json`: `baf4ff7db17fad450fb1dadc260330fd8f8accd0d6a6eba8ac6b9eb2b5b2e276`

### Independent Verification Scope
- **Physical Transitions**: Replay of all 162,000 recorded actions through `physical_transition` confirmed 100% agreement with recorded state transitions, rewards, and item flags.
- **World Model Reconstructions**: Rebuilding the tabular transition and reward model from the 162,000 actions for all 30 seeds matched the saved models in `learners.json` bit-for-bit.
- **Analytical Outcomes Regeneration**: Recomputing analytical outcomes from `records.json` matched `outcomes.json` exactly (recomputed outcomes compared directly with the JSON loaded from frozen `results/outcomes.json`).

### Historical Replay Attribution
- **Prior Independent Review**: Ran training replays for seeds 101 and 130 and verified all 162,000 saved transitions.
- **User-reported Historical Replay**: The user reported fresh training replay of all 30 seeds; no retained all-seed training logs are packaged. This is not an independently repeated all-seed training result.
- **Audit Verification**: Performed read-only physical transition validation and world-model reconstruction across all 30 seeds without retraining. Full historical training execution logs for all 30 seeds were not retained.

## 3. Scope of Corrections

1. **Machinery & Validation**:
   - Implemented strict records integrity validation in `analysis.py`, rejecting corrupt schemas, non-finite values, invalid seeds, and structural breaks.
   - Closed testing gaps in `test_characters.py`, `test_reach_world.py`, `test_reach_solver.py`, and `test_study.py`.
   - Added exclusive destination publication and CLI argument validation in `study.py` and `test_output_safety.py`.
2. **Evidence Presentation**:
   - Corrected pearl collection milestones and non-monotonic trend documentation in `docs/results.md`.
   - Clarified partial observability, action encoding, subsampling, and truncation bootstrapping in `docs/methods.md`.
   - Updated diagnostic figures (`escape-timing.png` categorical censor count at horizon, `character-activity.png` categorical legend and initial state labels).


## 4. Check records, runtime and limits

The user reported macOS/Python 3.9.6 audit execution, 39 targeted tests passing, 10 structural tests passing, a complete virtual-environment suite of 73 passing tests with no skips, and system Python with 72 passing tests and one dependency skip. The earlier exact corrected-state guard matched 49 files. These local execution and git identity/status claims are user-reported; the source ZIP does not contain git history or those complete execution logs.

Independent review of the delivered ZIP used Linux/Python 3.10.12, matplotlib 3.10.9 and Pillow 12.3.0: 73 full-suite tests passed without skips, all 51 packaged checksums passed, all frozen hashes matched, regenerated outcomes matched exactly, all 162,000 live-world saved-action transitions matched including NPC telemetry, and all 30 reconstructed models matched. No fresh training was performed by that review.

### Red-event chronology reported by the user

1. An ad-hoc verification snippet imported nonexistent `solve_reachability_without_key`; execution stopped. No product fix was needed. Existing grid/Reach solver test modules cover the actual no-key API and passed 10 tests.
2. A `git diff --name-only` invocation ran outside the repository and exited 129; execution stopped. Subsequent commands were required to establish the repository directory in each fresh shell. The user later reported the 49-file postimage guard passed; the ZIP review independently checked the expected source bytes.

### Bounded presentation follow-up

The final ZIP review found that the censor bar was still placed at an invented numeric loop coordinate with overlapping tick text, public documents still overstated replay/mechanism evidence, and audit command/configuration provenance was incomplete. The follow-up changes only the escape renderer/figure, those flagged document statements, this report and the separate audit manifest/checksum list. It preserves the four frozen results and validated machinery. The manifest distinguishes user-reported historical commands/runtime from the follow-up's verified runtime/commands. Audit output does not certify local git history, commit identity or a clean tree by itself.

### This follow-up execution (generated by the correction helper)

Runtime and packages: `{"machine": "arm64", "platform": "macOS-27.0-arm64-arm-64bit", "python_version": "3.9.6 (default, Aug 25 2026, 21:26:21) \n[Clang 21.0.0 (clang-2100.3.34.2)]", "verified_installed_packages": {"matplotlib": "3.9.2", "numpy": "2.0.2", "pillow": "11.3.0"}}`.

Existing proof modules: 10 tests passed. Full suite: 73 tests passed. Read-only saved-action verifier: 162,000 transitions, 30 models, exact outcomes, no training. Command logs and fresh production/miniature escape figures are retained in the reported isolated scratch directory. User must inspect the actual two escape images before a commit; this text does not assert that human visual gate has happened.
