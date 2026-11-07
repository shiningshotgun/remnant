# Changes

## Figure correction and Git-safe evidence packaging (2026-11-07)

Packaging correction only. No experiment, policy or full sweep was run; no study code changed.

- **Figure 2 annotation corrected.** It said the primary and legacy-learner arms "discover
  at the same tides". The first-discovery tide matches in only 15 of the 30 paired seeds, so
  it now reads "have similar, not identical discovery times". Only
  `figures/return_tally_v1/fig2_discovery_and_learning.png` was regenerated, by calling
  `fig2()` on the unchanged result file in the pinned plotting environment
  (`requirements.lock.txt`, CPython 3.12.15). Figures 1 and 3 are byte-identical; plotted data
  and other labels are unchanged.
- **Large baseline files stored compressed.** `results/records.json` (277,819,583 bytes) and
  `results/learners.json` (65,805,407 bytes) ship as `results/records.json.gz` (3,224,849 bytes)
  and `results/learners.json.gz` (1,150,294 bytes): standard-library gzip, level 9, empty
  header file name, mtime 0. Lossless; the raw SHA-256 values (`7f8f0c7c…a40498`,
  `3bd4e1f0…316a39`) are authoritative and match `results/manifest.json`. Sizes and hashes
  of both forms: `provenance/evidence_data_manifest.json`.
- **New `tools/unpack_evidence.py`** (standard library): `--verify` checks without writing;
  no flag restores both raw files, never overwriting, with fixed path pairs, bounded
  streaming decompression and per-file no-clobber placement.
- **`tools/verify_release.py`** passes on a clean compressed extraction, checks the two raw
  files whenever present, and with `--raw` requires them. Symlinks are reported. The two
  raw paths are git-ignored.
- **Tests:** `tests/test_evidence.py` (fixture trees for every failure case).
- **CI** (`.github/workflows/checks.yml`) now also verifies and unpacks the compressed
  evidence. It has not run remotely.
- Historical manifests and result provenance were not rewritten.

## Release preparation (2026-11-06)

Presentation, reproducibility and attribution pass for the accepted study
`return_tally_v1`. No experiment was re-run to produce any reported number.

### Evidence kept byte-identical

Accepted result, pre-registration and the code that produced the result are unchanged:

| File | SHA-256 |
|---|---|
| `results/return_tally_v1.json` | `b3c6a3ac2bbd830ffd269e203e73e8e2a33ea4853e871a783d6850666345d9a9` |
| `docs/preregistration_return_tally_v1.md` | `52291de791f5a3f0d06737dd0b2a93febf9ac4d014883b476aaabf50e7e0da67` |
| `experiments/run_return_tally_v1.py` | `cb5c4cb6331878b362c3cdd8d1f072db563617c037169703957ee54d20825012` |
| `remnant_core/return_tally.py` | `c1b030190ec27297224970a57dc52cf18be1d91f19251f71a2e6242e872feb8b` |
| `remnant_core/episodic_runner.py` | `524c54996780f355d7c279fa419afb284cc89c627867bf8284f52d2f2aecc1d0` |
| `results/metabucket_v1.json` | `a7fb4b76d33a4f23ab149df2676c8a50eb891698edf6d697c31347d9cee506c2` |
| `results/metabucket_td_v1.json` | `52b1b82f2fad742effd024ddf58f4b5846a80a54dce19de23aa340e30c1956ba` |

All other files under `results/` and `pilot_results/` are unchanged from the handoff bundle.

### Fixed: legacy drivers ran the full experiment on any unknown flag

- **Defect.** `experiments/run_metabucket_v1.py` accepted only `--smoke`, and
  `experiments/run_metabucket_td_v1.py` only `--smoke` and `--gate`. Any other argument
  (for example `--gate` on the first driver, or a typo) fell through to `main()`, which
  retrained the whole experiment and overwrote `results/metabucket_*.json`.
- **Impact.** Earlier, `run_metabucket_v1.py --gate` was run in the belief that it was a
  gate. It re-ran the experiment and rewrote `results/metabucket_v1.json`; the rewritten
  file was byte-identical to the committed one (SHA-256 above), so no evidence changed. No
  scientific result depends on the argument parsing.
- **Fix.** Exact-match argument parsing (`_parse_mode`). Valid invocations behave as before
  (no flag runs the experiment; `--smoke`; `--gate` on the TD driver). Anything else prints
  usage to stderr and exits 2 before any training or writing. Experiment logic is untouched.
  `run_metabucket_v1.py` still has no gate; none was invented.
- **Regression tests.** `tests/test_driver_cli.py`: the training function is replaced with
  one that fails if called, and copies of the drivers run against fixture result files in a
  temporary directory to show that unknown flags exit 2 with the fixture bytes unchanged.
- **Docs.** The claim that every driver has a gate is corrected (README, reproduction guide).
- **Not changed.** `run_return_tally_v1.py` is accepted-result code and is left
  byte-identical. It checks flags by membership: an unknown flag alone exits 2, `--run`
  refuses to overwrite an existing result file, and `--run` combined with other flags is
  not rejected. This is documented, not altered.

### Manifest and provenance

- The old root `sha256_manifest.txt` is moved unchanged to
  `provenance/legacy_sha256_manifest.txt` and labelled stale: six legacy source files have
  changed since it was written, and it lists `LICENSE`, which is not in this bundle.
- New `RELEASE_MANIFEST.sha256` covers every file in the release tree except itself,
  the private chronology (`work_done.md`) and runtime files. `tools/verify_release.py`
  reports mismatched, missing and unexpected files.

### Scientific account

- New `docs/methods.md` and `docs/results.md` for the accepted study. Earlier methods,
  results, the draft protocol and the earlier summary JSON moved to `docs/history/`, each
  with a banner. References to Route B result and figure directories that this release does
  not contain are marked as not included.
- The first write-up of return_tally_v1 moved to `docs/history/` with corrections listed
  at the top: arms share the exploration mechanism and coin, not training experience;
  determinism does not by itself guarantee every remembered maximum is attainable (time is
  not in the state); the legacy-failure explanations are diagnoses.
- Added a post-hoc paired sign test (`analysis/paired_return_tally_v1.py` and `.json`),
  computed from the result rows and kept separate from the pre-registered Fisher tests.
- `study_summary.json` (schema 3.0.0) now describes the accepted study.
- Added `docs/related_work.md` (contribution, model-free episodic control, Go-Explore,
  episodic curiosity), `docs/exploratory_negatives.md`, `dev/README.md`.

### Reproduction and presentation

- New README: premise, result, what was learned versus supplied, limits, data-flow
  diagram, quick start, evidence links, attribution.
- Figures derived from the result file: `figures/return_tally_v1/` via
  `analysis/figures_return_tally_v1.py`. Checked at full size and at phone width (about
  420 px).
- `tools/demo_frozen_policy.py` (trains one seed, then deploys the frozen policy with
  curiosity disabled and the stores checked), `tools/reproduce_return_tally_v1.py`
  (re-runs in a separate directory and compares).
- `requirements.txt` renamed to `requirements-plot.txt` (matplotlib 3.9.2, optional).
  `requirements.lock.txt` is unchanged.
- `.github/workflows/checks.yml`: routine checks on CPython 3.12. It has not run remotely.
- `AGENTS.md`: short navigation for automated readers. `docs/repository_metadata.md`:
  suggested description and topics.
