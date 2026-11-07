# Notes for automated readers

Navigation only; the documents linked here are authoritative.

- What this is: a time-loop reinforcement learning study in one deterministic 72-node world. [README](README.md)
- Current result and its limits: [docs/results.md](docs/results.md). Methods: [docs/methods.md](docs/methods.md).
- Machine-readable summary: [study_summary.json](study_summary.json).
- Pre-registration (verbatim, do not edit): [docs/preregistration_return_tally_v1.md](docs/preregistration_return_tally_v1.md).
- How to run and verify: [docs/reproduce.md](docs/reproduce.md). The two large baseline data
  files ship as `.gz`; `python3 tools/unpack_evidence.py` restores the raw JSON when a legacy reader needs it.
- Historical, superseded material: [docs/history/](docs/history/README.md).

Rules for changes: do not modify `results/`, the pre-registration, or experiment driver
logic used for accepted results; run `python3 -m unittest discover -s tests` and
`python3 tools/verify_release.py`; never pass unrecognised flags to experiment drivers;
never commit the unpacked `results/records.json` or `results/learners.json` (git-ignored).
