# Provenance

`legacy_sha256_manifest.txt` is the historical root manifest from an earlier commit, moved
here unchanged (formerly `sha256_manifest.txt`). It is **stale**: six legacy source files
(`remnant_core/agent.py`, `memory.py`, `repair_diagnostic.py`, `runner.py`, `study.py`,
`study_config.py`) have changed since it was written, and it lists `LICENSE`, which is not
in this source bundle. Keep it as provenance only; do not use it to verify this release.

The current release manifest is `RELEASE_MANIFEST.sha256` at the root, checked by
`python3 tools/verify_release.py`.

`evidence_data_manifest.json` records, for `results/records.json.gz` and
`results/learners.json.gz`, the compressed byte size and SHA-256 (source-payload hashes) and
the raw byte size and SHA-256 after decompression (extracted-data hashes). The raw values are
authoritative and equal those in `results/manifest.json` and `results/audit_manifest.json`.
See [docs/reproduce.md](../docs/reproduce.md#compressed-historical-evidence).
