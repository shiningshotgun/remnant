# Historical material

Earlier studies and documents, kept for the record. None of this is the current result;
see [results.md](../results.md).

| File | What it is |
|---|---|
| [baseline_methods.md](baseline_methods.md) | Methods of the original 30-seed baseline (`baseline_hidden_pearl_v1`, seeds 101–130) and Route B diagnostic. |
| [baseline_and_route_b_results.md](baseline_and_route_b_results.md) | Their results: 0/30 baseline escapes; Route B 0/3. Route B artifact directories are not in this release. |
| [controlled_study_protocol_draft.md](controlled_study_protocol_draft.md) | A draft memory-ablation protocol, marked invalid as written and never run. |
| [study_summary_baseline.json](study_summary_baseline.json) | The earlier machine-readable summary (baseline and Route B). |
| [return_tally_v1_first_report.md](return_tally_v1_first_report.md) | The first, superseded write-up of return_tally_v1, with corrections noted at the top. |

Baseline evidence in the repository: `results/records.json` and `results/learners.json`
(shipped gzip-compressed as `.json.gz`; restore with `python3 tools/unpack_evidence.py`),
`results/outcomes.json`, `results/manifest.json`, `results/audit_report.md`,
`results/audit_manifest.json`, `pilot_results/`, and figures in `figures/` and
`figures_derived/` (all baseline-only). The learned mode-selector runs are in
`results/metabucket_v1.json` and `results/metabucket_td_v1.json` (3 seeds each; learned
selectors 0/3 escapes, random-selection control 3/3 with planner-executed escapes).
The baseline is a descriptive study of a different agent and is not a matched arm of
return_tally_v1.
