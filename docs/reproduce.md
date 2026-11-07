# Reproduction guide

Tested interpreter: CPython 3.12.15 (Linux). The test suite and checks were also run on
CPython 3.13 (see [CHANGES.md](../CHANGES.md)). The study code uses only the standard
library. Run every command from the repository root of an extracted copy. None of the
commands below writes to `results/`, except `tools/unpack_evidence.py` without
`--verify`, which adds the two raw files described under
[Compressed historical evidence](#compressed-historical-evidence).

## 1. Verify the existing results (seconds)

```sh
python3 tools/verify_release.py                       # release manifest: hashes, missing and unexpected files
python3 tools/unpack_evidence.py --verify             # compressed historical data and its raw hashes (no writes)
python3 -m unittest discover -s tests                 # test suite (two optional figure tests skip without matplotlib)
python3 experiments/run_return_tally_v1.py --smoke    # construction check only, no outcome
python3 experiments/run_return_tally_v1.py --gate     # config, structure, summary recompute,
                                                      # fresh byte-equal re-run of first and last seed of every arm
python3 experiments/run_metabucket_td_v1.py --gate    # inherited selector study gate
python3 analysis/paired_return_tally_v1.py            # recompute the paired analysis (prints; add --write to save)
```

`experiments/run_metabucket_v1.py` has `--smoke` only; it has no gate. Running it with no
flag re-runs that historical experiment and overwrites its result file, so don't do that
in a tree whose evidence you want to keep. Unknown flags are rejected with exit code 2.

## Compressed historical evidence

The two largest historical baseline files are shipped gzip-compressed so that every file in
the source payload stays under GitHub's 100 MiB file limit. This changes storage only: the
raw bytes are unchanged and are checked by size and SHA-256 after decompression.

| Shipped (source payload) | Bytes | SHA-256 of shipped `.gz` |
|---|---:|---|
| `results/records.json.gz` | 3,224,849 | `51653c1191f242553ee571d83e20a8e803b5ea234806975dc6b898dc8f76a339` |
| `results/learners.json.gz` | 1,150,294 | `0cda144b944f82ea9c13ddf0679ebb2e2fb49d6fee5648310868b3d73be97770` |

| Extracted (raw data) | Bytes | SHA-256 of raw JSON (authoritative) |
|---|---:|---|
| `results/records.json` | 277,819,583 | `7f8f0c7c9c3e5dd7e0a1ebf9f697f95571c50293634898c3e68ee56825a40498` |
| `results/learners.json` | 65,805,407 | `3bd4e1f067e0d0ad0e908ce497c291dab631c47f5554ec872f11b8825e316a39` |

The raw hashes are the ones in `results/manifest.json` and `results/audit_manifest.json`.
The `.gz` bytes were written by CPython 3.12's `gzip` module (level 9, empty header file
name, mtime 0); another runtime may compress to different bytes, which is why the raw
hashes are the reference. All values are in `provenance/evidence_data_manifest.json`.

```sh
python3 tools/unpack_evidence.py --verify   # check .gz files and decompressed size/hash; writes nothing
python3 tools/unpack_evidence.py            # also write results/records.json and results/learners.json
python3 tools/verify_release.py --raw       # source payload plus both raw files required and checked
```

**When you need it.** Everything for the current study (`return_tally_v1`: tests, smoke,
gates, paired analysis, demo, figures, reproduction) runs directly from the compressed
source payload. Only legacy readers of the historical baseline need the raw files: for
example `remnant_core/figures.py` (defaults to `results/records.json`) or re-running the
baseline audit described in `results/audit_report.md`. Unpacking needs about 344 MB of disk
and takes a few seconds.

**Safety.** Unpacking never overwrites. An existing raw file with the right size and hash
counts as done (running it twice is safe); a different file, a symlink, a directory or a
symlinked `results/` directory makes it stop with exit status 1, naming the failed phase
and any file already placed. Output goes to a temporary file in `results/`, is fully checked,
then linked into place only if the destination is still free. The two raw paths are
git-ignored so they cannot be committed by accident.

**Verification in both states.** `python3 tools/verify_release.py` passes on a clean
extraction (raw files absent) and on an unpacked tree; if a raw file is present it is
checked, never ignored. `--raw` additionally requires both. Any other file not in the
manifest, including any other JSON in `results/`, is reported as unexpected.

## 2. Small demo of the learned frozen policy (about 1 second)

```sh
python3 tools/demo_frozen_policy.py            # or --seed N for N in 7001..7030
```

No trained state is shipped. The demo first trains the chosen seed for 100 tides with the
primary arm's configuration, then deploys the frozen policy: curiosity is disabled (calling
it raises), the Return Tally and map are checked unchanged, and the route is compared with
the accepted result for that seed.

## 3. Figures (optional dependency)

```sh
python3 -m pip install -r requirements-plot.txt      # matplotlib 3.9.2
python3 analysis/figures_return_tally_v1.py          # writes figures/return_tally_v1/*.png
python3 analysis/figures_return_tally_v1.py --out /tmp/remnant-figs   # or elsewhere
```

`requirements.lock.txt` pins a full plotting environment (matplotlib 3.9.2, numpy 2.0.2, …).
Installed into a fresh CPython 3.12 virtual environment, it reproduced the shipped PNG
files byte-for-byte.

## 4. Full reproduction in a separate directory

The driver's `--run` is one-shot and refuses to overwrite an existing results file, so a
full re-run belongs in a disposable copy. The helper below makes that copy for you (it
copies only `remnant_core/` and `experiments/`) and never touches this tree:

```sh
# fast: re-run chosen seeds of every arm and compare rows with the accepted ones
python3 tools/reproduce_return_tally_v1.py --dest /tmp/remnant-repro-quick --seeds 7001,7030

# full: 5 arms x 30 seeds x 100 tides, about 2 minutes on CPython 3.12;
# compares the SHA-256 of the new results file with the accepted one
python3 tools/reproduce_return_tally_v1.py --dest /tmp/remnant-repro-full --full
```

The destination must not exist and must be outside the source tree.

## Costs at a glance

| Task | Time (CPython 3.12) | Writes |
|---|---|---|
| Verify existing results (section 1) | under a minute | nothing |
| Unpack historical evidence | a few seconds, about 344 MB | `results/records.json`, `results/learners.json` |
| Demo | about 1 s | nothing |
| Figures | a few seconds | `figures/return_tally_v1/` or `--out` |
| Quick seed reproduction | seconds | the `--dest` directory |
| Full sweep | about 2 min | the `--dest` directory |
