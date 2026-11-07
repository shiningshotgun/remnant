"""Verify this source tree against RELEASE_MANIFEST.sha256.

Scope: every file in the tree, except the exclusions below. The manifest does not list
its own digest (that would be self-referential). The check reports
  - MISMATCH  a listed file whose SHA-256 differs,
  - MISSING   a listed file that is absent,
  - UNEXPECTED a file present in the tree but not listed (and not excluded).
Exit status 0 only when there are none of these.

Excluded by design (private or runtime material, never part of the release payload):
  RELEASE_MANIFEST.sha256   the manifest itself
  work_done.md              private maintainers' session chronology (git-ignored)
  __pycache__/, *.pyc, .git/, .venv/, .DS_Store   interpreter, VCS and OS runtime files

Unpacked evidence. The source payload ships results/records.json.gz and
results/learners.json.gz (listed and hashed here like every other file). The two raw files
they expand to, results/records.json and results/learners.json, are the only paths that may
appear without being listed. They are never ignored: whenever present they must be regular
files whose size and SHA-256 match provenance/evidence_data_manifest.json (itself listed
here). Any symlink in the tree is reported as LINK.
  default mode   raw files optional; checked if present            (clean source extraction)
  --raw          raw files required and checked                    (after tools/unpack_evidence.py)

Usage:
  python3 tools/verify_release.py            verify the source payload (read-only)
  python3 tools/verify_release.py --raw      also require both unpacked raw files (read-only)
  python3 tools/verify_release.py --write    maintainers only: rewrite the manifest from the tree
"""
import hashlib
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = "RELEASE_MANIFEST.sha256"
EXCLUDED_FILES = {MANIFEST, "work_done.md"}
EXCLUDED_DIRS = {"__pycache__", ".git", ".venv"}
EXCLUDED_SUFFIXES = (".pyc",)
EXCLUDED_NAMES = {".DS_Store"}


def _evidence_module():
    spec = importlib.util.spec_from_file_location(
        "remnant_unpack_evidence", os.path.join(os.path.dirname(os.path.abspath(__file__)), "unpack_evidence.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EVIDENCE = _evidence_module()
RAW_PATHS = set(EVIDENCE.RAW_PATHS)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _walk(root):
    """Yield (rel, is_link) for every non-excluded entry; symlinked dirs are not followed."""
    for dirpath, dirnames, filenames in os.walk(root):
        keep = []
        for d in sorted(dirnames):
            if d in EXCLUDED_DIRS:
                continue
            full = os.path.join(dirpath, d)
            if os.path.islink(full):
                yield os.path.relpath(full, root).replace(os.sep, "/"), True
            else:
                keep.append(d)
        dirnames[:] = keep
        for name in sorted(filenames):
            rel = os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, "/")
            if rel in EXCLUDED_FILES or name in EXCLUDED_NAMES or name.endswith(EXCLUDED_SUFFIXES):
                continue
            yield rel, os.path.islink(os.path.join(dirpath, name))


def tree_files(root=ROOT):
    """Regular files that belong in the manifest (excludes links and the two raw paths)."""
    return sorted(rel for rel, link in _walk(root) if not link and rel not in RAW_PATHS)


def tree_links(root=ROOT):
    return sorted(rel for rel, link in _walk(root) if link)


def read_manifest(root=ROOT):
    entries = {}
    with open(os.path.join(root, MANIFEST)) as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            digest, sep, rel = line.partition("  ")
            if not sep or len(digest) != 64 or rel in entries:
                raise SystemExit(f"malformed manifest line {n}: {line!r}")
            entries[rel] = digest
    return entries


def verify(root=ROOT, require_raw=False):
    listed = read_manifest(root)
    links = tree_links(root)
    present = set(tree_files(root))
    problems = [("LINK", rel) for rel in links]
    for rel in sorted(set(listed) & RAW_PATHS):
        problems.append(("UNEXPECTED", rel))  # raw data must never be listed as source
    for rel, digest in sorted(listed.items()):
        if rel in RAW_PATHS:
            continue
        if rel not in present:
            problems.append(("MISSING", rel))
        elif sha256(os.path.join(root, rel)) != digest:
            problems.append(("MISMATCH", rel))
    for rel in sorted(present - set(listed)):
        problems.append(("UNEXPECTED", rel))
    problems.extend(check_raw(root, require_raw, skip=set(links)))
    return listed, problems


def check_raw(root=ROOT, require_raw=False, skip=()):
    """Check the two unpacked raw files against the (listed) data manifest."""
    any_raw = any(os.path.lexists(os.path.join(root, r)) for r in RAW_PATHS)
    if not os.path.lexists(os.path.join(root, EVIDENCE.DATA_MANIFEST)) and not any_raw and not require_raw:
        return []  # nothing to check; if the data manifest is listed, its absence is MISSING
    try:
        entries = EVIDENCE.load_manifest(root)
    except EVIDENCE.EvidenceError as e:
        return [("RAW-MANIFEST", e.path)]
    out = []
    for _gz, _gb, _gs, raw_rel, raw_bytes, raw_sha in entries:
        if raw_rel in skip:
            if require_raw:
                out.append(("RAW-MISSING", raw_rel))  # a link is never accepted as raw data
            continue
        try:
            state = EVIDENCE.raw_state(root, raw_rel, raw_bytes, raw_sha)
        except EVIDENCE.EvidenceError:
            out.append(("RAW-MISMATCH", raw_rel))
            continue
        if state == "absent" and require_raw:
            out.append(("RAW-MISSING", raw_rel))
    return out


def write(root=ROOT):
    links = tree_links(root)
    if links:
        raise SystemExit(f"refusing to write a manifest for a tree with symlinks: {links}")
    files = tree_files(root)
    with open(os.path.join(root, MANIFEST), "w") as f:
        f.write("# Remnant release manifest: SHA-256 of every file in the source payload.\n")
        f.write("# Excludes itself, work_done.md (private), runtime files and the two unpacked raw\n")
        f.write("# evidence files (checked separately); see tools/verify_release.py.\n")
        for rel in files:
            f.write(f"{sha256(os.path.join(root, rel))}  {rel}\n")
    print(f"wrote {MANIFEST}: {len(files)} files")


def main():
    args = sys.argv[1:]
    if args == ["--write"]:
        write()
        return
    if args not in ([], ["--raw"]):
        print("usage: verify_release.py [--raw | --write]", file=sys.stderr)
        sys.exit(2)
    require_raw = args == ["--raw"]
    listed, problems = verify(require_raw=require_raw)
    for kind, rel in problems:
        print(f"{kind:12s} {rel}")
    if problems:
        print(f"FAILED: {len(problems)} problem(s) across {len(listed)} listed files")
        sys.exit(1)
    raw_present = [r for r in sorted(RAW_PATHS) if os.path.lexists(os.path.join(ROOT, r))]
    if len(raw_present) == len(RAW_PATHS):
        raw_note = "both unpacked raw files present and verified"
    elif raw_present:
        raw_note = f"unpacked raw file present and verified: {raw_present[0]}"
    else:
        raw_note = "raw evidence not unpacked (optional; see tools/unpack_evidence.py)"
    print(f"RELEASE OK{' (--raw)' if require_raw else ''}: {len(listed)} files match {MANIFEST}; "
          f"no missing or unexpected files; {raw_note}")


if __name__ == "__main__":
    main()
