"""Verify and (optionally) unpack the two compressed historical evidence files.

The baseline study's two large raw files are shipped gzip-compressed (standard library,
empty header file name, mtime 0) so that every file in the source payload is below
GitHub's 100 MiB blob limit. Compression is a storage change only: the raw bytes are
authoritative and are checked by size and SHA-256 after decompression.

    results/records.json.gz   ->  results/records.json
    results/learners.json.gz  ->  results/learners.json

Sizes and hashes live in provenance/evidence_data_manifest.json. The two path pairs are
fixed in this file; the manifest must name exactly these pairs and cannot choose any
other destination.

Usage (standard library only, no downloads, no training):
  python3 tools/unpack_evidence.py --verify   check compressed files and the decompressed
                                              stream (size, SHA-256) without writing
  python3 tools/unpack_evidence.py            verify, then write both raw files

Unpacking never overwrites. A raw file that is already present with the right size and
hash counts as done; anything else at a destination (a different file, a link, a
directory) is a failure. Each raw file is written to a temporary file inside results/,
fully checked, then linked into place only if the destination is still free. This is
per-file, not a multi-file transaction: a failure report says which files, if any, were
already in place. Exit status: 0 success, 1 failure, 2 usage error.
"""
import hashlib
import json
import os
import stat
import sys
import tempfile
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_MANIFEST = "provenance/evidence_data_manifest.json"
FIXED_PAIRS = (
    ("results/records.json.gz", "results/records.json"),
    ("results/learners.json.gz", "results/learners.json"),
)
RAW_PATHS = tuple(raw for _gz, raw in FIXED_PAIRS)
CHUNK = 1 << 20


class EvidenceError(Exception):
    def __init__(self, phase, path, reason):
        super().__init__(f"phase={phase} file={path}: {reason}")
        self.phase, self.path, self.reason = phase, path, reason


def _is_hex64(s):
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def load_manifest(root=ROOT):
    """Return [(gz_rel, gz_bytes, gz_sha, raw_rel, raw_bytes, raw_sha)] in FIXED_PAIRS order."""
    path = _checked_path(root, DATA_MANIFEST, phase="manifest")
    st = os.lstat(path) if os.path.lexists(path) else None
    if st is None or not stat.S_ISREG(st.st_mode):
        raise EvidenceError("manifest", DATA_MANIFEST, "missing or not a regular file")
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        entries = doc["entries"]
    except (ValueError, KeyError, TypeError) as e:
        raise EvidenceError("manifest", DATA_MANIFEST, f"malformed ({e})")
    if not isinstance(entries, list) or len(entries) != len(FIXED_PAIRS):
        raise EvidenceError("manifest", DATA_MANIFEST, "must list exactly the two fixed pairs")
    out = []
    for (gz_rel, raw_rel), e in zip(FIXED_PAIRS, entries):
        try:
            c, r = e["compressed"], e["raw"]
            got = (c["path"], r["path"])
            vals = (c["bytes"], c["sha256"], r["bytes"], r["sha256"])
        except (KeyError, TypeError) as err:
            raise EvidenceError("manifest", DATA_MANIFEST, f"malformed entry ({err})")
        if got != (gz_rel, raw_rel):
            raise EvidenceError("manifest", DATA_MANIFEST, f"unexpected path pair {got!r}")
        if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (vals[0], vals[2])):
            raise EvidenceError("manifest", DATA_MANIFEST, f"bad byte size for {raw_rel}")
        if not (_is_hex64(vals[1]) and _is_hex64(vals[3])):
            raise EvidenceError("manifest", DATA_MANIFEST, f"bad sha256 for {raw_rel}")
        out.append((gz_rel, vals[0], vals[1], raw_rel, vals[2], vals[3]))
    return out


def _checked_path(root, rel, phase):
    """Join a fixed relative path to root, refusing traversal and symlinked parent dirs."""
    allowed = {DATA_MANIFEST} | {p for pair in FIXED_PAIRS for p in pair}
    if rel not in allowed or os.path.isabs(rel) or ".." in rel.split("/"):
        raise EvidenceError(phase, rel, "path not in the fixed allow-list")
    parts = rel.split("/")
    cur = root
    for part in parts[:-1]:
        cur = os.path.join(cur, part)
        try:
            st = os.lstat(cur)
        except FileNotFoundError:
            raise EvidenceError(phase, rel, f"parent directory {part}/ is missing")
        if stat.S_ISLNK(st.st_mode):
            raise EvidenceError(phase, rel, f"parent directory {part}/ is a symlink; refusing")
        if not stat.S_ISDIR(st.st_mode):
            raise EvidenceError(phase, rel, f"parent {part} is not a directory")
    return os.path.join(root, *parts)


def _sha_file(path):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
            n += len(block)
    return n, h.hexdigest()


def check_compressed(root, gz_rel, gz_bytes, gz_sha):
    path = _checked_path(root, gz_rel, "verify-compressed")
    if not os.path.lexists(path):
        raise EvidenceError("verify-compressed", gz_rel, "missing")
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode):
        raise EvidenceError("verify-compressed", gz_rel, "not a regular file (link or directory)")
    n, h = _sha_file(path)
    if n != gz_bytes:
        raise EvidenceError("verify-compressed", gz_rel, f"size {n} != declared {gz_bytes}")
    if h != gz_sha:
        raise EvidenceError("verify-compressed", gz_rel, "SHA-256 differs from the data manifest")
    return path


def stream_raw(gz_path, gz_rel, raw_bytes, raw_sha, sink=None):
    """Decompress one gzip member, bounded by raw_bytes; check size and SHA-256.

    Rejects malformed headers, corrupt data, CRC/length trailer mismatch, truncation,
    trailing bytes after the member and output beyond the declared size.
    """
    d = zlib.decompressobj(wbits=31)  # gzip wrapper, single member; checks CRC32 and ISIZE
    h = hashlib.sha256()
    n = 0
    try:
        with open(gz_path, "rb") as f:
            while True:
                data = f.read(CHUNK)
                if not data:
                    break
                if d.eof:
                    raise EvidenceError("decompress", gz_rel, "trailing bytes after the gzip stream")
                while data:
                    out = d.decompress(data, CHUNK)
                    n += len(out)
                    if n > raw_bytes:
                        raise EvidenceError("decompress", gz_rel,
                                            f"output exceeds declared raw size {raw_bytes}")
                    h.update(out)
                    if sink is not None:
                        sink.write(out)
                    data = d.unconsumed_tail
                    if d.eof:
                        if data or d.unused_data:
                            raise EvidenceError("decompress", gz_rel,
                                                "trailing bytes after the gzip stream")
                        break
    except zlib.error as e:
        raise EvidenceError("decompress", gz_rel, f"malformed or corrupt gzip ({e})")
    if not d.eof:
        raise EvidenceError("decompress", gz_rel, "truncated gzip stream")
    if n != raw_bytes:
        raise EvidenceError("decompress", gz_rel, f"raw size {n} != declared {raw_bytes}")
    if h.hexdigest() != raw_sha:
        raise EvidenceError("decompress", gz_rel, "raw SHA-256 differs from the data manifest")


def raw_state(root, raw_rel, raw_bytes, raw_sha):
    """'absent', 'ok', or raise EvidenceError for anything else at the raw path."""
    path = _checked_path(root, raw_rel, "check-raw")
    if not os.path.lexists(path):
        return "absent"
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        raise EvidenceError("check-raw", raw_rel, "is a symlink; refusing")
    if not stat.S_ISREG(st.st_mode):
        raise EvidenceError("check-raw", raw_rel, "exists and is not a regular file")
    if st.st_size != raw_bytes:
        raise EvidenceError("check-raw", raw_rel,
                            f"existing file has size {st.st_size}, expected {raw_bytes}; left untouched")
    if _sha_file(path)[1] != raw_sha:
        raise EvidenceError("check-raw", raw_rel, "existing file SHA-256 differs; left untouched")
    return "ok"


def verify(root=ROOT, log=print):
    """Read-only: compressed files, decompressed streams, and any raw file already present."""
    for gz_rel, gz_bytes, gz_sha, raw_rel, raw_bytes, raw_sha in load_manifest(root):
        gz_path = check_compressed(root, gz_rel, gz_bytes, gz_sha)
        stream_raw(gz_path, gz_rel, raw_bytes, raw_sha)
        state = raw_state(root, raw_rel, raw_bytes, raw_sha)
        log(f"ok  {gz_rel} ({gz_bytes} bytes) -> {raw_rel} ({raw_bytes} bytes, sha256 {raw_sha[:12]}...)"
            f"  raw file: {'present, verified' if state == 'ok' else 'not unpacked'}")


def unpack(root=ROOT, log=print):
    """Verify, then write each missing raw file. Returns the list of files written."""
    entries = load_manifest(root)
    plan = []
    for gz_rel, gz_bytes, gz_sha, raw_rel, raw_bytes, raw_sha in entries:  # preflight, no writes
        gz_path = check_compressed(root, gz_rel, gz_bytes, gz_sha)
        state = raw_state(root, raw_rel, raw_bytes, raw_sha)
        plan.append((gz_path, gz_rel, raw_rel, raw_bytes, raw_sha, state))
    written = []
    for gz_path, gz_rel, raw_rel, raw_bytes, raw_sha, state in plan:
        if state == "ok":
            log(f"ok  {raw_rel} already present with the declared size and SHA-256")
            continue
        try:
            _materialize(root, gz_path, gz_rel, raw_rel, raw_bytes, raw_sha)
        except EvidenceError as e:
            e.materialized = list(written)
            raise
        written.append(raw_rel)
        log(f"wrote {raw_rel} ({raw_bytes} bytes, sha256 verified)")
    return written


def _materialize(root, gz_path, gz_rel, raw_rel, raw_bytes, raw_sha):
    dest = _checked_path(root, raw_rel, "place")
    fd, tmp = tempfile.mkstemp(prefix=".unpack-", suffix=".tmp", dir=os.path.dirname(dest))
    try:
        with os.fdopen(fd, "wb") as out:
            stream_raw(gz_path, gz_rel, raw_bytes, raw_sha, sink=out)
            out.flush()
            os.fsync(out.fileno())
        mask = os.umask(0)
        os.umask(mask)
        os.chmod(tmp, 0o666 & ~mask)  # ordinary file mode, not mkstemp's 0600
        if _sha_file(tmp) != (raw_bytes, raw_sha):
            raise EvidenceError("place", raw_rel, "temporary file failed re-check")
        _checked_path(root, raw_rel, "place")  # parents still real directories
        try:
            os.link(tmp, dest)  # fails if dest was occupied meanwhile; never clobbers
        except FileExistsError:
            raise EvidenceError("place", raw_rel, "destination became occupied; left untouched")
    finally:
        if os.path.lexists(tmp):
            os.unlink(tmp)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if args not in ([], ["--verify"]):
        print("usage: unpack_evidence.py [--verify]", file=sys.stderr)
        return 2
    try:
        if args == ["--verify"]:
            verify(ROOT)
            print("EVIDENCE OK: compressed files and raw streams match the data manifest (nothing written)")
        else:
            written = unpack(ROOT)
            print(f"UNPACK OK: {len(written)} file(s) written, "
                  f"{len(RAW_PATHS) - len(written)} already present and verified")
    except EvidenceError as e:
        done = getattr(e, "materialized", [])
        print(f"FAILED {e}", file=sys.stderr)
        print("raw files materialized by this run before the failure: "
              + (", ".join(done) if done else "none"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
