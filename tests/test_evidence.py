"""Compressed evidence: tools/unpack_evidence.py and the raw-aware tools/verify_release.py.

Failure cases run on small isolated fixture trees in temporary directories, never on the
real result files. The real compressed files are only read (stream check, no writes).
"""
import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_tool(name):
    spec = importlib.util.spec_from_file_location("evtest_" + name, os.path.join(ROOT, "tools", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load_tool("unpack_evidence")
V = _load_tool("verify_release")
RAW = {"results/records.json": b'{"records": [' + b'1, ' * 5000 + b'2]}\n',
       "results/learners.json": b'{"learners": "' + b'q' * 3000 + b'"}\n'}


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _gz(data):
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as g:
        g.write(data)
    return buf.getvalue()


def _read(path):
    with open(path, "rb") as f:
        return f.read()


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


class Fixture:
    """A minimal source tree: two .gz files, the data manifest, one other result file."""

    def __init__(self, tmp, gz_override=None, manifest_edit=None):
        self.root = tmp
        self.gz = {}
        entries = []
        for gz_rel, raw_rel in U.FIXED_PAIRS:
            data = (gz_override or {}).get(gz_rel, _gz(RAW[raw_rel]))
            self.gz[gz_rel] = data
            _write(os.path.join(tmp, gz_rel), data)
            entries.append({"compressed": {"path": gz_rel, "bytes": len(data), "sha256": _sha(data)},
                            "raw": {"path": raw_rel, "bytes": len(RAW[raw_rel]), "sha256": _sha(RAW[raw_rel])}})
        doc = {"schema": "test", "entries": entries}
        if manifest_edit:
            manifest_edit(doc)
        _write(os.path.join(tmp, U.DATA_MANIFEST), json.dumps(doc).encode())
        _write(os.path.join(tmp, "results", "return_tally_v1.json"), b"{}")

    def path(self, rel):
        return os.path.join(self.root, rel)

    def leftovers(self):
        return [n for n in os.listdir(self.path("results")) if n.startswith(".unpack-")]


def quiet(*_a, **_k):
    pass


class TestUnpackHappyPath(unittest.TestCase):
    def test_clean_compressed_tree_verifies_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            before = sorted(os.listdir(fx.path("results")))
            U.verify(tmp, log=quiet)
            self.assertEqual(sorted(os.listdir(fx.path("results"))), before)

    def test_round_trip_and_idempotence(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            self.assertEqual(U.unpack(tmp, log=quiet), list(U.RAW_PATHS))
            for rel, data in RAW.items():
                self.assertEqual(_sha(_read(fx.path(rel))), _sha(data))
            self.assertEqual(U.unpack(tmp, log=quiet), [])  # already correct: success, no writes
            U.verify(tmp, log=quiet)
            self.assertEqual(fx.leftovers(), [])

    def test_real_compressed_evidence_round_trips_to_raw_hashes(self):
        entries = U.load_manifest(ROOT)
        self.assertEqual([(e[0], e[3]) for e in entries], list(U.FIXED_PAIRS))
        expected = {"results/records.json": (277819583, "7f8f0c7c9c3e5dd7e0a1ebf9f697f95571c50293634898c3e68ee56825a40498"),
                    "results/learners.json": (65805407, "3bd4e1f067e0d0ad0e908ce497c291dab631c47f5554ec872f11b8825e316a39")}
        for gz_rel, gz_bytes, gz_sha, raw_rel, raw_bytes, raw_sha in entries:
            self.assertEqual((raw_bytes, raw_sha), expected[raw_rel])
            gz_path = U.check_compressed(ROOT, gz_rel, gz_bytes, gz_sha)
            self.assertLess(gz_bytes, 100 * 1024 * 1024)
            U.stream_raw(gz_path, gz_rel, raw_bytes, raw_sha)  # read-only


class TestUnpackRejections(unittest.TestCase):
    def assertFails(self, fn, phase):
        with self.assertRaises(U.EvidenceError) as cm:
            fn()
        self.assertEqual(cm.exception.phase, phase, str(cm.exception))
        return cm.exception

    def test_missing_compressed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            os.remove(fx.path("results/learners.json.gz"))
            self.assertFails(lambda: U.verify(tmp, log=quiet), "verify-compressed")
            self.assertFails(lambda: U.unpack(tmp, log=quiet), "verify-compressed")
            self.assertFalse(os.path.lexists(fx.path("results/records.json")))  # preflight wrote nothing

    def test_compressed_hash_or_size_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            data = bytearray(fx.gz["results/records.json.gz"])
            data[20] ^= 0xFF
            _write(fx.path("results/records.json.gz"), bytes(data))
            e = self.assertFails(lambda: U.verify(tmp, log=quiet), "verify-compressed")
            self.assertIn("SHA-256", e.reason)
        with tempfile.TemporaryDirectory() as tmp:
            Fixture(tmp, manifest_edit=lambda d: d["entries"][0]["compressed"].update(bytes=1))
            e = self.assertFails(lambda: U.verify(tmp, log=quiet), "verify-compressed")
            self.assertIn("size", e.reason)

    def test_transport_corruption_with_matching_manifest(self):
        good = _gz(RAW["results/records.json"])
        flipped = bytearray(good)
        flipped[len(good) // 2] ^= 0x55  # deflate body
        bad_crc = bytearray(good)
        bad_crc[-8] ^= 0x01  # CRC32 trailer
        for bad in (bytes(flipped), bytes(bad_crc)):
            with tempfile.TemporaryDirectory() as tmp:
                fx = Fixture(tmp, gz_override={"results/records.json.gz": bad})
                self.assertFails(lambda: U.verify(tmp, log=quiet), "decompress")
                self.assertFails(lambda: U.unpack(tmp, log=quiet), "decompress")
                self.assertFalse(os.path.lexists(fx.path("results/records.json")))
                self.assertEqual(fx.leftovers(), [])

    def test_malformed_truncated_and_trailing_gzip(self):
        good = _gz(RAW["results/records.json"])
        cases = {"not gzip": b"this is not gzip data at all",
                 "truncated": good[: len(good) - 12],
                 "trailing garbage": good + b"garbage",
                 "trailing zeros": good + b"\x00" * 16,
                 "second member": good + _gz(b"extra")}
        for label, bad in cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                fx = Fixture(tmp, gz_override={"results/records.json.gz": bad})
                self.assertFails(lambda: U.verify(tmp, log=quiet), "decompress")
                self.assertFails(lambda: U.unpack(tmp, log=quiet), "decompress")
                self.assertFalse(os.path.lexists(fx.path("results/records.json")))
                self.assertEqual(fx.leftovers(), [])

    def test_wrong_raw_size_and_hash_and_expansion_ceiling(self):
        n = len(RAW["results/records.json"])
        edits = {"larger declared": lambda d: d["entries"][0]["raw"].update(bytes=n + 1),
                 "ceiling": lambda d: d["entries"][0]["raw"].update(bytes=n - 1),
                 "raw hash": lambda d: d["entries"][0]["raw"].update(sha256="0" * 64)}
        for label, edit in edits.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                fx = Fixture(tmp, manifest_edit=edit)
                self.assertFails(lambda: U.unpack(tmp, log=quiet), "decompress")
                self.assertFalse(os.path.lexists(fx.path("results/records.json")))
                self.assertEqual(fx.leftovers(), [])

    def test_manifest_cannot_choose_paths(self):
        bad_paths = ["../escape.json", "/tmp/escape.json", "results/other.json", "results/records.json.gz"]
        for p in bad_paths:
            with self.subTest(p), tempfile.TemporaryDirectory() as tmp:
                Fixture(tmp, manifest_edit=lambda d, p=p: d["entries"][0]["raw"].update(path=p))
                self.assertFails(lambda: U.unpack(tmp, log=quiet), "manifest")
                self.assertFalse(os.path.lexists(os.path.join(os.path.dirname(tmp), "escape.json")))
        with tempfile.TemporaryDirectory() as tmp:
            Fixture(tmp, manifest_edit=lambda d: d["entries"].append(d["entries"][0]))
            self.assertFails(lambda: U.verify(tmp, log=quiet), "manifest")
        with self.assertRaises(U.EvidenceError):
            U._checked_path(ROOT, "../results/records.json", "x")

    def test_existing_mismatched_raw_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            _write(fx.path("results/learners.json"), b"someone else's file")
            self.assertFails(lambda: U.unpack(tmp, log=quiet), "check-raw")
            self.assertEqual(_read(fx.path("results/learners.json")), b"someone else's file")
            self.assertFalse(os.path.lexists(fx.path("results/records.json")))  # preflight before any write
            self.assertEqual(fx.leftovers(), [])

    def test_existing_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            os.mkdir(fx.path("results/records.json"))
            self.assertFails(lambda: U.unpack(tmp, log=quiet), "check-raw")
            self.assertTrue(os.path.isdir(fx.path("results/records.json")))

    @unittest.skipUnless(hasattr(os, "symlink"), "no symlinks")
    def test_symlinked_raw_and_parent_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            fx = Fixture(tmp)
            target = os.path.join(outside, "target.json")
            _write(target, RAW["results/records.json"])
            os.symlink(target, fx.path("results/records.json"))
            self.assertFails(lambda: U.unpack(tmp, log=quiet), "check-raw")
            self.assertFails(lambda: U.verify(tmp, log=quiet), "check-raw")
            self.assertEqual(_read(target), RAW["results/records.json"])
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            fx = Fixture(tmp)
            real = fx.path("results")
            moved = os.path.join(outside, "results")
            os.rename(real, moved)
            os.symlink(moved, real)
            self.assertFails(lambda: U.unpack(tmp, log=quiet), "verify-compressed")
            self.assertEqual(sorted(os.listdir(moved)),
                             ["learners.json.gz", "records.json.gz", "return_tally_v1.json"])

    def test_partial_failure_reports_what_was_materialized(self):
        good = _gz(RAW["results/learners.json"])
        bad = good[:-8] + b"\x00\x00\x00\x00" + good[-4:]  # wrong CRC, manifest hash matches file
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp, gz_override={"results/learners.json.gz": bad})
            with self.assertRaises(U.EvidenceError) as cm:
                U.unpack(tmp, log=quiet)
            self.assertEqual(cm.exception.phase, "decompress")
            self.assertEqual(cm.exception.materialized, ["results/records.json"])
            self.assertEqual(_read(fx.path("results/records.json")), RAW["results/records.json"])
            self.assertFalse(os.path.lexists(fx.path("results/learners.json")))
            self.assertEqual(fx.leftovers(), [])

    def test_destination_occupied_before_placement_is_not_clobbered(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            original = U.stream_raw

            def racing(gz_path, gz_rel, raw_bytes, raw_sha, sink=None):
                original(gz_path, gz_rel, raw_bytes, raw_sha, sink=sink)
                if gz_rel == "results/records.json.gz":
                    _write(fx.path("results/records.json"), b"arrived meanwhile")

            U.stream_raw = racing
            try:
                with self.assertRaises(U.EvidenceError) as cm:
                    U.unpack(tmp, log=quiet)
            finally:
                U.stream_raw = original
            self.assertEqual(cm.exception.phase, "place")
            self.assertEqual(cm.exception.materialized, [])
            self.assertEqual(_read(fx.path("results/records.json")), b"arrived meanwhile")
            self.assertEqual(fx.leftovers(), [])


class TestVerifyReleaseWithEvidence(unittest.TestCase):
    def _release(self, tmp):
        fx = Fixture(tmp)
        V.write(tmp)
        return fx

    def test_default_mode_clean_and_hydrated(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = self._release(tmp)
            listed, problems = V.verify(tmp)
            self.assertEqual(problems, [])
            self.assertNotIn("results/records.json", listed)
            self.assertIn("results/records.json.gz", listed)
            self.assertIn(U.DATA_MANIFEST, listed)
            self.assertEqual(sorted(V.verify(tmp, require_raw=True)[1]),
                             [("RAW-MISSING", "results/learners.json"), ("RAW-MISSING", "results/records.json")])
            U.unpack(tmp, log=quiet)
            self.assertEqual(V.verify(tmp)[1], [])
            self.assertEqual(V.verify(tmp, require_raw=True)[1], [])
            self.assertNotIn("results/records.json", V.tree_files(tmp))
            self.assertTrue(os.path.isfile(fx.path("results/records.json")))

    def test_corrupt_raw_is_checked_not_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = self._release(tmp)
            U.unpack(tmp, log=quiet)
            data = bytearray(RAW["results/records.json"])
            data[5] ^= 1
            _write(fx.path("results/records.json"), bytes(data))
            self.assertEqual(V.verify(tmp)[1], [("RAW-MISMATCH", "results/records.json")])
            self.assertEqual(V.verify(tmp, require_raw=True)[1], [("RAW-MISMATCH", "results/records.json")])

    def test_unrelated_extra_results_file_is_unexpected(self):
        for extra in ("results/other.json", "results/records.json.bak"):
            with self.subTest(extra), tempfile.TemporaryDirectory() as tmp:
                fx = self._release(tmp)
                _write(fx.path(extra), b"{}")
                self.assertEqual(V.verify(tmp)[1], [("UNEXPECTED", extra)])

    def test_case_variant_of_raw_evidence_path(self):
        # "results/Records.json" only differs from the protected raw path by case.
        # On case-insensitive filesystems (default macOS APFS, Windows NTFS) the
        # decoy occupies the raw path itself, so the verifier must flag both the
        # unexpected name and the raw mismatch. On case-sensitive filesystems the
        # two names are distinct files and only the unexpected name is reported.
        with tempfile.TemporaryDirectory() as probe:
            _write(os.path.join(probe, "CiPrObE.tmp"), b"")
            case_insensitive = os.path.exists(os.path.join(probe, "ciprobe.TMP"))
        with tempfile.TemporaryDirectory() as tmp:
            fx = self._release(tmp)
            _write(fx.path("results/Records.json"), b"{}")
            problems = V.verify(tmp)[1]
            if case_insensitive:
                self.assertEqual(sorted(problems), sorted([
                    ("UNEXPECTED", "results/Records.json"),
                    ("RAW-MISMATCH", "results/records.json"),
                ]))
            else:
                self.assertEqual(problems, [("UNEXPECTED", "results/Records.json")])

    @unittest.skipUnless(hasattr(os, "symlink"), "no symlinks")
    def test_linked_raw_never_passes(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            fx = self._release(tmp)
            target = os.path.join(outside, "records.json")
            _write(target, RAW["results/records.json"])
            os.symlink(target, fx.path("results/records.json"))
            self.assertIn(("LINK", "results/records.json"), V.verify(tmp)[1])
            raw_problems = V.verify(tmp, require_raw=True)[1]
            self.assertIn(("RAW-MISSING", "results/records.json"), raw_problems)
            with self.assertRaises(SystemExit):
                V.write(tmp)

    def test_shipped_data_manifest_matches_tool_and_release(self):
        entries = U.load_manifest(ROOT)
        listed = V.read_manifest(ROOT)
        for gz_rel, gz_bytes, gz_sha, raw_rel, _rb, _rs in entries:
            self.assertEqual(listed[gz_rel], gz_sha)
            self.assertEqual(os.path.getsize(os.path.join(ROOT, gz_rel)), gz_bytes)
            self.assertNotIn(raw_rel, listed)
        with open(os.path.join(ROOT, ".gitignore")) as f:
            ignored = f.read().split("\n")
        for raw in U.RAW_PATHS:
            self.assertIn("/" + raw, ignored)
        self.assertNotIn("/results/records.json.gz", ignored)


class TestCli(unittest.TestCase):
    def _run(self, tool, *args):
        return subprocess.run([sys.executable, os.path.join(ROOT, "tools", tool), *args],
                              cwd=ROOT, capture_output=True, text=True, timeout=120)

    def test_unknown_flags_rejected(self):
        for tool, args in (("unpack_evidence.py", ["--bogus"]), ("unpack_evidence.py", ["--verify", "x"]),
                           ("verify_release.py", ["--bogus"]), ("verify_release.py", ["--raw", "--write"])):
            p = self._run(tool, *args)
            self.assertEqual(p.returncode, 2, (tool, args, p.stdout, p.stderr))
            self.assertIn("usage", p.stderr)

    def test_failure_exit_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            Fixture(tmp, gz_override={"results/records.json.gz": b"not gzip"})
            old = U.ROOT
            U.ROOT = tmp
            try:
                err = io.StringIO()
                with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                    rc = U.main([])
            finally:
                U.ROOT = old
            self.assertEqual(rc, 1)
            self.assertIn("phase=decompress", err.getvalue())
            self.assertIn("materialized by this run before the failure: none", err.getvalue())


if __name__ == "__main__":
    unittest.main()
