"""Regression tests: legacy experiment drivers must reject unknown flags without training
or writing results.

Defect fixed for release: run_metabucket_v1.py (and run_metabucket_td_v1.py) treated any
unrecognised flag as "run the full experiment", which retrains and overwrites
results/*.json. These tests never touch the real results: subprocess checks run a copy of
each driver inside a temporary directory with a fixture results file, and in-process
checks replace the training function with one that fails if called.
"""
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVERS = {
    "run_metabucket_v1.py": ("metabucket_v1.json", ["--smoke"]),
    "run_metabucket_td_v1.py": ("metabucket_td_v1.json", ["--smoke", "--gate"]),
}
UNKNOWN = [["--gate"], ["--bogus"], ["--smoke", "--extra"], ["--run"], ["-h"]]


def _load(name):
    spec = importlib.util.spec_from_file_location("drv_" + name[:-3], os.path.join(ROOT, "experiments", name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestParseMode(unittest.TestCase):
    def test_valid_invocations_preserved(self):
        for name, (_res, valid) in DRIVERS.items():
            mod = _load(name)
            self.assertEqual(mod._parse_mode(["prog"]), "run", name)
            for flag in valid:
                self.assertEqual(mod._parse_mode(["prog", flag]), flag[2:], name)

    def test_unknown_flags_rejected_without_training(self):
        for name, (_res, valid) in DRIVERS.items():
            mod = _load(name)
            for args in UNKNOWN:
                if len(args) == 1 and args[0] in valid:
                    continue
                self.assertIsNone(mod._parse_mode(["prog"] + args), (name, args))
                with mock.patch.object(mod, "run_arm", side_effect=AssertionError("trained")), \
                        mock.patch.object(mod, "smoke", side_effect=AssertionError("smoke")), \
                        mock.patch.object(sys, "argv", ["prog"] + args), \
                        mock.patch("sys.stderr"):
                    with self.assertRaises(SystemExit) as cm:
                        mod.main()
                    self.assertEqual(cm.exception.code, 2)


class TestIsolatedSubprocess(unittest.TestCase):
    def test_unknown_flag_leaves_fixture_bytes_unchanged(self):
        fixture = b'{"fixture": "must not change"}'
        digest = hashlib.sha256(fixture).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(os.path.join(ROOT, "remnant_core"), os.path.join(tmp, "remnant_core"),
                            ignore=shutil.ignore_patterns("__pycache__"))
            os.makedirs(os.path.join(tmp, "experiments"))
            os.makedirs(os.path.join(tmp, "results"))
            for name, (res, valid) in DRIVERS.items():
                shutil.copy(os.path.join(ROOT, "experiments", name), os.path.join(tmp, "experiments", name))
                res_path = os.path.join(tmp, "results", res)
                with open(res_path, "wb") as f:
                    f.write(fixture)
                for args in (["--bogus"], ["--smoke", "--extra"]) + ((["--gate"],) if "--gate" not in valid else ()):
                    p = subprocess.run([sys.executable, os.path.join(tmp, "experiments", name)] + args,
                                       cwd=tmp, capture_output=True, text=True, timeout=60)
                    self.assertEqual(p.returncode, 2, (name, args, p.stdout, p.stderr))
                    self.assertEqual(p.stdout, "", (name, args))
                    self.assertIn("usage:", p.stderr)
                    with open(res_path, "rb") as f:
                        self.assertEqual(hashlib.sha256(f.read()).hexdigest(), digest, (name, args))
                self.assertEqual(sorted(os.listdir(os.path.join(tmp, "results"))),
                                 sorted(r for r, _v in list(DRIVERS.values())[:list(DRIVERS).index(name) + 1]))


if __name__ == "__main__":
    unittest.main()
