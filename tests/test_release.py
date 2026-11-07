"""Release checks: verifier logic, derived analysis, summary hashes, docs links, demo."""
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analysis import paired_return_tally_v1 as paired  # noqa: E402


def _load_tool(name):
    spec = importlib.util.spec_from_file_location("tool_" + name, os.path.join(ROOT, "tools", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TestVerifier(unittest.TestCase):
    def setUp(self):
        self.v = _load_tool("verify_release")

    def _tree(self, tmp):
        for rel, data in {"a.txt": b"a", "d/b.txt": b"b", "work_done.md": b"private",
                          "d/__pycache__/x.pyc": b"c"}.items():
            p = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(data)

    def test_clean_tree_and_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._tree(tmp)
            self.assertEqual(self.v.tree_files(tmp), ["a.txt", "d/b.txt"])
            self.v.write(tmp)
            listed, problems = self.v.verify(tmp)
            self.assertEqual(problems, [])
            self.assertNotIn(self.v.MANIFEST, listed)

    def test_detects_mismatch_missing_unexpected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._tree(tmp)
            self.v.write(tmp)
            with open(os.path.join(tmp, "a.txt"), "wb") as f:
                f.write(b"changed")
            os.remove(os.path.join(tmp, "d/b.txt"))
            with open(os.path.join(tmp, "new.txt"), "wb") as f:
                f.write(b"n")
            _l, problems = self.v.verify(tmp)
            self.assertEqual(sorted(problems), [("MISMATCH", "a.txt"), ("MISSING", "d/b.txt"),
                                                ("UNEXPECTED", "new.txt")])


class TestDerivedAnalysis(unittest.TestCase):
    def test_paired_analysis_recomputes_from_rows(self):
        with open(os.path.join(ROOT, "results", "return_tally_v1.json"), "rb") as f:
            raw = f.read()
        res = paired.analyse(json.loads(raw))
        res["source_sha256"] = hashlib.sha256(raw).hexdigest()
        with open(os.path.join(ROOT, "analysis", "paired_return_tally_v1.json")) as f:
            shipped = json.load(f)
        self.assertEqual(json.loads(json.dumps(res)), shipped)
        t = res["posthoc_paired_sign_test"]
        self.assertEqual((t["q_curious"]["wins"], t["q_curious"]["losses"]), (29, 0))
        self.assertEqual(t["q_curious"]["p_one_sided"], 0.5 ** 29)
        for c in ("ec_amnesic", "ec_only"):
            self.assertEqual((t[c]["wins"], t[c]["losses"]), (30, 0))
            self.assertEqual(t[c]["p_one_sided"], 0.5 ** 30)
        self.assertEqual(res["arms"]["ec_curious"]["wilson95"], [0.8865, 1.0])

    def test_sign_test(self):
        self.assertEqual(paired.sign_test_one_sided(0, 0), 1.0)
        self.assertAlmostEqual(paired.sign_test_one_sided(2, 1), 0.5)


class TestSummaryAndDocs(unittest.TestCase):
    def test_summary_artifact_hashes(self):
        with open(os.path.join(ROOT, "study_summary.json")) as f:
            s = json.load(f)
        self.assertEqual(s["artifacts"]["results/return_tally_v1.json"],
                         "b3c6a3ac2bbd830ffd269e203e73e8e2a33ea4853e871a783d6850666345d9a9")
        for rel, digest in s["artifacts"].items():
            self.assertEqual(_sha(os.path.join(ROOT, rel)), digest, rel)
        for rel in s["docs"].values():
            self.assertTrue(os.path.exists(os.path.join(ROOT, rel)), rel)

    def test_relative_markdown_links_resolve(self):
        link = re.compile(r"\]\(([^)\s]+)\)")
        bad = []
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", ".venv")]
            for name in filenames:
                if not name.endswith(".md") or name == "work_done.md":
                    continue
                path = os.path.join(dirpath, name)
                with open(path, encoding="utf-8") as f:
                    text = f.read()
                for target in link.findall(text):
                    if target.startswith(("http://", "https://", "#", "mailto:")):
                        continue
                    target = target.split("#")[0]
                    if not os.path.exists(os.path.normpath(os.path.join(dirpath, target))):
                        bad.append((os.path.relpath(path, ROOT), target))
        self.assertEqual(bad, [])


class TestDemo(unittest.TestCase):
    def test_demo_runs_frozen_policy(self):
        p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "demo_frozen_policy.py")],
                           cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("frozen deployment: 5/5 tides escaped; first tide 42 strides", p.stdout)
        self.assertIn("matches accepted results/return_tally_v1.json for this seed: True", p.stdout)


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed (optional)")
class TestFigures(unittest.TestCase):
    def test_figures_render_outside_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "figures_return_tally_v1.py"),
                                "--out", tmp], cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(sorted(os.listdir(tmp)), ["fig1_endpoint_by_arm.png",
                                                       "fig2_discovery_and_learning.png",
                                                       "fig3_frozen_route.png"])


if __name__ == "__main__":
    unittest.main()
