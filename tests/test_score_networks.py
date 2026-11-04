"""Oracle tests for score_networks.py: the released *_hand.npz fixtures, scored with the sweep's
seeds, must reproduce the first records of results/<game>.episodes.json exactly."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import sweep  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
RESULTS = os.path.join(ROOT, "results")
SCRIPT = os.path.join(ROOT, "score_networks.py")
GAMES = ["breakout", "seaquest", "asterix"]
EPISODES = 20
NEW_KEYS = {"source_npz", "source_sha256"}


def fixture(game):
    return os.path.join(FIXTURES, f"{game}_hand.npz")


def run_scorer(*args):
    return subprocess.run(
        [sys.executable, SCRIPT, *args], cwd=ROOT, capture_output=True, text=True
    )


class OracleTests(unittest.TestCase):
    """One scorer run per game at 20 episodes, compared record-by-record against the results."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.outputs = {}
        t0 = time.monotonic()
        for game in GAMES:
            p = run_scorer("--game", game, "--npz", fixture(game), "--out-dir", cls.tmp,
                           "--episodes", str(EPISODES))
            assert p.returncode == 0, p.stderr
            with open(os.path.join(cls.tmp, f"{game}_hand.episodes.json")) as fh:
                cls.outputs[game] = json.load(fh)
        print(f"\n[score_networks oracles: {len(GAMES)} games x {EPISODES} episodes in "
              f"{time.monotonic() - t0:.1f}s]", flush=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def check_game(self, game):
        with open(os.path.join(RESULTS, f"{game}.episodes.json")) as fh:
            expected = json.load(fh)["records"][:EPISODES]
        got = self.outputs[game]["records"]
        self.assertEqual(len(got), EPISODES)
        for e, g in zip(expected, got):
            self.assertEqual(g["ep"], e["ep"])
            self.assertEqual(g["returns"], e["returns"])
            self.assertEqual(g["length"], e["length"])
            self.assertEqual(g["censored"], e["censored"])

    def test_breakout_oracle(self):
        self.check_game("breakout")

    def test_seaquest_oracle(self):
        self.check_game("seaquest")

    def test_asterix_oracle(self):
        self.check_game("asterix")

    def test_schema_matches_results_plus_source_fields(self):
        with open(os.path.join(RESULTS, "breakout.episodes.json")) as fh:
            ref = json.load(fh)
        out = self.outputs["breakout"]
        self.assertEqual(set(out), set(ref) | NEW_KEYS)
        self.assertEqual(set(out["records"][0]), set(ref["records"][0]))
        self.assertEqual(out["source_npz"], fixture("breakout"))
        self.assertEqual(len(out["source_sha256"]), 64)

    def test_refuses_to_overwrite_without_force(self):
        target = os.path.join(self.tmp, "breakout_hand.episodes.json")
        with open(target, "rb") as fh:
            before = fh.read()
        p = run_scorer("--game", "breakout", "--npz", fixture("breakout"), "--out-dir", self.tmp,
                       "--episodes", "1")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("refusing to overwrite", p.stderr)
        with open(target, "rb") as fh:
            self.assertEqual(fh.read(), before)

    def test_rejects_duplicate_output_paths_even_with_force(self):
        """Two inputs with the same basename in different directories map to one output path."""
        src = tempfile.mkdtemp()
        out = tempfile.mkdtemp()
        try:
            a = os.path.join(src, "a", "net.npz")
            b = os.path.join(src, "b", "net.npz")
            for p in (a, b):
                os.makedirs(os.path.dirname(p))
                shutil.copy(fixture("breakout"), p)
            for extra in ((), ("--force",)):
                p = run_scorer("--game", "breakout", "--npz", a, b, "--out-dir", out,
                               "--episodes", "1", *extra)
                self.assertNotEqual(p.returncode, 0)
                self.assertIn("duplicate output paths", p.stderr)
                self.assertFalse(os.path.exists(os.path.join(out, "net.episodes.json")))
        finally:
            shutil.rmtree(src, ignore_errors=True)
            shutil.rmtree(out, ignore_errors=True)


class LoadNetTests(unittest.TestCase):
    SHAPES = [(600, 32), (32,), (32, 32), (32,), (32, 6), (6,)]

    def test_load_net_by_path(self):
        net = sweep.load_net(fixture("breakout"))
        self.assertEqual([t.shape for t in net], self.SHAPES)

    def test_load_net_by_game_still_works(self):
        tmp = tempfile.mkdtemp()
        try:
            shutil.copy(fixture("breakout"), os.path.join(tmp, "breakout.hand.npz"))
            saved = sweep.NET_DIR
            sweep.NET_DIR = tmp
            try:
                by_game = sweep.load_net("breakout")
            finally:
                sweep.NET_DIR = saved
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        by_path = sweep.load_net(fixture("breakout"))
        for g, p in zip(by_game, by_path):
            np.testing.assert_array_equal(g, p)


class CliTests(unittest.TestCase):
    def test_help(self):
        p = run_scorer("--help")
        self.assertEqual(p.returncode, 0)
        self.assertIn("--npz", p.stdout)


if __name__ == "__main__":
    unittest.main()
