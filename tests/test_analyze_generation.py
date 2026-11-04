import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyze_generation as ag  # noqa: E402

GAME = "breakout"


def attempt(index, outcome="replied", block=True, applied=5, statements=5, cut_off=False,
            wall=500.0, sha=None, cached=1000):
    r = {"index": index, "game": GAME, "outcome": outcome, "wall_seconds": wall,
         "usage": {"input_tokens": 5000, "cached_input_tokens": cached, "output_tokens": 2000},
         "reply_sha256": None}
    if outcome == "replied":
        evaluable = block and applied >= 1
        r.update({"reply_sha256": sha or f"sha-{index}", "block_found": block, "cut_off": cut_off,
                  "statements": statements, "applied": applied, "clean": block and not cut_off and applied == statements,
                  "evaluable": evaluable})
    return r


def score_file(path, r10, lengths=None, censored=None, eps=None):
    """``eps[i]`` is the episode index written on record ``i``; records are written in list order."""
    r10 = np.asarray(r10, dtype=float)
    n = len(r10)
    lengths = np.full(n, 800) if lengths is None else lengths
    censored = np.zeros(n, dtype=bool) if censored is None else censored
    eps = list(range(n)) if eps is None else list(eps)
    recs = [{"ep": int(eps[i]), "returns": {str(h): float(r10[i]) for h in (100, 300, 1000, 3000, 10000, 30000)},
             "length": int(lengths[i]), "censored": bool(censored[i])} for i in range(n)]
    with open(path, "w") as f:
        json.dump({"game": GAME, "episodes": n, "max_steps": 30000, "records": recs}, f)


class Fixture:
    """Writes attempts, score files and a released file into a temp dir and runs the CLI."""

    def __init__(self, attempts, means, released_mean, n_ep=50, ppo100="10:1", ppo500="100:1", markdown=False,
                 released_n_ep=None):
        self.d = tempfile.TemporaryDirectory()
        root = self.d.name
        os.makedirs(os.path.join(root, "att"))
        os.makedirs(os.path.join(root, "sc"))
        for a in attempts:
            with open(os.path.join(root, "att", f"attempt-{a['index']}.json"), "w") as f:
                json.dump(a, f)
            if a.get("evaluable"):
                # a scalar mean gives a constant score file; an array of length n_ep is written as-is
                score_file(os.path.join(root, "sc", f"attempt-{a['index']}.episodes.json"),
                           np.broadcast_to(np.asarray(means[a["index"]], dtype=float), (n_ep,)))
        score_file(os.path.join(root, "released.json"), np.full(released_n_ep or n_ep, released_mean))
        self.out = os.path.join(root, "report.json")
        self.md = os.path.join(root, "report.md")
        argv = ["--game", GAME, "--attempts", os.path.join(root, "att"), "--scores", os.path.join(root, "sc"),
                "--released", os.path.join(root, "released.json"), "--ppo100", ppo100, "--ppo500", ppo500,
                "--out", self.out]
        if markdown:
            argv += ["--markdown", self.md]
        assert ag.main(argv) == 0
        with open(self.out) as f:
            self.report = json.load(f)


def ten_attempts():
    """8 evaluable, 1 blockless reply, 1 transport failure, plus one not-submitted (index 11)."""
    atts = [attempt(i, wall=100.0 + 10 * i, cached=100 * i) for i in range(1, 9)]
    atts[0]["clean"] = False  # one evaluable but not clean
    atts.append(attempt(9, block=False, applied=0, statements=0, wall=190.0))
    atts.append(attempt(10, outcome="transport-failure", wall=200.0))
    atts.append(attempt(11, outcome="not-submitted"))
    return atts


class WilsonTests(unittest.TestCase):
    def test_eight_of_nine_matches_hand_formula(self):
        z, k, n = 1.959964, 8, 9
        p = k / n
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        w = ag.wilson(8, 9)
        self.assertAlmostEqual(w["rate"], p, places=4)
        self.assertAlmostEqual(w["low"], centre - half, places=4)
        self.assertAlmostEqual(w["high"], centre + half, places=4)
        self.assertAlmostEqual(w["low"], 0.5650, places=4)
        self.assertAlmostEqual(w["high"], 0.9801, places=4)

    def test_zero_denominator(self):
        w = ag.wilson(0, 0)
        self.assertIsNone(w["rate"])
        self.assertEqual(w["denominator"], 0)


class RateTests(unittest.TestCase):
    def test_rates_over_submitted_denominator(self):
        means = {i: 5.0 + i for i in range(1, 9)}
        fx = Fixture(ten_attempts(), means, released_mean=50.0, markdown=True)
        r = fx.report
        self.assertEqual(r["n_submitted"], 10)
        self.assertEqual(r["not_submitted_count"], 1)
        self.assertEqual(r["transport_count"], 1)
        self.assertEqual((r["evaluable_rate"]["count"], r["evaluable_rate"]["denominator"]), (8, 10))
        self.assertEqual((r["clean_rate"]["count"], r["clean_rate"]["denominator"]), (7, 10))
        self.assertEqual((r["replied_rate"]["count"], r["block_rate"]["count"]), (9, 8))
        self.assertAlmostEqual(r["evaluable_rate"]["rate"], 0.8)
        self.assertEqual(len(r["networks"]), 8)
        self.assertEqual(r["distribution"]["ecdf"], sorted(means.values()))
        self.assertEqual(r["distribution"]["min"], 6.0)
        self.assertEqual(r["distribution"]["max"], 13.0)
        self.assertAlmostEqual(r["distribution"]["median"], 9.5)
        # markdown: first section leads with rates, ECDF summary, percentile and claim status
        with open(fx.md) as f:
            md = f.read()
        head = md.split("## ")[1]  # text of the first section after the title
        self.assertTrue(head.startswith("Headline"))
        for needle in ("Evaluable rate", "Clean rate", "median", "IQR", "min", "max", "percentile", "NOT ALLOWED"):
            self.assertIn(needle, head)
        self.assertNotIn("mean across networks", md.lower())
        self.assertIn("## Thresholds", md)


class PercentileTests(unittest.TestCase):
    def _run(self, n, released, tie=False):
        atts = [attempt(i) for i in range(1, n + 1)]
        means = {i: float(i) for i in range(1, n + 1)}
        if tie:
            means[n] = released
        return Fixture(atts, means, released_mean=released).report["released"]

    def test_small_n_not_allowed(self):
        rel = self._run(10, released=100.0)
        self.assertEqual(rel["n_evaluable"], 10)
        self.assertEqual(rel["percentile"], 1.0)
        self.assertFalse(rel["percentile_claim_allowed"])
        self.assertIsNone(rel["percentile_claim_alpha"])

    def test_thirty_all_below_allowed(self):
        rel = self._run(30, released=100.0)
        self.assertTrue(rel["percentile_claim_allowed"])
        self.assertAlmostEqual(rel["percentile_claim_alpha"], 0.9 ** 30, places=6)
        self.assertEqual(rel["percentile"], 1.0)

    def test_tie_not_allowed(self):
        rel = self._run(30, released=100.0, tie=True)
        self.assertFalse(rel["percentile_claim_allowed"])
        self.assertAlmostEqual(rel["percentile"], 29 / 30)

    def test_attempt_level_counts_non_evaluable_as_below(self):
        rel = Fixture(ten_attempts(), {i: float(i) for i in range(1, 9)}, released_mean=4.5).report["released"]
        self.assertAlmostEqual(rel["percentile"], 4 / 8)
        self.assertAlmostEqual(rel["percentile_attempt_level"], (2 + 4) / 10)


class ThresholdTests(unittest.TestCase):
    def test_counts_and_near_boundary(self):
        # PPO 100M mean 10 (SE 1): near iff |mean-10| < 2; PPO 500M mean 100 (SE 1)
        means = {1: 5.0, 2: 9.5, 3: 11.0, 4: 20.0, 5: 100.5, 6: 150.0}
        atts = [attempt(i) for i in means]
        t = Fixture(atts, means, released_mean=200.0).report["thresholds"]
        self.assertEqual((t["exceeds_100m"]["count"], t["exceeds_100m"]["denominator"]), (4, 6))
        self.assertEqual(t["near_boundary_100m"]["count"], 2)
        self.assertEqual(t["exceeds_500m"]["count"], 2)
        self.assertEqual(t["near_boundary_500m"]["count"], 1)


class VarianceTests(unittest.TestCase):
    def test_recovers_planted_components(self):
        rng = np.random.RandomState(0)
        k, m = 30, 1000
        effects = rng.normal(0, 3.0, k)
        noise = rng.normal(0, 2.0, m)
        noise = (noise - noise.mean()) / noise.std(ddof=1) * 2.0      # exact within variance 4
        effects = (effects - effects.mean()) / effects.std(ddof=1) * 3.0  # exact variance of means 9
        panel = effects[:, None] + noise[None, :]
        v = ag.variance_components(panel)
        self.assertAlmostEqual(v["within_network_variance"], 4.0, delta=0.2)
        self.assertAlmostEqual(v["between_network_variance"], 9.0, delta=0.45)
        self.assertAlmostEqual(v["paired_mean_variance_across_networks"], 9.0, delta=0.45)
        self.assertAlmostEqual(v["ratio"], 9.0 / 4.0, delta=0.12)

    def test_no_between_variance_floors_at_zero(self):
        rng = np.random.RandomState(1)
        panel = rng.normal(0, 1, (5, 200))
        panel -= panel.mean(axis=1, keepdims=True)
        self.assertEqual(ag.variance_components(panel)["between_network_variance"], 0.0)

    def test_zero_networks(self):
        atts = [attempt(1, block=False, applied=0, statements=0), attempt(2, outcome="transport-failure")]
        v = Fixture(atts, {}, released_mean=50.0).report["variance_components"]
        self.assertEqual((v["n_networks"], v["n_episodes"]), (0, 0))
        for k in ("within_network_variance", "between_network_variance", "ratio",
                  "paired_mean_variance_across_networks"):
            self.assertIsNone(v[k])

    def test_one_network_reports_within_variance_only(self):
        rng = np.random.RandomState(2)
        r10 = rng.normal(5.0, 2.0, 50)
        v = Fixture([attempt(1)], {1: r10}, released_mean=50.0).report["variance_components"]
        self.assertEqual((v["n_networks"], v["n_episodes"]), (1, 50))
        self.assertAlmostEqual(v["within_network_variance"], float(r10.var(ddof=1)))
        for k in ("between_network_variance", "ratio", "paired_mean_variance_across_networks"):
            self.assertIsNone(v[k])


class PairedPanelTests(unittest.TestCase):
    """The panel pairs seeds by each record's ``ep``, never by array position."""

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.rng = np.random.RandomState(3)

    def _write(self, name, r10, eps):
        p = os.path.join(self.d.name, name)
        score_file(p, r10, eps=eps)
        return p

    def test_reordered_records_align_on_ep(self):
        m = 40
        a = self.rng.normal(0, 1, m)                 # network A, written in ep order
        b_by_ep = self.rng.normal(0, 1, m)           # network B, values indexed by ep
        perm = self.rng.permutation(m)               # B's file lists records in a scrambled order
        sa = ag.load_scores(self._write("a.json", a, eps=range(m)))
        sb = ag.load_scores(self._write("b.json", b_by_ep[perm], eps=perm))
        panel = ag.paired_panel([("a", sa), ("b", sb)])
        self.assertEqual(panel.shape, (2, m))
        np.testing.assert_allclose(panel[0], a)
        np.testing.assert_allclose(panel[1], b_by_ep)
        # positional stacking would have paired the wrong seeds
        self.assertFalse(np.allclose(sb["r10"], b_by_ep))
        # the paired across-network variance is therefore order-invariant
        sb_sorted = ag.load_scores(self._write("b2.json", b_by_ep, eps=range(m)))
        v1 = ag.variance_components(panel)
        v2 = ag.variance_components(ag.paired_panel([("a", sa), ("b", sb_sorted)]))
        self.assertAlmostEqual(v1["paired_mean_variance_across_networks"],
                               v2["paired_mean_variance_across_networks"])

    def test_reordered_records_through_cli(self):
        root = self.d.name
        os.makedirs(os.path.join(root, "att"))
        os.makedirs(os.path.join(root, "sc"))
        m = 30
        base = self.rng.normal(0, 1, m)
        for i, shift in ((1, 0.0), (2, 1.0), (3, 2.0)):
            with open(os.path.join(root, "att", f"attempt-{i}.json"), "w") as f:
                json.dump(attempt(i), f)
            perm = self.rng.permutation(m)
            score_file(os.path.join(root, "sc", f"attempt-{i}.episodes.json"), (base + shift)[perm], eps=perm)
        score_file(os.path.join(root, "released.json"), np.full(m, 50.0), eps=self.rng.permutation(m))
        out = os.path.join(root, "report.json")
        self.assertEqual(ag.main(["--game", GAME, "--attempts", os.path.join(root, "att"),
                                  "--scores", os.path.join(root, "sc"), "--released",
                                  os.path.join(root, "released.json"), "--ppo100", "10:1",
                                  "--ppo500", "100:1", "--out", out]), 0)
        with open(out) as f:
            v = json.load(f)["variance_components"]
        # networks differ by a constant per seed, so the paired across-network variance is exactly var([0,1,2])
        self.assertEqual((v["n_networks"], v["n_episodes"]), (3, m))
        self.assertAlmostEqual(v["paired_mean_variance_across_networks"], 1.0)

    def test_missing_episode_is_rejected_not_truncated(self):
        sa = ag.load_scores(self._write("a.json", np.zeros(10), eps=range(10)))
        sb = ag.load_scores(self._write("b.json", np.zeros(9), eps=range(9)))
        with self.assertRaises(SystemExit) as cm:
            ag.paired_panel([("a", sa), ("b", sb)])
        self.assertIn("missing [9]", str(cm.exception))

    def test_same_count_different_seeds_is_rejected(self):
        sa = ag.load_scores(self._write("a.json", np.zeros(10), eps=range(10)))
        sb = ag.load_scores(self._write("b.json", np.zeros(10), eps=range(1, 11)))
        with self.assertRaises(SystemExit):
            ag.paired_panel([("a", sa), ("b", sb)])

    def test_duplicate_episode_index_is_rejected(self):
        with self.assertRaises(SystemExit):
            ag.load_scores(self._write("a.json", np.zeros(3), eps=[0, 1, 1]))

    def test_released_file_must_share_the_seeds(self):
        atts = [attempt(1), attempt(2)]
        with self.assertRaises(SystemExit):
            Fixture(atts, {1: 1.0, 2: 2.0}, released_mean=50.0, n_ep=50, released_n_ep=49)


class ExchangeabilityTests(unittest.TestCase):
    def test_spearman_and_collisions(self):
        self.assertAlmostEqual(ag.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(ag.spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)
        self.assertAlmostEqual(ag.spearman([1, 2, 3, 4], [1, 1, 2, 2]), 0.8944, places=4)  # tied ranks
        self.assertIsNone(ag.spearman([1, 2, 3], [5, 5, 5]))
        atts = ten_attempts()
        atts[2]["reply_sha256"] = atts[1]["reply_sha256"]
        e = Fixture(atts, {i: 5.0 + i for i in range(1, 9)}, released_mean=50.0).report["exchangeability"]
        self.assertAlmostEqual(e["spearman_wall_seconds_vs_index"], 1.0)
        self.assertAlmostEqual(e["spearman_mean_10000_vs_index"], 1.0)
        self.assertEqual(e["reply_sha256_collisions"]["count"], 1)
        self.assertEqual(e["reply_sha256_collisions"]["groups"], {"sha-2": [2, 3]})
        self.assertEqual(e["cached_input_tokens"]["n"], 10)
        self.assertEqual(e["cached_input_tokens"]["max"], 1000.0)


class SchemaTests(unittest.TestCase):
    def test_report_has_every_key(self):
        r = Fixture(ten_attempts(), {i: 5.0 + i for i in range(1, 9)}, released_mean=50.0).report
        for k in ("replied_rate", "block_rate", "evaluable_rate", "clean_rate"):
            self.assertEqual(set(r[k]), {"count", "denominator", "rate", "low", "high"})
        for k in ("not_submitted_count", "transport_count", "networks", "thresholds", "distribution",
                  "released", "variance_components", "exchangeability"):
            self.assertIn(k, r)
        net_keys = {"index", "mean_10000", "se_10000", "mean_length", "censored_share_10000",
                    "censored_share_30000", "mean_30000", "exceeds_100m", "exceeds_500m",
                    "near_boundary_100m", "near_boundary_500m"}
        self.assertTrue(net_keys <= set(r["networks"][0]))
        self.assertEqual(set(r["thresholds"]), {"exceeds_100m", "exceeds_500m", "near_boundary_100m", "near_boundary_500m"})
        self.assertEqual(set(r["distribution"]), {"ecdf", "median", "iqr", "min", "max"})
        for k in ("percentile", "n_evaluable", "percentile_claim_allowed", "percentile_claim_alpha",
                  "percentile_attempt_level", "mean_10000", "se_10000"):
            self.assertIn(k, r["released"])
        self.assertEqual(set(r["variance_components"]),
                         {"n_networks", "n_episodes", "within_network_variance", "between_network_variance",
                          "ratio", "paired_mean_variance_across_networks"})
        self.assertEqual(set(r["exchangeability"]),
                         {"spearman_wall_seconds_vs_index", "spearman_mean_10000_vs_index",
                          "reply_sha256_collisions", "cached_input_tokens"})


if __name__ == "__main__":
    unittest.main()
