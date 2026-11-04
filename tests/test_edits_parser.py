import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edits_parser import NPZ_KEYS, ParseResult, parse_reply, to_npz  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
ORACLE = {  # game: (statements, applied, non-zero total) from the authors' apply.json
    "breakout": (92, 92, 815),
    "seaquest": (122, 122, 803),
    "asterix": (55, 55, 637),
}


def block(body: str) -> str:
    return "prose\n\n```edits\n" + body + "\n```\n"


class OracleTests(unittest.TestCase):
    def check(self, game):
        with open(os.path.join(FIXTURES, f"{game}_reply.md")) as f:
            r = parse_reply(f.read())
        self.assertIsInstance(r, ParseResult)
        self.assertTrue(r.block_found)
        self.assertFalse(r.cut_off)
        self.assertEqual(r.errors, [])
        statements, applied, nonzero = ORACLE[game]
        self.assertEqual((r.statements, r.applied), (statements, applied))
        self.assertEqual(sum(int((t != 0).sum()) for t in r.tensors.values()), nonzero)
        hand = np.load(os.path.join(FIXTURES, f"{game}_hand.npz"))
        for name, key in NPZ_KEYS.items():
            self.assertEqual(r.tensors[name].dtype, np.float32)
            self.assertTrue(np.array_equal(r.tensors[name], hand[key]), key)

    def test_breakout(self):
        self.check("breakout")

    def test_seaquest(self):
        self.check("seaquest")

    def test_asterix(self):
        self.check("asterix")

    def test_to_npz_round_trip(self):
        with open(os.path.join(FIXTURES, "breakout_reply.md")) as f:
            r = parse_reply(f.read())
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.npz")
            to_npz(r, path)
            saved = np.load(path)
            self.assertEqual(set(saved.files), set(NPZ_KEYS.values()))
            for name, key in NPZ_KEYS.items():
                self.assertTrue(np.array_equal(saved[key], r.tensors[name]))


class SyntheticTests(unittest.TestCase):
    def parse(self, body):
        r = parse_reply(block(body))
        self.assertTrue(r.block_found)
        return r

    def test_shapes_and_zero_start(self):
        r = parse_reply(block("# nothing"))
        self.assertEqual({k: v.shape for k, v in r.tensors.items()},
                         {"W1": (600, 32), "b1": (32,), "W2": (32, 32), "b2": (32,),
                          "W3": (32, 6), "b3": (6,)})
        self.assertTrue(all(not v.any() for v in r.tensors.values()))
        self.assertEqual((r.statements, r.applied), (0, 0))

    def test_integer_index(self):
        r = self.parse("b1[3] = 2\nW2[1, 2] = -1.5")
        self.assertEqual(r.tensors["b1"][3], 2)
        self.assertEqual(r.tensors["W2"][1, 2], np.float32(-1.5))
        self.assertEqual(int((r.tensors["b1"] != 0).sum()), 1)
        self.assertEqual(int((r.tensors["W2"] != 0).sum()), 1)

    def test_half_open_range(self):
        r = self.parse("b2[2:5] = 1")
        self.assertEqual(r.tensors["b2"].tolist(), [0, 0, 1, 1, 1] + [0] * 27)

    def test_stride(self):
        r = self.parse("b3[0:6:2] = 1")
        self.assertEqual(r.tensors["b3"].tolist(), [1, 0, 1, 0, 1, 0])

    def test_whole_axis(self):
        r = self.parse("b1[:] = -8\nW3[:, 1] = 2")
        self.assertTrue((r.tensors["b1"] == -8).all())
        self.assertTrue((r.tensors["W3"][:, 1] == 2).all())
        self.assertEqual(int((r.tensors["W3"] != 0).sum()), 32)

    def test_scalar_fill_2d_block(self):
        r = self.parse("W2[0:2, 0:3] = 7")
        self.assertTrue((r.tensors["W2"][0:2, 0:3] == 7).all())
        self.assertEqual(int((r.tensors["W2"] != 0).sum()), 6)

    def test_1d_list(self):
        r = self.parse("b3[1:4] = [1, 2.5, -3]")
        self.assertEqual(r.tensors["b3"].tolist(), [0, 1, 2.5, -3, 0, 0])
        r = self.parse("W3[0:2, 0:2] = [1, 2, 3, 4]")  # row-major over the block
        self.assertEqual(r.tensors["W3"][0:2, 0:2].tolist(), [[1, 2], [3, 4]])

    def test_1d_list_wrong_length_skipped(self):
        r = self.parse("b3[1:4] = [1, 2]")
        self.assertEqual((r.statements, r.applied), (1, 0))
        self.assertFalse(r.tensors["b3"].any())

    def test_2d_list(self):
        r = self.parse("W2[0:2, 0:2] = [[1, 2], [3, 4]]")
        self.assertEqual(r.tensors["W2"][0:2, 0:2].tolist(), [[1, 2], [3, 4]])
        self.assertEqual(r.applied, 1)

    def test_2d_list_space_after_opening_bracket(self):
        r = self.parse("W2[0:2, 0:2] = [ [1, 2], [3, 4] ]")
        self.assertEqual(r.tensors["W2"][0:2, 0:2].tolist(), [[1, 2], [3, 4]])
        self.assertEqual(r.applied, 1)

    def test_2d_list_shape_mismatch_skipped(self):
        r = self.parse("W2[0:2, 0:3] = [[1, 2], [3, 4]]")
        self.assertEqual((r.statements, r.applied), (1, 0))

    def test_compound_operators(self):
        r = self.parse("b3[:] = 2\nb3[0] *= 3\nb3[1] += 0.5\nb3[2] -= 5")
        self.assertEqual(r.tensors["b3"].tolist(), [6, 2.5, -3, 2, 2, 2])
        self.assertEqual((r.statements, r.applied), (4, 4))
        r = self.parse("W2[0:2, 0:2] = 1\nW2[0:2, 0:2] *= 4")
        self.assertEqual(r.tensors["W2"][0:2, 0:2].tolist(), [[4, 4], [4, 4]])

    def test_later_overwrites_earlier(self):
        r = self.parse("b1[0:4] = 1\nb1[2] = 9")
        self.assertEqual(r.tensors["b1"][:4].tolist(), [1, 1, 9, 1])

    def test_bad_statement_skipped_and_recorded(self):
        r = self.parse("b1[0] = 1\nW9[0, 0] = 5\nb1[1] = 2\nb1[zz] = 3\nb1[2] = 4")
        self.assertEqual((r.statements, r.applied), (5, 3))
        self.assertEqual(r.tensors["b1"][:3].tolist(), [1, 2, 4])
        self.assertEqual([(n, s) for n, s, _ in r.errors],
                         [(2, "W9[0, 0] = 5"), (4, "b1[zz] = 3")])
        self.assertTrue(all(reason for _, _, reason in r.errors))

    def test_value_bound(self):
        r = self.parse("b1[0] = 1000000\nb1[1] = 1000001\nb1[2:4] = [1, -2e6]\nb1[4] = 1")
        self.assertEqual((r.statements, r.applied), (4, 2))
        self.assertEqual(r.tensors["b1"][:5].tolist(), [1e6, 0, 0, 0, 1])
        self.assertEqual([n for n, _, _ in r.errors], [2, 3])

    def test_multi_line_list(self):
        r = self.parse("W2[0:2, 0:3] = [\n  [1, 2, 3],   # first row\n\n  [4, 5, 6]\n]\nb1[0] = 1")
        self.assertEqual(r.tensors["W2"][0:2, 0:3].tolist(), [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(r.tensors["b1"][0], 1)
        self.assertEqual((r.statements, r.applied), (2, 2))

    def test_malformed_line_does_not_swallow_later_statements(self):
        # An unbalanced bracket outside a list value is a one-line bad statement, not a
        # continuation: the valid statements after it still apply.
        r = self.parse("b1[bad = 1\nb1[0] = 1\nb1[1] = 2")
        self.assertEqual((r.statements, r.applied), (3, 2))
        self.assertEqual(r.tensors["b1"][:2].tolist(), [1, 2])
        self.assertEqual([(n, s) for n, s, _ in r.errors], [(1, "b1[bad = 1")])
        # A non-list value with a stray bracket is also a single-line statement.
        r = self.parse("b1[0] = 1 [\nb1[1] = 2")
        self.assertEqual((r.statements, r.applied), (2, 1))
        self.assertEqual(r.tensors["b1"][:2].tolist(), [0, 2])
        self.assertEqual([n for n, _, _ in r.errors], [1])

    def test_comments_and_blank_lines_not_counted(self):
        r = self.parse("# comment\n\nb1[0] = 1  # trailing\n\n# another")
        self.assertEqual((r.statements, r.applied), (1, 1))

    def test_no_block(self):
        r = parse_reply("Just prose.\n\n```python\nx = 1\n```\n")
        self.assertFalse(r.block_found)
        self.assertFalse(r.cut_off)
        self.assertEqual((r.statements, r.applied), (0, 0))
        self.assertTrue(all(not v.any() for v in r.tensors.values()))

    def test_unclosed_block(self):
        r = parse_reply("prose\n```edits\nb1[0] = 1\nb1[1] = 2\n")
        self.assertTrue(r.block_found)
        self.assertTrue(r.cut_off)
        self.assertEqual((r.statements, r.applied), (2, 2))
        self.assertEqual(r.tensors["b1"][:2].tolist(), [1, 2])

    def test_only_last_block_counts(self):
        text = block("b1[0] = 1\nb1[1] = 1") + "\nmore prose\n" + block("b1[2] = 5")
        r = parse_reply(text)
        self.assertFalse(r.cut_off)
        self.assertEqual((r.statements, r.applied), (1, 1))
        self.assertEqual(r.tensors["b1"][:3].tolist(), [0, 0, 5])


if __name__ == "__main__":
    unittest.main()
