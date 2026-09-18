"""Round-trip and correctness checks for the byte-pair encoder."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tokenizers"))

from bpe import compression_ratio, decode, encode, get_pair_counts, merge, train


class TestPrimitives(unittest.TestCase):
    def test_pair_counts(self):
        counts = get_pair_counts([1, 2, 1, 2, 1])
        self.assertEqual(counts[(1, 2)], 2)
        self.assertEqual(counts[(2, 1)], 2)

    def test_merge_non_overlapping(self):
        # overlapping pairs collapse left-to-right, not greedily chained
        self.assertEqual(merge([1, 1, 1], (1, 1), 9), [9, 1])
        self.assertEqual(merge([1, 2, 1, 2], (1, 2), 9), [9, 9])

    def test_merge_no_match_is_identity(self):
        self.assertEqual(merge([1, 2, 3], (7, 8), 9), [1, 2, 3])


class TestTrainEncodeDecode(unittest.TestCase):
    def test_vocab_structure(self):
        merges, vocab = train("aaabdaaabac", 258)
        self.assertEqual(len(vocab), 258)  # 256 bytes + 2 merges
        self.assertEqual(vocab[0], b"\x00")
        # merges are numbered consecutively from 256
        self.assertEqual(sorted(merges.values()), [256, 257])

    def test_roundtrip_on_real_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 3
        merges, vocab = train(text, 280)
        ids = encode(text, merges)
        self.assertEqual(decode(ids, vocab), text)

    def test_roundtrip_multibyte_utf8(self):
        text = "héllo wörld — naïve café ☕" * 5
        merges, vocab = train(text, 300)
        ids = encode(text, merges)
        self.assertEqual(decode(ids, vocab), text)

    def test_encode_with_foreign_merges(self):
        # encoding text unseen during training still round-trips
        merges, vocab = train("aaaa bbbb aaaa bbbb", 270)
        unseen = "cccc dddd cccc"
        self.assertEqual(decode(encode(unseen, merges), vocab), unseen)

    def test_training_compresses_or_trivial(self):
        text = "banana banana banana banana"
        merges, _ = train(text, 280)
        ids = encode(text, merges)
        # merged text can't be worse than byte-level, and real merges help here
        self.assertLessEqual(len(ids), len(text.encode("utf-8")))
        self.assertLess(len(ids), len(text.encode("utf-8")))

    def test_vocab_size_256_is_noop(self):
        merges, vocab = train("hello", 256)
        self.assertEqual(merges, {})
        self.assertEqual(len(vocab), 256)


class TestCompression(unittest.TestCase):
    def test_ratio_positive(self):
        text = "hello world"
        ids = list(text.encode("utf-8"))
        self.assertEqual(compression_ratio(text, ids), 1.0)


if __name__ == "__main__":
    unittest.main()
