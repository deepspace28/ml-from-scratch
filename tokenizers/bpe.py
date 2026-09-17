"""Byte-pair encoding, built from scratch.

Same idea GPT-2 uses for its tokenizer, minus the regex pre-splitting rule
(and GPT-4's byte-level variant). Run this file directly to see it train,
encode, and decode round-trip. No dependencies.

    .venv/Scripts/python tokenizers/bpe.py
"""

import argparse
from collections import Counter


def get_pair_counts(ids):
    counts = Counter()
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts


def merge(ids, pair, new_id):
    merged = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


def train(text, vocab_size, verbose=False):
    """Byte-tokenize text, then greedily merge the most frequent pair
    (vocab_size - 256) times. Returns (merges, vocab).

    merges maps a pair -> new id, in training order. Decoding needs that
    exact order, so we store it instead of a plain dict lookup any time.
    """
    ids = list(text.encode("utf-8"))
    merges = {}
    vocab = {i: bytes([i]) for i in range(256)}

    for step in range(vocab_size - 256):
        counts = get_pair_counts(ids)
        if not counts:
            break
        pair = max(counts, key=counts.get)
        new_id = 256 + step
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        if verbose:
            print(f"step {step}: {pair} -> {new_id} (count {counts[pair]})")

    return merges, vocab


def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        counts = get_pair_counts(ids)
        present = [(merges[p], p) for p in counts if p in merges]
        if not present:
            break
        pair = min(present)[1]
        ids = merge(ids, pair, merges[pair])
    return ids


def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def compression_ratio(text, ids):
    return len(text.encode("utf-8")) / len(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="makemore/names.txt")
    parser.add_argument("--vocab-size", type=int, default=288)
    parser.add_argument("--prompt", default="the quick brown fox")
    args = parser.parse_args()

    with open(args.corpus, encoding="utf-8") as f:
        text = f.read()

    merges, vocab = train(text, args.vocab_size, verbose=True)
    print(f"\ntrained vocab of {len(vocab)} tokens "
          f"({len(merges)} merges) on {len(text)} chars")

    ids = encode(args.prompt, merges)
    decoded = decode(ids, vocab)
    print(f"prompt:    {args.prompt!r}")
    print(f"tokens:    {ids}")
    print(f"decoded:   {decoded!r}")
    assert decoded == args.prompt, "round-trip failed"
    print(f"compression ratio: {compression_ratio(args.prompt, ids):.2f} bytes -> 1 token")


if __name__ == "__main__":
    main()
