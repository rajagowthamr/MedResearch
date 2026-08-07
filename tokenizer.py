"""
Byte-level BPE tokenizer, from scratch.  The v4 step.

WHY REPLACE THE CHARACTER TOKENIZER
-----------------------------------
v2 spends its whole 256-slot context on 256 characters — roughly one long
sentence. And it burns capacity learning to spell: predicting the 'a' in
"pneumonia" is a task it must solve thousands of times.

BPE gives common sequences their own token, so:
  - "pneumonia" becomes ~2 tokens instead of 9
  - 256 tokens covers ~1000 characters of context, a whole paragraph
  - capacity shifts from spelling to word relationships

HOW IT WORKS
------------
1. Start with the 256 possible BYTE values as the base vocabulary.
2. Count every adjacent pair of tokens in the corpus.
3. Merge the most frequent pair into one new token.
4. Repeat until the vocabulary reaches the target size.

BYTE-level, not character-level, is deliberate. Starting from the 256 bytes
means EVERY possible input is representable, so there is no unknown token and
no need for the min_char_freq pruning v2 required. The 1103-character unicode
tail that forced v2's "�" fallback simply stops being a problem.

TWO IMPLEMENTATION DETAILS THAT MATTER
--------------------------------------
1. PRE-TOKENIZATION. We split on a GPT-2 style regex first, so merges can
   never span a word boundary. Without it you get junk tokens like "the_pat"
   spanning the gap between words, and " the"/"the"/"The" fragment
   inconsistently.
2. FREQUENCY-WEIGHTED UNIQUE CHUNKS. We never re-scan the 60MB corpus. We
   count unique word-chunks once, then merge over those ~200k unique chunks
   weighted by their counts. Same result, orders of magnitude faster.

    python tokenizer.py            # train on medical_text.txt -> tokenizer.json
"""
import collections
import json
import os
import re
import sys

# GPT-2's pre-tokenization pattern: keeps a leading space attached to a word
# (" patient" is one chunk), and splits letters / digits / punctuation apart.
PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+(?!\S)|\s+"""
)


class BPETokenizer:
    def __init__(self, merges=None, vocab=None):
        # merges maps a pair of token ids -> the new token id, in learned order
        self.merges = merges or {}
        # vocab maps token id -> the raw bytes it stands for
        self.vocab = vocab or {i: bytes([i]) for i in range(256)}

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    @classmethod
    def train(cls, text, vocab_size, verbose=True):
        assert vocab_size >= 256, "vocab must leave room for the 256 byte values"
        n_merges = vocab_size - 256

        # 1. pre-tokenize, then collapse to UNIQUE chunks with counts. This is
        #    the whole speed trick: 60MB becomes ~200k unique chunks.
        counts = collections.Counter(PAT.findall(text))
        # each unique chunk starts as its list of utf-8 byte values
        words = {chunk: list(chunk.encode("utf-8")) for chunk in counts}
        if verbose:
            print(f"corpus {len(text):,} chars -> {sum(counts.values()):,} chunks, "
                  f"{len(counts):,} unique")

        vocab = {i: bytes([i]) for i in range(256)}
        merges = {}

        # 2. Count all adjacent pairs ONCE, and remember which chunks contain
        #    each pair. Recounting the whole corpus per merge would be
        #    O(merges x corpus) — hours for 4096 merges. Updating only the
        #    chunks that actually changed makes it seconds.
        pairs = collections.Counter()
        pair_chunks = collections.defaultdict(set)
        for chunk, ids in words.items():
            f = counts[chunk]
            for p in zip(ids, ids[1:]):
                pairs[p] += f
                pair_chunks[p].add(chunk)

        for step in range(n_merges):
            if not pairs:
                break
            # 3. merge the most frequent pair into a brand new token id
            best = max(pairs, key=pairs.get)
            freq = pairs[best]
            new_id = 256 + step
            merges[best] = new_id
            vocab[new_id] = vocab[best[0]] + vocab[best[1]]

            # 4. rewrite only the chunks containing that pair, and repair the
            #    pair counts incrementally as we go
            for chunk in list(pair_chunks[best]):
                ids = words[chunk]
                f = counts[chunk]
                for p in zip(ids, ids[1:]):        # withdraw old contributions
                    pairs[p] -= f
                    if pairs[p] <= 0:
                        del pairs[p]
                new_ids = _merge(ids, best, new_id)
                words[chunk] = new_ids
                for p in zip(new_ids, new_ids[1:]):   # add new ones
                    pairs[p] += f
                    pair_chunks[p].add(chunk)
            pairs.pop(best, None)
            pair_chunks.pop(best, None)

            if verbose and (step < 8 or step % 500 == 0):
                tok = vocab[new_id].decode("utf-8", errors="replace")
                print(f"  merge {step:>5}: {freq:>9,}x  -> {tok!r}")

        return cls(merges, vocab)

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------
    def encode(self, text):
        out = []
        for chunk in PAT.findall(text):
            ids = list(chunk.encode("utf-8"))
            # apply the learned merges in the ORDER they were learned — a later
            # merge may depend on a token an earlier one created
            while len(ids) >= 2:
                # of the pairs present, pick the one learned earliest
                pair = min(
                    (p for p in zip(ids, ids[1:]) if p in self.merges),
                    key=lambda p: self.merges[p], default=None,
                )
                if pair is None:
                    break
                ids = _merge(ids, pair, self.merges[pair])
            out.extend(ids)
        return out

    def decode(self, ids):
        data = b"".join(self.vocab[i] for i in ids)
        # errors="replace": a truncated multi-byte character mid-generation is
        # normal and must not crash the caller
        return data.decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return len(self.vocab)

    def token_str(self, i):
        """Human-readable form, for inspecting what was learned."""
        return self.vocab[i].decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # persistence  -  JSON keys must be strings, so pairs are stringified
    # ------------------------------------------------------------------
    def save(self, path):
        with open(path, "w") as f:
            json.dump({
                "merges": {f"{a},{b}": c for (a, b), c in self.merges.items()},
                "vocab_size": self.vocab_size,
            }, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        merges = {tuple(int(x) for x in k.split(",")): v
                  for k, v in d["merges"].items()}
        # rebuild vocab by replaying the merges in learned order
        vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), c in sorted(merges.items(), key=lambda kv: kv[1]):
            vocab[c] = vocab[a] + vocab[b]
        return cls(merges, vocab)


def _merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


if __name__ == "__main__":
    vocab_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000_000

    with open("medical_text.txt", errors="ignore") as f:
        text = f.read(sample)

    tok = BPETokenizer.train(text, vocab_size)
    tok.save("tokenizer.json")

    # --- the numbers that justify the whole exercise ---
    probe = text[:200_000]
    ids = tok.encode(probe)
    ratio = len(probe) / len(ids)
    print(f"\nvocab_size        : {tok.vocab_size:,}")
    print(f"compression       : {ratio:.2f} chars per token")
    print(f"256 tokens covers : ~{256*ratio:.0f} characters "
          f"(v2's char tokenizer: 256)")
    assert tok.decode(ids) == probe, "round-trip failed"
    print("round-trip        : exact")

    print("\nlongest tokens learned:")
    for i in sorted(tok.vocab, key=lambda i: -len(tok.vocab[i]))[:12]:
        print(f"  {tok.token_str(i)!r}")

    print("\nhow it splits a clinical sentence:")
    s = "The patient presented with pneumonia and hypertension."
    print("  " + " | ".join(tok.token_str(i) for i in tok.encode(s)))
    print(f"  {len(s)} chars -> {len(tok.encode(s))} tokens")
