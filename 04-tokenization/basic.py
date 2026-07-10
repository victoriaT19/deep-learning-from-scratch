from base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        merges, vocab = {}, {idx: bytes([idx]) for idx in range(256)}
        for i in range(vocab_size - 256):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i + 1}: {pair} -> {idx} ({vocab[idx]!r}) seen {stats[pair]} times")
        self.merges, self.vocab = merges, vocab

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids
