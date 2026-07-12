import regex as re

from base import Tokenizer, get_stats, merge

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):
    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = pattern or GPT4_SPLIT_PATTERN
        self.compiled = re.compile(self.pattern)
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        chunks = [list(ch.encode("utf-8")) for ch in re.findall(self.compiled, text)]
        merges, vocab = {}, {idx: bytes([idx]) for idx in range(256)}
        for i in range(vocab_size - 256):
            stats = {}
            for chunk in chunks:
                get_stats(chunk, stats)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            chunks = [merge(chunk, pair, idx) for chunk in chunks]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i + 1}: {pair} -> {idx} ({vocab[idx]!r})")
        self.merges, self.vocab = merges, vocab

    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        parts = []
        for idx in ids:
            if idx in self.vocab:
                parts.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                parts.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id {idx}")
        return b"".join(parts).decode("utf-8", errors="replace")

    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode_ordinary(self, text):
        ids = []
        for chunk in re.findall(self.compiled, text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode(self, text, allowed_special="none_raise"):
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        else:
            special = {}
            assert all(token not in text for token in self.special_tokens)
        if not special:
            return self.encode_ordinary(text)
        pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        ids = []
        for part in re.split(pattern, text):
            ids.extend([special[part]] if part in special else self.encode_ordinary(part))
        return ids
