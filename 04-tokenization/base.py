import unicodedata


def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def render_token(t: bytes) -> str:
    s = t.decode("utf-8", errors="replace")
    return "".join(f"\\u{ord(ch):04x}" if unicodedata.category(ch)[0] == "C" else ch for ch in s)


class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, prefix):
        with open(prefix + ".model", "w", encoding="utf-8") as f:
            f.write("minbpe v1\n")
            f.write(f"{getattr(self, 'pattern', '')}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for p0, p1 in self.merges:
                f.write(f"{p0} {p1}\n")

    def load(self, model_file):
        with open(model_file, encoding="utf-8") as f:
            assert f.readline().strip() == "minbpe v1"
            self.pattern = f.readline().rstrip("\n")
            for _ in range(int(f.readline())):
                special, idx = f.readline().strip().split()
                self.special_tokens[special] = int(idx)
            idx = 256
            for line in f:
                p0, p1 = map(int, line.split())
                self.merges[(p0, p1)] = idx
                idx += 1
        self.vocab = self._build_vocab()
