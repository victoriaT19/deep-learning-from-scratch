import math
import re
from collections import Counter

TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text):
    return [t.lower() for t in TOKEN.findall(text)]


def chunk(text, size=80, overlap=20):
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


class TfidfIndex:
    def __init__(self, docs):
        self.docs = docs
        tokenized = [tokenize(d) for d in docs]
        df = Counter(t for toks in tokenized for t in set(toks))
        n = len(docs)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self.vectors = [self._vector(toks) for toks in tokenized]

    def _vector(self, toks):
        tf = Counter(toks)
        v = {t: (c / len(toks)) * self.idf.get(t, 0.0) for t, c in tf.items()} if toks else {}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def search(self, query, k=3):
        q = self._vector(tokenize(query))
        scores = [sum(q.get(t, 0.0) * w for t, w in v.items()) for v in self.vectors]
        best = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.docs[i], scores[i]) for i in best]
