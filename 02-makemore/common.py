import random
import sys
import urllib.request
from pathlib import Path

import torch

DATA = Path(__file__).resolve().parent / "data" / "names.txt"
URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"


def load_words():
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, DATA)
    return DATA.read_text().split()


def vocab(words):
    chars = sorted(set("".join(words)))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    stoi["."] = 0
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def build_dataset(words, stoi, block_size):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


def splits(block_size, seed=42):
    words = load_words()
    random.Random(seed).shuffle(words)
    stoi, itos = vocab(words)
    n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
    return (
        build_dataset(words[:n1], stoi, block_size),
        build_dataset(words[n1:n2], stoi, block_size),
        build_dataset(words[n2:], stoi, block_size),
        stoi,
        itos,
    )


def steps_arg(default):
    return int(sys.argv[1]) if len(sys.argv) > 1 else default
