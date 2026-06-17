import urllib.request
from pathlib import Path

import torch

DATA = Path(__file__).resolve().parent / "data" / "input.txt"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text():
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, DATA)
    return DATA.read_text(encoding="utf-8")


class CharData:
    def __init__(self, text, split=0.9):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(split * len(data))
        self.train, self.val = data[:n], data[n:]

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def batch(self, split, block_size, batch_size, device="cpu"):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)
