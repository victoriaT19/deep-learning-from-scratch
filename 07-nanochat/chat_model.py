from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 261
    max_len: int = 128
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) for t in (q, k, v))
        if cache is not None:
            if cache.get("k") is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        causal = cache is None or q.size(2) > 1
        if causal and k.size(2) != q.size(2):
            past = k.size(2) - q.size(2)
            mask = torch.ones(q.size(2), k.size(2), dtype=torch.bool, device=x.device).tril(past)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.attn = nn.LayerNorm(cfg.dim), Attention(cfg)
        self.n2 = nn.LayerNorm(cfg.dim)
        self.mlp = nn.Sequential(nn.Linear(cfg.dim, 4 * cfg.dim), nn.GELU(), nn.Linear(4 * cfg.dim, cfg.dim))

    def forward(self, x, cache=None):
        x = x + self.attn(self.n1(x), cache)
        return x + self.mlp(self.n2(x))


class ChatModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos = nn.Embedding(cfg.max_len, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None, mask=None, caches=None, start=0):
        T = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(start, start + T, device=idx.device))
        for i, block in enumerate(self.blocks):
            x = block(x, None if caches is None else caches[i])
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
        if mask is not None:
            loss = (loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()
        return logits, loss
