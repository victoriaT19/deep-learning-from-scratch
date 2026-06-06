import torch
import torch.nn.functional as F

from common import splits, steps_arg
from layers import BatchNorm1d, Embedding, Linear, Sequential, Tanh

block_size, n_embd, n_hidden = 3, 10, 100
(Xtr, Ytr), (Xdev, Ydev), _, stoi, itos = splits(block_size)
V = len(stoi)
g = torch.Generator().manual_seed(2147483647)


class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []


def build(batchnorm):
    layers = [Embedding(V, n_embd, g), Flatten()]
    dims = [n_embd * block_size] + [n_hidden] * 5
    for a, b in zip(dims, dims[1:]):
        layers += [Linear(a, b, bias=not batchnorm, generator=g)] + ([BatchNorm1d(b)] if batchnorm else []) + [Tanh()]
    layers += [Linear(n_hidden, V, generator=g)]
    with torch.no_grad():
        layers[-1].weight *= 0.1
        for layer in layers:
            if isinstance(layer, Linear) and layer is not layers[-1]:
                layer.weight *= 5 / 3
    model = Sequential(layers)
    for p in model.parameters():
        p.requires_grad = True
    return model


def train(model, steps):
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
        loss = F.cross_entropy(model(Xtr[ix]), Ytr[ix])
        for p in model.parameters():
            p.grad = None
        loss.backward()
        lr = 0.1 if i < steps * 3 // 4 else 0.01
        for p in model.parameters():
            p.data += -lr * p.grad
    return loss.item()


def saturation(model):
    stats = []
    for layer in model.layers:
        if isinstance(layer, Tanh):
            t = layer.out
            stats.append(f"{(t.abs() > 0.97).float().mean().item():.1%}")
    return stats


@torch.no_grad()
def dev_loss(model):
    model.train(False)
    out = F.cross_entropy(model(Xdev), Ydev).item()
    model.train(True)
    return out


steps = steps_arg(50000)
for batchnorm in (False, True):
    model = build(batchnorm)
    train(model, steps)
    with torch.no_grad():
        model(Xtr[:1000])
    print(f"batchnorm={batchnorm}: dev loss {dev_loss(model):.4f}, saturated tanh per layer {saturation(model)}")
