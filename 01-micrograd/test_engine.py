import torch

from engine import Value


def test_against_torch():
    a, b = Value(-4.0), Value(2.0)
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()

    at, bt = torch.tensor(-4.0, dtype=torch.double, requires_grad=True), torch.tensor(2.0, dtype=torch.double, requires_grad=True)
    ct = at + bt
    dt = at * bt + bt**3
    ct = ct + ct + 1
    ct = ct + 1 + ct + (-at)
    dt = dt + dt * 2 + (bt + at).relu()
    dt = dt + 3 * dt + (bt - at).relu()
    et = ct - dt
    ft = et**2
    gt = ft / 2.0
    gt = gt + 10.0 / ft
    gt.backward()

    assert abs(g.data - gt.item()) < 1e-6
    assert abs(a.grad - at.grad.item()) < 1e-6
    assert abs(b.grad - bt.grad.item()) < 1e-6


def test_tanh_exp_log():
    x = Value(0.7)
    y = (x.tanh() * x.exp()).log()
    y.backward()
    xt = torch.tensor(0.7, dtype=torch.double, requires_grad=True)
    yt = (xt.tanh() * xt.exp()).log()
    yt.backward()
    assert abs(x.grad - xt.grad.item()) < 1e-9
