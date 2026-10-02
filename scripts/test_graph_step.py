"""Check CUDA-graphed ScatterField.step matches eager (loss, grads) and time fwd+bwd for k=20."""
import time

import torch
import torch.nn as nn

from scripts import scatter_field as sf
from scripts.train_scatter_field import EXPS, build


class StepWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pos, vel, mass, mask, field):
        p, v, f, dv = self.m.step(pos, vel, mass, mask, field)
        return p, v, f


def run(step, m, pos, vel, mass, mask, field, k):
    m.zero_grad()
    p, v, f = pos, vel, field
    loss = 0
    for i in range(k):
        p, v, f = (step[i] if isinstance(step, (list, tuple)) else step)(p, v, mass, mask, f)
        loss = loss + (p ** 2).mean() + (v ** 2).mean()
    loss.backward()
    return loss.item()


def main():
    dev = torch.device("cuda")
    torch.manual_seed(0)
    m = build(EXPS["A"], "ms_kp_pot", 0.1).to(dev)
    with torch.no_grad():
        for p_ in m.parameters():
            p_.add_(0.002 * torch.randn_like(p_))
    B, N, k = 32, 2, 20
    pos = ((torch.rand(B, N, 2, device=dev) - 0.5) * 10).requires_grad_(True)
    vel = (torch.randn(B, N, 2, device=dev) * 0.5).requires_grad_(True)
    mass = torch.ones(B, N, device=dev)
    mask = torch.ones(B, N, device=dev)
    field = m.init_field(B, dev).requires_grad_(True)
    wrap = StepWrap(m)
    eager = lambda p, v, ms, mk, f: wrap(p, v, ms, mk, f)
    l0 = run(eager, m, pos, vel, mass, mask, field, k)
    g0 = [p_.grad.clone() for p_ in m.parameters() if p_.grad is not None]
    sample = lambda: (pos.detach().clone().requires_grad_(True), vel.detach().clone().requires_grad_(True), mass, mask, field.detach().clone().requires_grad_(True))
    graphed = torch.cuda.make_graphed_callables(tuple(StepWrap(m) for _ in range(k)), tuple(sample() for _ in range(k)))
    l1 = run(graphed, m, pos, vel, mass, mask, field, k)
    g1 = [p_.grad.clone() for p_ in m.parameters() if p_.grad is not None]
    print("loss eager/graph", l0, l1)
    print("max grad rel diff", max(((a - b).abs().max() / (a.abs().max() + 1e-12)).item() for a, b in zip(g0, g1)), len(g0), len(g1))

    def t(fn, n=10):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        s = time.time()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.time() - s) / n * 1000

    print("eager ms", t(lambda: run(eager, m, pos, vel, mass, mask, field, k)))
    print("graph ms", t(lambda: run(graphed, m, pos, vel, mass, mask, field, k)))


if __name__ == "__main__":
    main()
