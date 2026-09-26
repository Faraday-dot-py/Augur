import sys
import torch

sys.path.insert(0, ".")
from scripts import tiling_emulator as te

torch.manual_seed(4738)
N, T = 20000, 8
pos = torch.randn(N, 2, dtype=torch.float64) * torch.rand(N, 1, dtype=torch.float64) * 3
tid = te.partition(pos, T, "morton")
F = te.Forest(pos, tid, T, "C")
ts = F.tid_sorted
worst = 0.0
nshared = 0
for n in range(F.n_nodes):
    a, c = int(F.start[n]), int(F.count[n])
    if c < 2:
        continue
    p = F.pos[a:a + c]
    t = ts[a:a + c]
    if len(torch.unique(t)) < 2:
        continue
    nshared += 1
    parts = [p[t == s].sum(0) for s in torch.unique(t)]
    fwd = sum(parts[1:], parts[0]) / c
    rev = sum(parts[::-1][1:], parts[::-1][0]) / c
    ref = p.sum(0) / c
    worst = max(worst, float((fwd - ref).abs().max()), float((fwd - F.com[n]).abs().max()))
    assert torch.equal(fwd, fwd)
    worst_order = float((fwd - rev).abs().max())
print("shared nodes", nshared, "max |partial-sum com - global com|", worst, "last order diff", worst_order)
