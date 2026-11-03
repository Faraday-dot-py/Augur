import torch

from scripts import tier0_corrector as tc
from scripts import train_tier0_corrector as trn
from scripts import train_tiered as ttr


class StubBase:
    """Frozen toy base: harmonic pull toward the origin with the wrong stiffness."""

    def __init__(self, k):
        self.k = k

    def init_state(self, B):
        return None

    def force(self, pos, vel, m, mk, st):
        return -self.k * pos * mk[..., None], st, None

    def edge(self, pos):
        return (32.0 - pos.abs().amax(-1)).clamp(0.0, 32.0)


def scene(B=3, N=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = (torch.rand(B, N, 2, generator=g) - 0.5) * 40
    vel = torch.randn(B, N, 2, generator=g) * 0.3
    mk = torch.ones(B, N)
    mk[:, -5:] = 0
    return pos * mk[..., None], vel * mk[..., None], torch.ones(B, N), mk


def test_identity_at_init():
    pos, vel, m, mk = scene()
    c = tc.ResidualCorrector(use_ext=True)
    out = c(pos, torch.randn_like(pos), mk, torch.rand(3, 40) * 16, 7, torch.randn_like(pos))
    assert torch.equal(out, torch.zeros_like(out))


def randomised(use_ext=False, momfix=True):
    c = tc.ResidualCorrector(use_ext=use_ext, momfix=momfix)
    torch.nn.init.normal_(c.mlp[-1].weight, std=0.5)
    return c


def test_momfix_and_mask():
    pos, vel, m, mk = scene()
    c = randomised()
    a = torch.randn_like(pos) * 0.1
    out = c(pos, a, mk, torch.rand(3, 40) * 16, 3)
    assert out.abs().max() > 0
    assert torch.allclose((out * mk[..., None]).sum(1), torch.zeros(3, 2), atol=1e-6)
    assert (out[:, -5:] == 0).all()
    assert c(pos, a, mk, torch.rand(3, 40), 3).shape == (3, 40, 2)
    c2 = randomised(momfix=False)
    assert c2(pos, a, mk, torch.rand(3, 40) * 16, 3).sum(1).abs().max() > 1e-4


def test_permutation_equivariant():
    pos, vel, m, mk = scene()
    c = randomised()
    a = torch.randn_like(pos) * 0.1
    e = torch.rand(3, 40) * 16
    perm = torch.randperm(40)
    o1 = c(pos, a, mk, e, 5)
    o2 = c(pos[:, perm], a[:, perm], mk[:, perm], e[:, perm], 5)
    assert torch.allclose(o1[:, perm], o2, atol=1e-6)


def test_translation_by_whole_cells():
    pos, vel, m, mk = scene()
    c = randomised()
    a = torch.randn_like(pos) * 0.1
    e = torch.rand(3, 40) * 16
    cell = c.extent / c.grid
    o1 = c(pos, a, mk, e, 5)
    o2 = c(pos + 4 * cell, a, mk, e, 5)
    assert torch.allclose(o1, o2, atol=1e-5)


def test_linear_cost_runs_at_large_n():
    c = randomised()
    N = 60000
    pos = (torch.rand(1, N, 2) - 0.5) * 120
    out = c(pos, torch.randn_like(pos) * 0.1, torch.ones(1, N), torch.rand(1, N) * 16, 1)
    assert out.shape == (1, N, 2) and torch.isfinite(out).all()


def test_oracle_variants():
    pos, vel, m, mk = scene(N=30)
    a_f = torch.randn_like(pos) * 0.1
    at = ttr.true_acc(pos, m, mk)
    assert torch.allclose(tc.Oracle("lam", 1.0)(a_f, pos, m, mk, 0), at, atol=1e-6)
    assert torch.equal(tc.Oracle("lam", 0.0)(a_f, pos, m, mk, 0), a_f)
    assert torch.allclose(tc.Oracle("lam", 0.5)(a_f, pos, m, mk, 0), 0.5 * (a_f + at), atol=1e-6)
    o = tc.Oracle("steps", 5)
    assert torch.allclose(o(a_f, pos, m, mk, 10), at, atol=1e-6) and torch.equal(o(a_f, pos, m, mk, 3), a_f)
    o = tc.Oracle("parts", 0.5)
    r = o(a_f, pos, m, mk, 1)
    sel = (r - a_f).abs().sum(-1) > 0
    assert 0.2 < sel.float().mean() < 0.8
    assert torch.equal(sel, (o(a_f, pos, m, mk, 2) - a_f).abs().sum(-1) > 0)
    o = tc.Oracle("dense", 0.1)
    r = o(a_f, pos, m, mk, 1)
    assert 1 <= int(((r - a_f).abs().sum(-1) > 0).sum(1).max()) <= 4


def test_closed_loop_learns_stiffness_error():
    torch.manual_seed(0)
    pos, vel, m, mk = scene(B=4, N=24)
    pos, vel = pos * 0.3, vel * 0.1
    truth_base = StubBase(0.25)
    P, V = [pos], [vel]
    acc_t = tc.make_acc(truth_base)
    a, _ = acc_t(pos, vel, 0, m, mk, None)
    p, v = pos, vel
    for t in range(1, 21):
        p, v, a, _ = tc.verlet(acc_t, p, v, a, None, t, m, mk)
        P.append(p), V.append(v)
    d = dict(P=torch.stack(P, 1).double(), V=torch.stack(V, 1).double(), M=m.double(), mask=mk.double())
    base = StubBase(0.05)
    c = tc.ResidualCorrector(width=16, momfix=False)
    opt = torch.optim.Adam(c.parameters(), lr=3e-3)
    idx = torch.arange(4)
    z = torch.zeros(4, dtype=torch.long)

    def run(back):
        return trn.closed_loop_loss(tc.make_acc(base, c), base, c, d, idx, z, 20, 5, 0.0, back)

    first = run(lambda l: None)
    for _ in range(150):
        opt.zero_grad()
        run(lambda l: l.backward() if l.requires_grad else None)
        opt.step()
    last = run(lambda l: None)
    assert last["pos"] < 0.3 * first["pos"]
