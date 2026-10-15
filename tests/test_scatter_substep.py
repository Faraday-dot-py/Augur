import importlib.util
import os

try:
    import pytest
except ImportError:
    pytest = None
import torch

from scripts import scatter_field as sf


def _model(seed=4738, zero_pair=False):
    torch.manual_seed(seed)
    m = sf.ScatterField(grid=32, extent=16.0, net="unet", recurrent=True, kernel=True, pp=2.0, split=True, potential=True, verlet=True)
    for p in list(m.net.out.parameters()) + list(m.kmlp[-1].parameters()) + list(m.ppmlp[-1].parameters()):
        torch.nn.init.normal_(p, std=0.05)
    if zero_pair:
        for p in m.ppmlp[-1].parameters():
            torch.nn.init.zeros_(p)
    return m.eval()


def _scene(B=3, N=24, seed=4738):
    g = torch.Generator().manual_seed(seed)
    pos = (torch.rand(B, N, 2, generator=g) - 0.5) * 6
    vel = torch.randn(B, N, 2, generator=g) * 0.5
    mass = torch.rand(B, N, generator=g) + 0.5
    mask = torch.ones(B, N)
    mask[0, -3:] = 0
    return pos, vel, mass, mask


def test_force_parts_default_bit_identical():
    m = _model()
    pos, vel, mass, mask = _scene()
    f0 = m.init_field(3, "cpu")[0]
    with torch.no_grad():
        a, fld = m._force(pos, vel, mass, mask, f0)
        a2, fld2, app = m._force(pos, vel, mass, mask, f0, parts=True)
    assert torch.equal(a, a2) and torch.equal(fld, fld2)
    assert app.abs().max() > 0


def test_force_default_matches_master():
    ref = os.environ.get("REF_SF")
    if not ref or not os.path.exists(ref):
        if pytest is None:
            print("SKIP REF_SF not given")
            return
        pytest.skip("REF_SF (master scripts/scatter_field.py) not given")
    spec = importlib.util.spec_from_file_location("sf_master", ref)
    sfm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sfm)
    m = _model()
    torch.manual_seed(4738)
    mm = sfm.ScatterField(grid=32, extent=16.0, net="unet", recurrent=True, kernel=True, pp=2.0, split=True, potential=True, verlet=True)
    mm.load_state_dict(m.state_dict())
    mm.eval()
    pos, vel, mass, mask = _scene()
    with torch.no_grad():
        a, fld = m._force(pos, vel, mass, mask, m.init_field(3, "cpu")[0])
        b, fldb = mm._force(pos, vel, mass, mask, mm.init_field(3, "cpu")[0])
        st = m.init_field(3, "cpu")
        stm = mm.init_field(3, "cpu")
        p1, v1, _, _ = m.step(pos, vel, mass, mask, st)
        p2, v2, _, _ = mm.step(pos, vel, mass, mask, stm)
    assert torch.equal(a, b) and torch.equal(fld, fldb)
    assert torch.equal(p1, p2) and torch.equal(v1, v2)


def test_sym_pair_momentum_exact():
    m = _model()
    pos, vel, mass, mask = _scene(N=40)
    pos = pos * 0.3
    with torch.no_grad():
        f = m.pp_acc_sym(pos, mass, mask, 16)
    p = ((mass * mask)[..., None] * f).sum(1)
    assert f.abs().max() > 0
    assert p.abs().max() < 1e-4 * ((mass * mask)[..., None] * f).abs().sum(1).max()


def test_sym_pair_equals_knn_term_when_not_truncated():
    m = _model()
    pos, vel, mass, mask = _scene(N=12)
    with torch.no_grad():
        a = m.pp_acc(pos, mass, mask, 16)
        b = m.pp_acc_sym(pos, mass, mask, 16)
    assert torch.allclose(a, b, atol=1e-5)


def test_respa_zero_pair_equals_verlet1():
    m = _model(zero_pair=True)
    pos, vel, mass, mask = _scene()
    with torch.no_grad():
        st = m.init_field(3, "cpu")
        p1, v1, (f1, a1), _ = m.step(pos, vel, mass, mask, st)
        m.respa = 4
        p2, v2, (f2, a2), _ = m.step(pos, vel, mass, mask, m.init_field(3, "cpu"))
    assert torch.allclose(p1, p2, atol=1e-5) and torch.allclose(v1, v2, atol=1e-5) and torch.allclose(a1, a2, atol=1e-4)


def test_apply_opt_nf():
    m = _model()
    sf.apply_opt(m, "nf4@2")
    assert m.respa == 4 and m.respa_rc == 2.0


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            try:
                fn()
                print("PASS", k, flush=True)
            except BaseException as e:
                if pytest is None or not isinstance(e, pytest.skip.Exception):
                    raise
                print("SKIP", k, e, flush=True)
