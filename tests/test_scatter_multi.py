import numpy as np
import torch

from scripts import gravity_sim as gs
from scripts import scatter_field as sf
from scripts import scatter_multi as sm

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_e(seed=4738):
    torch.manual_seed(seed)
    m = sf.ScatterField(grid=128, extent=64.0, net="unet", kernel=True, pp=2.0, split=True, potential=True, verlet=True, momfix=False).to(DEV)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return m.eval()


def scene(n=60, seed=4738):
    g = torch.Generator().manual_seed(seed)
    pos = ((torch.rand(1, n, 2, generator=g) - 0.5) * 40).to(DEV)
    vel = (torch.randn(1, n, 2, generator=g) * 0.5).to(DEV)
    one = torch.ones(1, n, device=DEV)
    return pos, vel, one, one.clone()


def test_warm_start_reproduces_e_force():
    e = make_e()
    m = sm.build().to(DEV).eval()
    m.init_from_e(e.state_dict())
    pos, vel, mass, mask = scene()
    cond = torch.randn(1, sm.DC, device=DEV)
    m.set_law(cond, None)
    with torch.no_grad():
        a_e, _ = e.force(pos, vel, mass, mask, e.init_field(1, DEV)[0])
        a_m, _ = m.force(pos, vel, mass, mask, m.init_field(1, DEV)[0])
    rel = (a_m - a_e).norm() / a_e.norm()
    assert rel < 2e-3, float(rel)


def test_typed_path_with_one_species_equals_untyped():
    m = sm.build().to(DEV).eval()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    pos, vel, mass, mask = scene()
    cond = torch.randn(1, sm.DC, device=DEV)
    f0 = m.init_field(1, DEV)[0]
    with torch.no_grad():
        m.set_law(cond, None)
        a0, _ = m.force(pos, vel, mass, mask, f0)
        m.set_law(cond, torch.zeros_like(mask))
        a1, _ = m.force(pos, vel, mass, mask, f0)
    assert torch.allclose(a0, a1, atol=1e-5, rtol=1e-4)


def test_species_matter_and_batch_conditioning():
    m = sm.build().to(DEV).eval()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    pos, vel, mass, mask = scene()
    typ = (torch.rand(1, 60, device=DEV) < 0.5).float()
    cond = torch.randn(1, sm.DC, device=DEV)
    f0 = m.init_field(1, DEV)[0]
    with torch.no_grad():
        m.set_law(cond, typ)
        a1, _ = m.force(pos, vel, mass, mask, f0)
        m.set_law(cond, 1 - typ)
        a2, _ = m.force(pos, vel, mass, mask, f0)
        pb, vb, mb, kb = [x.repeat(2, 1, 1) if x.dim() == 3 else x.repeat(2, 1) for x in (pos, vel, mass, mask)]
        m.set_law(cond.repeat(2, 1), typ.repeat(2, 1))
        ab, _ = m.force(pb, vb, mb, kb, m.init_field(2, DEV)[0])
    assert (a1 - a2).norm() > 1e-4
    assert torch.allclose(ab[0], a1[0], atol=1e-5, rtol=1e-4) and torch.allclose(ab[1], a1[0], atol=1e-5, rtol=1e-4)


def test_simulate_gravity_matches_gravity_sim():
    rng = np.random.default_rng(4738)
    n = 40
    p = rng.uniform(-8, 8, (n, 2))
    v = rng.normal(0, 0.5, (n, 2))
    Pg, Vg = gs.rollout_torch(p, v, 5, DEV)
    slots = sm.slots_for("single", "pow2", 1.0)
    par = torch.tensor(sm.params_of(slots), device=DEV)[None]
    P, V = sm.simulate(torch.tensor(p, device=DEV)[None], torch.tensor(v, device=DEV)[None], torch.zeros(1, n, device=DEV),
                       torch.ones(1, n, device=DEV, dtype=torch.float64), par, 5)
    assert np.abs(P[0].cpu().numpy() - Pg).max() < 1e-9


def test_charge_structure_signs_and_cond():
    slots = sm.slots_for("charge", "pow2", 2.0)
    par = torch.tensor(sm.params_of(slots), device=DEV)[None]
    pos = torch.tensor([[[0.0, 0.0], [3.0, 0.0]]], device=DEV, dtype=torch.float64)
    mk = torch.ones(1, 2, device=DEV, dtype=torch.float64)
    like = sm.exact_acc(pos, torch.tensor([[0.0, 0.0]], device=DEV, dtype=torch.float64), mk, par)[0, 0, 0]
    unlike = sm.exact_acc(pos, torch.tensor([[0.0, 1.0]], device=DEV, dtype=torch.float64), mk, par)[0, 0, 0]
    assert like < 0 < unlike
    assert len(sm.cond_of(slots)) == sm.DC


def test_gen_cases_small():
    cases = [("pow2", "single", None), ("yuk8", "charge", None), ("pow3", "predprey", None), ("lj", "single", None)]
    d = sm.gen_cases(cases, 4738, DEV, 5, chunk=4)
    assert d["P"].shape[1:] == (6, sm.N_PAD, 2) and d["P"].shape[0] >= 3
    assert torch.isfinite(d["P"]).all()
