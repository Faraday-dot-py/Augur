import argparse

import numpy as np
import torch

from scripts import gravity_sim as gs
from scripts import scatter_field as sf
from scripts import train_scatter_cl as cl
from scripts import train_scatter_field as tsf


def tiny(verlet=False):
    torch.manual_seed(4738)
    return sf.ScatterField(grid=16, extent=16.0, net="local", width=8, layers=2, recurrent=True, momfix=True, verlet=verlet)


def scenes(n, steps, seed=4738):
    return gs.make_dataset(n, (3, 6), steps, seed, scale=False)


def test_exact_rollout_matches_gravity_sim():
    p, v = gs.init_bodies(5, np.random.default_rng(4738))
    Pg, Vg = gs.rollout(p, v, 10)
    P, V = cl.exact_rollout(torch.tensor(p - gs.CENTER)[None], torch.tensor(v)[None], torch.ones(1, 5, dtype=torch.float64), torch.ones(1, 5, dtype=torch.float64), 10)
    assert np.abs(P[0].numpy() - (Pg - gs.CENTER)).max() < 1e-8
    assert np.abs(V[0].numpy() - Vg).max() < 1e-8


def test_exact_rollout_masses_and_padding():
    rng = np.random.default_rng(4738)
    p, v = gs.init_bodies(4, rng)
    m = rng.uniform(0.1, 3.0, 4)
    Pr, Vr = sf.rollout_mass_torch(p, v, m, 8, "cpu")
    pad = lambda x, fill: torch.cat([torch.tensor(x), torch.full((2,) + x.shape[1:], fill, dtype=torch.float64)])[None]
    mk = torch.tensor([1.0] * 4 + [0.0] * 2, dtype=torch.float64)[None]
    P, V = cl.exact_rollout(pad(p - gs.CENTER, 0.0), pad(v, 0.0), pad(m, 7.0), mk, 8)
    assert np.abs(P[0, :, :4].numpy() - (Pr - gs.CENTER)).max() < 1e-8
    assert np.abs(V[0, :, :4].numpy() - Vr).max() < 1e-8


def test_relabel_from_truth_state_reproduces_truth():
    data = scenes(3, 30)
    P, V, M, mask = sf.to_tensors(data, "cpu")
    s, k = 5, 12
    Pt, Vt = cl.relabel(P[:, s], V[:, s], M, mask, k)
    assert (Pt.double() - P[:, s:s + k + 1]).abs().max() < 1e-4
    Pd, Vd = cl.exact_rollout(P[:, s], V[:, s], M, mask, k)
    assert (Pd - P[:, s:s + k + 1]).abs().max() < 1e-8
    assert (Vd - V[:, s:s + k + 1]).abs().max() < 1e-8


def test_relabel_batch_state_is_model_state_and_detached():
    model = tiny(verlet=True)
    P, V, M, mask = [t.float() for t in sf.to_tensors(scenes(4, 5), "cpu")]
    args = argparse.Namespace(noise=0.0, reset_field=False, dt=0.1, eps=0.5)
    pos, vel, state, Pt, Vt = cl.relabel_batch(model, P[:, 0], V[:, 0], M, mask, 3, 6, args)
    pr, vr, m, st = P[:, 0], V[:, 0], M, model.init_field(4, "cpu")
    with torch.no_grad():
        for _ in range(3):
            pr, vr, m, st, _ = sf._step_m(model, pr, vr, m, mask, st)
    assert torch.equal(pos, pr) and torch.equal(vel, vr)
    assert not pos.requires_grad and not state[0].requires_grad
    assert not torch.allclose(pos, P[:, 3])
    assert torch.allclose(Pt[:, 0], pos) and Pt.shape == (4, 7, P.shape[1], 2)
    args.reset_field = True
    assert cl.relabel_batch(model, P[:, 0], V[:, 0], M, mask, 3, 6, args)[2][0].abs().max() == 0


def test_perturb_zero_mean_and_masked():
    mk = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    pos, vel = torch.zeros(1, 4, 2), torch.zeros(1, 4, 2)
    p, v = cl.perturb(pos, vel, mk, 1e-3, torch.Generator().manual_seed(4738))
    assert p[0, 3].abs().max() == 0 and p[0].sum(0).abs().max() < 1e-9 and p.abs().max() > 0


def test_stage0_equals_current_trainer(tmp_path):
    data = scenes(8, 30)
    tensors = sf.to_tensors(data, "cpu")
    common = dict(lr=1e-3, seed=4738, opt="none", batch=4, k_start=2, k_end=3, time_budget=0.0, iters=3, log_every=1000, device="cpu", steps=30,
                  loss_cap=0.0, long_windows=0)
    a, b = tiny(), tiny()
    tsf.train(a, tensors, argparse.Namespace(**common), str(tmp_path / "a.pt"), lambda s: None)
    cl.train(b, tensors, None, argparse.Namespace(mode="truth", **common, mix_truth=0.25, noise=0.0, reset_field=False, eval_every=10 ** 9, dt=0.1, eps=0.5),
             str(tmp_path / "b.pt"), lambda s: None)
    for x, y in zip(a.parameters(), b.parameters()):
        assert torch.equal(x, y)


def test_relabel_training_step_runs_and_gate_advances(tmp_path):
    data = scenes(8, 30)
    tensors = sf.to_tensors(data, "cpu")
    evs = [sf.to_tensors(scenes(4, 30, seed=9000), "cpu")]
    args = argparse.Namespace(lr=1e-3, seed=4738, batch=4, k_start=2, k_end=3, iters=6, log_every=1000, device="cpu", steps=30, loss_cap=0.0, mode="relabel",
                              mix_truth=0.25, noise=1e-3, reset_field=False, eval_every=2, dt=0.1, eps=0.5, min_stage_its=0, max_stage_its=2, plateau=0.02,
                              patience=1)
    model = tiny(verlet=True)
    _, hist, decisions = cl.train(model, tensors, evs, args, str(tmp_path / "c.pt"), lambda s: None)
    assert decisions and decisions[0]["from"] == 0 and decisions[0]["to"] == 5
    assert all(torch.isfinite(p).all() for p in model.parameters())
