import argparse
import copy

import numpy as np
import torch

from scripts import eval_scatter_solar_stability as ev
from scripts import train_scatter_solar_stability as tr


class Spring(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k = torch.nn.Parameter(torch.tensor(0.5))
        self.dt = 0.1

    def init_field(self, B, device, dtype=torch.float32):
        return None

    def step_m(self, pos, vel, mass, mask, field):
        vel = vel - self.k * pos * self.dt
        return pos + vel * self.dt, vel, mass, field, None


def tiny_args(iters, **kw):
    a = dict(batch=2, iters=iters, lr=1e-2, warmup=2, k_start=2, k_end=5, spike_factor=0.0, ckpt_every=5, log_every=5, seed=4738, fresh=False)
    a.update(kw)
    return argparse.Namespace(**a)


def tiny_data():
    g = torch.Generator().manual_seed(0)
    P = torch.randn(3, 40, 4, 2, generator=g)
    return P, torch.randn(3, 40, 4, 2, generator=g), torch.ones(3, 4)


def test_schedule():
    assert tr.k_at(0, 100, 10, 100) == 10 and tr.k_at(99, 100, 10, 100) == 100
    assert tr.lr_at(0, 100, 1.0, 10) < tr.lr_at(9, 100, 1.0, 10)
    assert tr.lr_at(99, 100, 1.0, 1) < 0.06


def test_resume_matches_uninterrupted(tmp_path):
    data = tiny_data()
    ref = Spring()
    tr.train(ref, data, tiny_args(10), str(tmp_path / "a.pt"), lambda s: None)
    m = Spring()
    tr.STOP[0] = False
    tr.train(m, data, tiny_args(10), str(tmp_path / "b.pt"), lambda s: tr.STOP.__setitem__(0, True) if "it 5/" in s else None)
    tr.STOP[0] = False
    assert torch.load(str(tmp_path / "b.pt"), weights_only=False)["it"] == 5
    m2 = Spring()
    it, _, _ = tr.train(m2, data, tiny_args(10), str(tmp_path / "b.pt"), lambda s: None)
    assert it == 10
    assert torch.allclose(ref.k, m2.k)


def test_training_scenes_disjoint_from_eval_seeds():
    assert tr.TRAIN_SEED_BASE > 12000
    P, V, M = tr.make_scenes(1, 3, torch.device("cpu"), log=lambda s: None)
    assert P.shape == (1, 4, 210, 2) and M.shape == (1, 210)


def test_assimilate_moves_fraction():
    p, v = torch.zeros(1, 3, 2), torch.zeros(1, 3, 2)
    p2, v2 = ev.assimilate(p, v, torch.ones(1, 3, 2), 2 * torch.ones(1, 3, 2), 0.1)
    assert torch.allclose(p2, torch.full((1, 3, 2), 0.1)) and torch.allclose(v2, torch.full((1, 3, 2), 0.2))


def test_rollout_assimilated_only_at_multiples():
    scene = dict(pos=np.ones((3, 2)), vel=np.zeros((3, 2)), mass=np.ones(3))
    Pt = np.zeros((11, 3, 2))
    Vt = np.zeros((11, 3, 2))
    plain, _ = ev.rollout_assimilated(Spring(), scene, Pt, Vt, 10, 0, 0.1, torch.device("cpu"))
    corr, _ = ev.rollout_assimilated(Spring(), scene, Pt, Vt, 10, 5, 0.1, torch.device("cpu"))
    assert np.allclose(plain[:5], corr[:5])
    assert np.abs(corr[5]).max() < np.abs(plain[5]).max()


def test_verdict_rules():
    def tab(arm, base):
        return {"1": {"b": {"survivors": {"2000": base}}, "a": {"survivors": {"2000": arm}}}}
    assert ev.verdict(tab(7, 3), "a", "b", [1])[0] == "PASS"
    assert ev.verdict(tab(5, 3), "a", "b", [1])[0] == "INCONCLUSIVE"
    assert ev.verdict(tab(3, 2), "a", "b", [1])[0] == "KILL"
    assert ev.verdict(tab(6, 6), "a", "b", [1])[0] == "KILL"
