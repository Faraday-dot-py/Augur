import argparse

import torch

from scripts import train_scatter_solar_moon_arms as ma
from scripts import train_scatter_solar_stability as tr
from tests.test_scatter_solar_stability import Spring, tiny_args


def aux_data():
    g = torch.Generator().manual_seed(1)
    P = torch.randn(3, 40, 6, 2, generator=g)
    V = torch.randn(3, 40, 6, 2, generator=g)
    M = torch.tensor([[10.0, 0.1, 0.1, 0.001, 0.0001, 0.0001]]).repeat(3, 1)
    K = torch.tensor([[0, 1, 1, 2, 3, 3]]).repeat(3, 1)
    H = torch.tensor([[-1, -1, -1, 1, -1, -1]]).repeat(3, 1)
    isb = (K == 3).long()
    return P, V, M, K, H, isb.cumsum(1) - 1, isb.sum(1).float()


def test_staircase_ends_full_and_starts_without_belt():
    s = tr.staircase_schedule(6000, 4, 30, 12)
    assert s[0] == (0, 4, 0.0) and s[-1][1:] == (30, 1.0)
    ks = [x[1] for x in s]
    fs = [x[2] for x in s]
    assert ks == sorted(ks) and fs == sorted(fs) and max(fs) == 1.0


def run(tmp_path, **kw):
    m = Spring()
    tr.STOP[0] = False
    a = tiny_args(10, **kw)
    tr.train(m, aux_data(), a, str(tmp_path / "x.pt"), lambda s: None)
    return m.k.item()


def test_default_mode_matches_plain_data(tmp_path):
    plain = Spring()
    tr.STOP[0] = False
    tr.train(plain, aux_data()[:3], tiny_args(10), str(tmp_path / "p.pt"), lambda s: None)
    assert abs(plain.k.item() - run(tmp_path)) < 1e-7


def test_modes_run_and_differ(tmp_path):
    base = run(tmp_path)
    w = run(tmp_path, loss_mode="moon-weighted", moon_weight=20.0, fresh=True)
    r = run(tmp_path, loss_mode="relative-error", fresh=True)
    c = run(tmp_path, curriculum_stages=4, fresh=True)
    assert len({round(x, 6) for x in (base, w, r, c)}) == 4


def test_moon_weight_one_is_default(tmp_path):
    assert abs(run(tmp_path) - run(tmp_path, loss_mode="moon-weighted", moon_weight=1.0, fresh=True)) < 1e-7


def test_snapshots(tmp_path):
    run(tmp_path, snapshot_every=5)
    assert (tmp_path / "x_it5.pt").exists() and (tmp_path / "x_it10.pt").exists()
