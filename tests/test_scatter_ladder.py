import math

import torch

from scripts import scatter_field as sf
from scripts import scatter_ladder as sl


def lattice(G, h):
    idx = torch.arange(2 * G, dtype=torch.float64)
    d = torch.minimum(idx, 2 * G - idx) * h
    return torch.sqrt(d[:, None] ** 2 + d[None, :] ** 2)


def fft_pot(rho, K, G, h):
    rf = torch.fft.rfft2(rho.double(), s=(2 * G, 2 * G))
    return (torch.fft.irfft2(rf * torch.fft.rfft2(K), s=(2 * G, 2 * G)) * h ** 2)[..., :G, :G]


def blobs(G, seed=4738, n=12):
    g = torch.Generator().manual_seed(seed)
    rho = torch.zeros(2, 1, G, G, dtype=torch.float64)
    for b in range(2):
        for _ in range(n):
            x, y = torch.randint(G // 4, 3 * G // 4, (2,), generator=g).tolist()
            rho[b, 0, y, x] += float(torch.rand(1, generator=g)) + 0.5
    return rho


def test_fit_recovers_known_amplitudes():
    sig = sl.sigmas()
    r = lattice(128, 0.5)
    true = torch.tensor([0.3, -0.5, 0.2, 0.7, -0.1, 0.05, 0.4], dtype=torch.float64)
    amps = sl.fit_amps(r, sl.ladder_kernel(true, sig, r), sig)
    assert torch.allclose(amps, true, atol=1e-6)


def test_blur_ladder_matches_fft_convolution_of_same_kernel():
    G, h = 128, 0.5
    sig = sl.sigmas()
    amps = torch.tensor([0.6, -0.3, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=torch.float64)
    rho = blobs(G)
    ref = fft_pot(rho, sl.ladder_kernel(amps, sig, lattice(G, h)), G, h)
    got = sl.ladder_potential(rho, amps.tolist(), h)
    rel = (got - ref).abs().sum() / ref.abs().sum()
    assert rel < 0.05


def test_exponent_read_on_power_law():
    G, h = 128, 0.5
    sig = sl.sigmas()
    r = lattice(G, h)
    amps = sl.fit_amps(r, 1.0 / (r ** 2 + 0.25), sig)
    assert abs(sl.read_exponent(amps, sig) - 3.0) / 3.0 < 0.10


def test_exponent_read_on_inverse_distance_potential_is_finite():
    G, h = 128, 0.5
    sig = sl.sigmas()
    r = lattice(G, h)
    amps = sl.fit_amps(r, 1.0 / torch.sqrt(r ** 2 + 0.25), sig)
    assert math.isfinite(sl.read_exponent(amps, sig))


def test_tiled_with_full_halo_equals_single_grid():
    amps = [0.6, -0.3, 0.5, 0.4, 0.3, 0.2, 0.1]
    rho = blobs(128).float()
    one = sl.ladder_potential(rho, amps, 0.5)
    tiled = sl.ladder_potential_tiled(rho, amps, 0.5, 64, 64)
    assert torch.allclose(one, tiled, atol=1e-4 * one.abs().max().item())


def test_potential_is_linear_in_density():
    amps = [0.6, -0.3, 0.5, 0.4, 0.3, 0.2, 0.1]
    a, b = blobs(128, 1).float(), blobs(128, 2).float()
    lhs = sl.ladder_potential(a + 2 * b, amps, 0.5)
    rhs = sl.ladder_potential(a, amps, 0.5) + 2 * sl.ladder_potential(b, amps, 0.5)
    assert torch.allclose(lhs, rhs, atol=1e-4 * lhs.abs().max().item())


def test_ladder_scatter_field_steps_on_cpu():
    torch.manual_seed(4738)
    m = sl.LadderScatterField(grid=32, extent=16.0, net="unet", kernel=True, pp=2.0, split=True, potential=True, verlet=True, momfix=False).eval()
    m.set_amps([0.5, -0.2, 0.3, 0.2, 0.1])
    pos = torch.randn(2, 20, 2) * 3
    vel = torch.zeros(2, 20, 2)
    mass = torch.ones(2, 20)
    mask = torch.ones(2, 20)
    field = m.init_field(2, torch.device("cpu"))
    p, v, field, _ = m.step(pos, vel, mass, mask, field)
    assert torch.isfinite(p).all() and p.shape == pos.shape


def test_learned_kernel_matches_fft_kernel_transform():
    torch.manual_seed(4738)
    m = sf.ScatterField(grid=32, extent=16.0, net="unet", kernel=True, pp=2.0, split=True, potential=True, verlet=True)
    for p in m.kmlp.parameters():
        torch.nn.init.normal_(p, std=0.1)
    r, K = sl.learned_kernel(m)
    ref = torch.fft.irfft2(m._kernel_fft(torch.zeros(1, 1, 32, 32)), s=(64, 64))
    assert torch.allclose(K.float(), ref, atol=1e-5)


def test_fit_error_is_zero_for_exact_ladder():
    sig = sl.sigmas()
    r = lattice(128, 0.5)
    true = torch.tensor([0.3, -0.5, 0.2, 0.7, -0.1, 0.05, 0.4], dtype=torch.float64)
    K = sl.ladder_kernel(true, sig, r)
    assert sl.fit_error(r, K, sl.fit_amps(r, K, sig), sig) < 1e-6


def test_stage_one_script_runs_and_resumes(tmp_path, monkeypatch):
    import json
    import sys

    from scripts import scatter_ladder_stage1 as s1

    torch.manual_seed(4738)
    e = s1.build_e(torch.device("cpu"))
    for p in e.kmlp.parameters():
        torch.nn.init.normal_(p, std=0.1)
    ck = tmp_path / "e.pt"
    torch.save({"model": e.state_dict()}, ck)
    out = tmp_path / "o.json"
    argv = ["x", "--e-ckpt", str(ck), "--out", str(out), "--n", "2", "--seeds", "9100", "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    s1.main()
    res = json.load(open(out))
    assert set(res) == {"fit:E:pow2", "tiled", "timing", "rollout:9100"}
    assert all(k in res["rollout:9100"] for k in ("err20_ratio_mean", "err20_ratio_per_seed", "err20_E_noise_floor", "err5_ratio_mean", "force_rel_ratio_mean"))
    assert (tmp_path / "o.json.done").exists()
    mt = out.stat().st_mtime_ns
    s1.main()
    assert out.stat().st_mtime_ns == mt or json.load(open(out)) == res


def test_stage_one_script_fits_a_multi_law_checkpoint(tmp_path, monkeypatch):
    import json
    import sys

    from scripts import scatter_ladder_stage1 as s1
    from scripts import scatter_multi as sm

    torch.manual_seed(4738)
    e = s1.build_e(torch.device("cpu"))
    for p in e.kmlp.parameters():
        torch.nn.init.normal_(p, std=0.1)
    torch.save({"model": e.state_dict()}, tmp_path / "e.pt")
    mm = sm.build()
    for p in mm.kmlp.parameters():
        torch.nn.init.normal_(p, std=0.1)
    torch.save({"model": mm.state_dict()}, tmp_path / "m.pt")
    out = tmp_path / "o.json"
    monkeypatch.setattr(sys, "argv", ["x", "--e-ckpt", str(tmp_path / "e.pt"), "--main-ckpt", str(tmp_path / "m.pt"), "--out", str(out), "--device", "cpu", "--skip-rollout", "--n", "2"])
    s1.main()
    res = json.load(open(out))
    assert {f"fit:main:{s}" for s in sm.TRAIN_SHAPES} <= set(res)
    assert "exponent_rel_err" in res["fit:main:pow3"] and "exponent_rel_err" not in res["fit:main:lj"]
