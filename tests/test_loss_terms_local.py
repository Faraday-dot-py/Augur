import torch

from scripts import loss_terms_local as lt


def scene(B=2, N=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(B, N, 2, generator=g) * 20
    pos[:, :20] = pos[:, :20] * 0.15 + 5
    mask = torch.ones(B, N)
    mask[1, -10:] = 0
    return pos, mask


def test_density_bins_split_and_order():
    pos, mask = scene()
    bins = lt.density_bins(pos, mask)
    for b in range(2):
        ok = mask[b] > 0
        n = int(ok.sum())
        counts = [int(((bins[b] == k) & ok).sum()) for k in range(3)]
        assert sum(counts) == n
        assert abs(counts[0] - 0.5 * n) <= 1 and abs(counts[2] - 0.1 * n) <= 1
        d2 = ((pos[b][:, None] - pos[b][None]) ** 2).sum(-1)
        cnt = ((d2 < lt.RADIUS ** 2) & ok[None]).sum(-1)
        assert cnt[(bins[b] == 2) & ok].min() >= cnt[(bins[b] == 0) & ok].max()
    assert (bins[1, -10:] == 0).all()


def test_work_bias_zero_when_exact_and_signed_not_cancelled():
    pos, mask = scene()
    bins = lt.density_bins(pos, mask)
    v0 = torch.randn(2, 60, 2)
    vt = v0 + 0.3 * torch.randn(2, 60, 2)
    m = torch.ones(2, 60)
    assert float(lt.work_bias_step(vt, vt, v0, m, mask, bins)) == 0.0
    dke_t = 0.5 * ((vt ** 2).sum(-1) - (v0 ** 2).sum(-1))
    g = torch.Generator().manual_seed(1)
    sgn = torch.sign(torch.randn(2, 60, generator=g))
    shift = 0.2
    one_sided = torch.sqrt(((dke_t + shift) * 2 + (v0 ** 2).sum(-1)).clamp_min(0))
    vm_one = vt / vt.norm(dim=-1, keepdim=True) * one_sided[..., None]
    term_one = lt.work_bias_step(vm_one, vt, v0, m, mask, bins)
    dke_m = 0.5 * ((vm_one ** 2).sum(-1) - (v0 ** 2).sum(-1))
    err = ((dke_m - dke_t) * mask)
    ref = 0.0
    cnt = 0
    for b in range(2):
        for k in range(3):
            sel = (bins[b] == k) & (mask[b] > 0)
            if sel.any():
                ref += float(err[b][sel].mean() ** 2)
                cnt += 1
    assert abs(float(term_one) - ref / cnt) < 1e-5 * max(1.0, ref)
    scale = torch.sqrt(((dke_t + sgn * shift) * 2 + (v0 ** 2).sum(-1)).clamp_min(1e-9))
    zero_mean = vt / vt.norm(dim=-1, keepdim=True) * scale[..., None]
    assert float(lt.work_bias_step(zero_mean, vt, v0, m, mask, bins)) < float(term_one)


def test_work_bias_gradient_and_mask():
    pos, mask = scene()
    bins = lt.density_bins(pos, mask)
    v0 = torch.randn(2, 60, 2)
    vt = torch.randn(2, 60, 2)
    vm = (vt + 0.1).clone().requires_grad_(True)
    lt.work_bias_step(vm, vt, v0, torch.ones(2, 60), mask, bins).backward()
    assert torch.isfinite(vm.grad).all()
    assert (vm.grad[1, -10:] == 0).all()


def test_knn_mean_matches_bruteforce_and_ignores_padding():
    pos, mask = scene(seed=3)
    err = torch.randn(2, 60, 2)
    idx, ok = lt.knn_index(pos, mask, 16)
    got = lt.knn_mean(err * mask[..., None], idx, ok)
    for b in range(2):
        valid = (mask[b] > 0).nonzero().squeeze(-1)
        for i in valid[:5].tolist():
            d = ((pos[b, valid] - pos[b, i]) ** 2).sum(-1)
            nn = valid[d.topk(16, largest=False).indices]
            assert torch.allclose(got[b, i], err[b, nn].mean(0), atol=1e-5)


def test_coherent_term_rewards_incoherent_noise_less_than_shifts():
    pos, mask = scene(seed=4)
    vel = torch.zeros(2, 60, 2)
    g = torch.Generator().manual_seed(5)
    noise = torch.randn(2, 60, 2, generator=g) * 0.1
    shift = torch.zeros(2, 60, 2) + 0.1
    t_noise = float(lt.coherent_step(pos + noise, pos, vel, vel, mask))
    t_shift = float(lt.coherent_step(pos + shift, pos, vel, vel, mask))
    assert t_shift > 3 * t_noise
    assert abs(t_shift - 0.01) < 1e-5
    assert float(lt.coherent_step(pos, pos, vel, vel, mask)) == 0.0
    pm = (pos + shift).clone().requires_grad_(True)
    lt.coherent_step(pm, pos, vel, vel, mask).backward()
    assert torch.isfinite(pm.grad).all() and (pm.grad[1, -10:] == 0).all()
