import pytest
import torch

from model import sparse_voxel_scatter_field as sv

torch.set_default_dtype(torch.float32)


def make(seed=4738, **kw):
    torch.manual_seed(seed)
    m = sv.SparseVoxelScatterField(width=8, **kw)
    for p in m.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.05)
    return m.eval()


def scene(n, b=1, seed=4738, sigma=4.0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(b, n, 3, generator=g) * sigma
    vel = torch.randn(b, n, 3, generator=g) * 0.3
    return pos, vel, torch.ones(b, n), torch.ones(b, n)


def test_pack_unpack_roundtrip():
    u = torch.randint(0, 1 << 19, (50, 3))
    b = torch.randint(0, 8, (50,))
    key = sv.pack(b, u)
    assert torch.equal(sv.unpack(key & ((1 << sv.BSHIFT) - 1)), u)
    assert torch.equal(key >> sv.BSHIFT, b)


def test_deposit_conserves_mass_and_momentum():
    m = make()
    pos, vel, mass, mask = scene(60)
    levels, u0, frac, bidx, real = m.structure(pos, mask)
    x = m.deposit(levels, u0, frac, bidx, real, vel, mass)
    assert float(x[:, 0].sum() * m.h ** 3) == pytest.approx(60.0, rel=1e-5)
    assert torch.allclose(x[:, 1:].sum(0) * m.h ** 3, (vel * mass[..., None]).sum((0, 1)), atol=1e-3)


def test_levels_shrink_to_top_cap_and_parents_valid():
    m = make()
    pos, vel, mass, mask = scene(300)
    levels, *_ = m.structure(pos, mask)
    assert int(torch.bincount(levels[-1].keys >> sv.BSHIFT).max()) <= m.top_cap
    for l, L in enumerate(levels[:-1]):
        assert L.parent.max() < L.nparent == levels[l + 1].keys.shape[0]


def test_node_count_independent_of_extent():
    m = make()
    pos, vel, mass, mask = scene(200)
    far = pos.clone()
    far[:, :100] += 5000.0
    far2 = pos.clone()
    far2[:, :100] += 50.0
    n_far = m.structure(far, mask)[0][0].keys.shape[0]
    n_near = m.structure(far2, mask)[0][0].keys.shape[0]
    assert abs(n_far - n_near) < 0.3 * n_near
    assert n_far <= 8 * 27 * 200


@pytest.mark.parametrize("n", [5, 40, 200])
def test_force_finite_and_momentum_free(n):
    m = make()
    pos, vel, mass, mask = scene(n)
    with torch.no_grad():
        a, _ = m.force(pos, vel, mass, mask, (None, None)[0])
    assert a.shape == (1, n, 3) and torch.isfinite(a).all()
    assert float((a * mass[..., None]).sum(1).abs().max()) < 1e-4


def test_voxel_shift_and_permutation_invariance():
    m = make()
    pos, vel, mass, mask = scene(80)
    with torch.no_grad():
        a0, _ = m.force(pos, vel, mass, mask, None)
        a1, _ = m.force(pos + torch.tensor([7.0, -3.0, 11.0]) * m.h * 16, vel, mass, mask, None)
        perm = torch.randperm(80, generator=torch.Generator().manual_seed(3))
        a2, _ = m.force(pos[:, perm], vel[:, perm], mass[:, perm], mask[:, perm], None)
    assert torch.allclose(a1, a0, atol=2e-3)
    assert torch.allclose(a2, a0[:, perm], atol=1e-4)


def test_padding_invariance():
    m = make()
    pos, vel, mass, mask = scene(60)
    pad = lambda t: torch.cat([t, torch.randn(1, 20, *t.shape[2:]) * 30 if t.dim() == 3 else torch.zeros(1, 20)], 1)
    with torch.no_grad():
        a0, _ = m.force(pos, vel, mass, mask, None)
        a1, _ = m.force(pad(pos), pad(vel), pad(mass), torch.cat([mask, torch.zeros(1, 20)], 1), None)
    assert torch.allclose(a1[:, :60], a0, atol=1e-4)
    assert float(a1[:, 60:].abs().max()) == 0.0


def test_pair_term_matches_brute_force():
    m = make(cell_cap=64)
    torch.nn.init.zeros_(m.ppmlp[-1].weight)
    torch.nn.init.constant_(m.ppmlp[-1].bias, 1.0)
    pos, vel, mass, mask = scene(150, sigma=2.0)
    with torch.no_grad():
        got = m.pair_acc(pos, mass, mask)
        d = pos[0][None] - pos[0][:, None]
        r = torch.sqrt((d ** 2).sum(-1) + 1e-8)
        r.fill_diagonal_(1e6)
        rs, idx = torch.topk(r, 16, dim=-1, largest=False)
        ds = torch.gather(d, 1, idx[..., None].expand(-1, -1, 3))
        ok = (rs < m.pp).float()
        w = (1 - (rs / m.pp).clamp(max=1.0) ** 2) ** 2 * ok
        ref = (w[..., None] * ds / rs[..., None]).sum(1)
    assert m.cell_overflow == 0
    assert torch.allclose(got[0], ref, atol=1e-4)


def test_unequal_scenes_in_batch_match_single_when_depth_equal():
    m = make()
    p1, v1, m1, k1 = scene(100, seed=5)
    p2, v2, m2, k2 = scene(100, seed=6)
    with torch.no_grad():
        a, _ = m.force(torch.cat([p1, p2]), torch.cat([v1, v2]), torch.cat([m1, m2]), torch.cat([k1, k2]), None)
        a1, _ = m.force(p1, v1, m1, k1, None)
    assert torch.allclose(a[0], a1[0], atol=5e-3)


def test_step_gradient_reaches_modules_with_nonzero_signal():
    m = make()
    m.train()
    torch.nn.init.normal_(m.out.weight, std=0.05)
    for mm in (m.kmlp, m.ppmlp):
        torch.nn.init.normal_(mm[-1].weight, std=0.05)
    pos, vel, mass, mask = scene(300, sigma=60.0)
    p, v, f, dv = m.step(pos, vel, mass, mask, m.init_field(1, pos.device))
    (p.pow(2).sum() + v.pow(2).sum()).backward()
    missing = [n for n, q in m.named_parameters() if q.grad is None or not torch.isfinite(q.grad).all()]
    assert not missing, missing


def tiny_train_scenes():
    def small(num, steps, seed, dev):
        rng = torch.Generator().manual_seed(seed)
        P = torch.randn(num, steps + 1, 12, 3, generator=rng) * 3
        V = torch.randn(num, steps + 1, 12, 3, generator=rng) * 0.3
        return P.double(), V.double(), torch.ones(num, 12).double(), torch.ones(num, 12).double()

    return small


def segment(tmp_path, monkeypatch, name, iters, stop_at, dirs=None):
    import argparse

    from scripts import three_dimensional_sparse_voxel_scatter_field as tv

    monkeypatch.setattr(tv, "build", lambda dev: make().train())
    monkeypatch.setattr(tv, "train_scenes", tiny_train_scenes())
    out, ck = (tmp_path / name / "out", tmp_path / name / "ckpt") if dirs is None else dirs
    out.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(out=str(out), ckpt_dir=str(ck), ckpt_file="", stop_at=stop_at, iters=iters, train=3, val_scenes=2, steps=8, seed=4738, batch=2, k_start=1, k_end=3,
                              log_every=2, device="cpu", lr=5e-4)
    tv.STOP["flag"] = False
    try:
        tv.train(args, torch.device("cpu"))
        code = None
    except SystemExit as e:
        code = e.code
    return code, out, ck


def test_stop_at_then_resume_matches_uninterrupted(tmp_path, monkeypatch):
    code, out_a, ck_a = segment(tmp_path, monkeypatch, "straight", 8, 8)
    assert code is None and (out_a / "train_done.json").exists()
    code, out_b, ck_b = segment(tmp_path, monkeypatch, "split", 8, 4)
    assert code == 0
    assert not (out_b / "train_done.json").exists()
    name = "three_dimensional_sparse_voxel_scatter_field"
    sl = torch.load(ck_b / f"{name}-at-4-its.pt", weights_only=False)
    assert sl["it"] == 4
    code, _, _ = segment(tmp_path, monkeypatch, "split", 8, 8, dirs=(out_b, ck_b))
    assert code is None and (out_b / "train_done.json").exists()
    a = torch.load(ck_a / f"{name}.pt", weights_only=False)
    b = torch.load(ck_b / f"{name}.pt", weights_only=False)
    assert a["it"] == b["it"] == 8
    assert torch.equal(a["gen"], b["gen"])
    for k in a["model"]:
        assert torch.allclose(a["model"][k], b["model"][k], atol=1e-6), k
    assert all(torch.equal(a["opt"]["state"][i]["step"], b["opt"]["state"][i]["step"]) for i in a["opt"]["state"])


def test_stop_at_refuses_checkpoint_at_or_past_it(tmp_path, monkeypatch):
    code, out, ck = segment(tmp_path, monkeypatch, "past", 8, 4)
    assert code == 0
    code, _, _ = segment(tmp_path, monkeypatch, "past", 8, 4, dirs=(out, ck))
    assert isinstance(code, str) and "not below" in code


def test_stop_at_beyond_iters_refused(tmp_path, monkeypatch):
    code, _, _ = segment(tmp_path, monkeypatch, "beyond", 4, 6)
    assert isinstance(code, str) and "exceeds" in code


def test_final_segment_checkpoint_carries_generator_state(tmp_path, monkeypatch):
    code, out, ck = segment(tmp_path, monkeypatch, "off", 4, 0)
    assert code is None
    assert "gen" in torch.load(ck / "three_dimensional_sparse_voxel_scatter_field.pt", weights_only=False)


def test_load_candidate_reads_ckpt_file_and_iteration(tmp_path, monkeypatch):
    import argparse

    from scripts import three_dimensional_sparse_voxel_scatter_field as tv

    monkeypatch.setattr(tv, "build", lambda dev: make())
    path = tmp_path / "slice.pt"
    torch.save({"model": make(seed=1).state_dict(), "it": 10000}, path)
    args = argparse.Namespace(ckpt_file=str(path), ckpt_dir=str(tmp_path / "unused"))
    m = tv.load_candidate(args, torch.device("cpu"))
    assert args.ckpt_it == 10000 and tv.ckpt_path(args) == str(path)
    assert torch.equal(m.state_dict()["inp.weight"], make(seed=1).state_dict()["inp.weight"])
    assert not any(p.requires_grad for p in m.parameters())
