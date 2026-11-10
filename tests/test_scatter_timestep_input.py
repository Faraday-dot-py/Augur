import torch

from scripts import scatter_field as sf

KW = dict(grid=32, extent=16.0, net="unet", kernel=True, pp=2.0, split=True, potential=True, verlet=True, width=8)


def tiny(**kw):
    torch.manual_seed(0)
    return sf.ScatterField(**{**KW, **kw})


def randomise(m):
    g = torch.Generator().manual_seed(1)
    for p in m.parameters():
        p.data = torch.randn(p.shape, generator=g) * 0.05
    return m


def scene(n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = (torch.rand(1, n, 2, generator=g) * 8 - 4)
    vel = torch.randn(1, n, 2, generator=g) * 0.3
    return pos, vel, torch.ones(1, n), torch.ones(1, n)


def accel(m, pos, vel, mass, mask):
    f = m.init_field(1, "cpu")[0]
    return m.force(pos, vel, mass, mask, f)[0]


def test_dtin_warm_start_reproduces_control_at_every_step_size():
    base = randomise(tiny())
    dm = tiny(dtin=True)
    sf.load_compat(dm, base.state_dict())
    s = scene()
    ref = accel(base, *s)
    for dt in (0.05, 0.075, 0.1, 0.2):
        dm.dt = dt
        assert torch.allclose(ref, accel(dm, *s), atol=1e-5)


def test_dtin_output_depends_on_step_size_once_columns_are_nonzero():
    dm = randomise(tiny(dtin=True))
    s = scene()
    dm.dt = 0.1
    a = accel(dm, *s)
    dm.dt = 0.05
    assert not torch.allclose(a, accel(dm, *s), atol=1e-6)


def test_dtin_off_ignores_step_size():
    m = randomise(tiny())
    s = scene()
    m.dt = 0.1
    a = accel(m, *s)
    m.dt = 0.05
    assert torch.equal(a, accel(m, *s))


def test_dt_feature_uses_force_evaluation_step():
    m = tiny(dtin=True)
    m.dt = 0.1
    assert m.dt_feat() == 0.0
    m.substeps = 2
    assert abs(m.dt_feat() - torch.log(torch.tensor(0.5)).item()) < 1e-6


def test_dtin_adds_one_input_channel_and_one_mlp_column():
    base, dm = tiny(), tiny(dtin=True)
    assert dm.net.inp.weight.shape[1] == base.net.inp.weight.shape[1] + 1
    assert dm.kmlp[0].weight.shape[1] == base.kmlp[0].weight.shape[1] + 1
    assert dm.ppmlp[0].weight.shape[1] == base.ppmlp[0].weight.shape[1] + 1


def test_train_path_gen_train_resume_eval(tmp_path, monkeypatch):
    from scripts import train_scatter_field as tsf
    from scripts import train_scatter_timestep as tt

    monkeypatch.setitem(tsf.VARIANTS, "tiny_dtin", dict(net="unet", kernel=True, pp=2.0, split=True, potential=True, verlet=True, grid=32, dtin=True, width=8))
    monkeypatch.setitem(tsf.VARIANTS, "tiny_ctrl", dict(net="unet", kernel=True, pp=2.0, split=True, potential=True, verlet=True, grid=32, width=8))
    monkeypatch.setattr(tt, "VARIANT", "tiny_dtin")
    base = ["--device", "cpu", "--data-dir", str(tmp_path / "d"), "--out-dir", str(tmp_path / "o"), "--ckpt-dir", str(tmp_path / "c"),
            "--train-scenes", "4", "--steps", "6", "--eval-scenes", "2", "--eval-steps", "21", "--val-scenes", "2", "--val-every", "2", "--batch", "2",
            "--k-start", "2", "--k-end", "3", "--ckpt-every", "2", "--log-every", "2", "--iters", "4"]
    assert tt.main(["--mode", "gen"] + base) == 0
    for dt in tt.DT_TRAIN:
        assert (tmp_path / "d" / f"train_dt{dt:g}_seed4738.pt").exists()
    for dt in tt.DT_TRAIN + tt.DT_EVAL_ONLY:
        for s in tt.EVAL_SEEDS:
            assert (tmp_path / "d" / f"eval_dt{dt:g}_seed{s}.pt").exists()
    ctrl = tt.build_model("tiny_ctrl", torch.device("cpu"))
    torch.save({"model": ctrl.state_dict()}, tmp_path / "init.pt")
    assert tt.main(["--mode", "train", "--init-ckpt", str(tmp_path / "init.pt")] + base) == 0
    st = torch.load(tmp_path / "c" / "timestep-as-input-model.pt", weights_only=False)
    assert st["it"] == 4 and len(st["val_hist"]) == 2
    assert tt.main(["--mode", "train", "--init-ckpt", str(tmp_path / "init.pt")] + base) == 0
    assert tt.main(["--mode", "eval"] + base) == 0
    import json
    r = json.load(open(tmp_path / "o" / "timestep-as-input-model_eval.json"))
    assert set(r["dt0.075"]) == {"9100", "9200", "9300"} and r["dt0.1"]["9100"]["finite"]


def test_plateau_guard_ten_percent_windows_three_percent_drop():
    from scripts import train_scatter_timestep as tt

    flat = [(i, 1.0) for i in range(10, 101, 10)]
    falling = [(i, 1.0 if i <= 80 else 0.9) for i in range(10, 101, 10)]
    small = [(i, 1.0 if i <= 80 else 0.98) for i in range(10, 101, 10)]
    assert not tt.plateau_falling(flat, 100)
    assert tt.plateau_falling(falling, 100)
    assert not tt.plateau_falling(small, 100)
    assert not tt.plateau_falling([], 100)
