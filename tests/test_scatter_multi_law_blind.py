import sys

import torch

from scripts import scatter_multi as sm
from scripts import train_scatter_multi as tsm

DEV = torch.device("cpu")


def randomised(law_blind, seed=4738):
    torch.manual_seed(seed)
    m = sm.build(law_blind=law_blind)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    with torch.no_grad():
        m.kmlp.l1.weight[:, 2:] = 0.0
        m.ppmlp.mlp.l1.weight[:, 2:] = 0.0
    return m.eval()


def force_with(m, cond, trunk_sensitive_only=False):
    torch.manual_seed(1)
    pos = (torch.rand(1, 40, 2) - 0.5) * 40
    vel = torch.randn(1, 40, 2) * 0.5
    one = torch.ones(1, 40)
    m.set_law(cond, None)
    with torch.no_grad():
        a, _ = m.force(pos, vel, one, one.clone(), m.init_field(1, DEV)[0])
    return a


def test_law_blind_trunk_ignores_theta():
    c1, c2 = torch.zeros(1, sm.DC), torch.randn(1, sm.DC)
    blind = randomised(True)
    seeing = randomised(False)
    assert torch.allclose(force_with(blind, c1), force_with(blind, c2), atol=1e-6)
    assert not torch.allclose(force_with(seeing, c1), force_with(seeing, c2), atol=1e-6)


def test_law_blind_flag_changes_no_parameter_names_or_shapes():
    a, b = sm.build(law_blind=True).state_dict(), sm.build(law_blind=False).state_dict()
    assert a.keys() == b.keys() and all(a[k].shape == b[k].shape for k in a)


def test_trainer_head_only_law_blind_smoke(tmp_path, monkeypatch):
    argv = ["train_scatter_multi.py", "--tag", "smoke", "--device", "cpu", "--iters", "6", "--batch", "2", "--steps", "6", "--k-start", "2",
            "--k-end", "3", "--per-shape", "2", "--init-ckpt", "", "--ckpt-dir", str(tmp_path), "--ckpt-every", "3", "--snap-every", "100000",
            "--log-every", "3", "--law-blind", "--freeze-trunk", "--val-every", "3", "--val-per-shape", "1", "--val-k", "3", "--gate-every", "1000"]
    monkeypatch.setattr(sys, "argv", argv)
    tsm.main()
    st = torch.load(tmp_path / "smoke.pt", weights_only=False)
    assert st["it"] == 6 and [v["it"] for v in st["val"]] == [3, 6]
    assert (tmp_path / "smoke.done").exists()
    torch.manual_seed(4738)
    fresh = sm.build(law_blind=True).state_dict()
    assert all(torch.equal(st["model"][k], v) for k, v in fresh.items() if k.startswith("net."))
    assert any(not torch.equal(st["model"][k], v) for k, v in fresh.items() if k.startswith("kmlp."))
