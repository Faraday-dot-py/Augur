import json
import sys

import torch

from model.tiered import TierNet, TieredField, ZeroTier0, nfeat
from scripts import train_tiered as tt


def field(net, K, n0, G=8, halo=4):
    return TieredField(ZeroTier0(), [net] * K, n0, 0.5, G, halo, 2, static=True)


def scene(n=30, seed=4738, box=30.0):
    g = torch.Generator().manual_seed(seed)
    pos = (torch.rand(1, n, 2, generator=g) - 0.5) * box
    return pos, torch.randn(1, n, 2, generator=g), torch.ones(1, n), torch.ones(1, n)


def test_default_tiernet_is_unchanged_by_the_scale_option():
    a = TierNet(nfeat(True, False, False), 1.0, 8, 2).state_dict()
    b = TierNet(nfeat(True, False, False), 1.0, 8, 2, scale_ch=0).state_dict()
    assert a.keys() == b.keys() and all(a[k].shape == b[k].shape for k in a)


def test_one_shared_net_extends_to_more_tiers_and_sees_the_scale():
    torch.manual_seed(4738)
    net = TierNet(nfeat(True, False, False) + 1, 0.5, 8, 2, scale_ch=1, log_rho=True, err=False)
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.05)
    f2, f3 = field(net, 2, 128), field(net, 3, 256)
    pos, vel, m, mk = scene()
    with torch.no_grad():
        a2, _ = f2.upper_force(pos, vel, m, mk, f2.init_state(1, pos.device))
        a3, _ = f3.upper_force(pos, vel, m, mk, f3.init_state(1, pos.device))
    assert torch.isfinite(a2).all() and torch.isfinite(a3).all() and a3.shape == pos.shape
    assert not torch.allclose(a2, a3)


def test_scale_channel_changes_a_shared_net_output():
    torch.manual_seed(4738)
    net = TierNet(nfeat(True, False, False) + 1, 0.5, 8, 2, scale_ch=1, log_rho=True, err=False)
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.05)
    x = torch.randn(1, nfeat(True, False, False) + 1, 8, 8).abs()
    x2 = x.clone()
    x2[:, -1] += 3.0
    with torch.no_grad():
        assert not torch.allclose(net(x, cell=1.0)[0], net(x2, cell=1.0)[0])
        assert torch.allclose(net(x, cell=2.0)[0][:, 1], net(x, cell=1.0)[0][:, 1] / 2, atol=1e-6)


def test_trainer_shared_joint_far_then_extension_eval_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(tt, "N0", 256)
    base = ["train_tiered.py", "--device", "cpu", "--ckpt-dir", str(tmp_path), "--target", "far", "--joint", "--shared-scale", "--tiers", "2",
            "--width", "4", "--layers", "2", "--scenes", "3", "--steps", "10", "--iters", "4", "--batch", "2", "--ckpt-every", "2", "--log-every", "2",
            "--val-every", "2", "--val-scenes", "2", "--tag", "smoke"]
    monkeypatch.setattr(sys, "argv", ["x", "--mode", "S"] + base[1:])
    tt.main()
    st = torch.load(tmp_path / "tiered_Ssmoke.pt", weights_only=False)
    assert [v["it"] for v in st["val"]] == [2, 4] and all(k.split(".")[0] in "01" for k in st["nets"])
    assert torch.equal(st["nets"]["0.inp.weight"], st["nets"]["1.inp.weight"])
    out = tmp_path / "far.json"
    monkeypatch.setattr(sys, "argv", ["x", "--mode", "far", "--out", str(out), "--eval-scenes", "2", "--eval-steps", "10"] + base[1:5] + base[5:7] + ["--tiers", "3"] +
                        ["--joint", "--shared-scale", "--target", "far", "--width", "4", "--layers", "2", "--tag", "smoke", "--ckpt-dir", str(tmp_path)])
    tt.main()
    assert set(json.load(open(out))) == {"9100", "9200", "9300"}
