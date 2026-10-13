"""Unit tests for the frozen-base geometry branch (ContactScatterField frozen_geo, cs.build name "Bf"). Run on Polaris: python scripts/bfrozen_test.py [E ckpt]."""
import os
import sys
import tempfile
import types

import numpy as np
import torch

from scripts import scatter_contact as sc
from scripts import scatter_contact_scenes as cs
from scripts import scatter_field as sf
from scripts import scatter_scenes as ss
from scripts import train_scatter_field as tsf

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CK = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt"
OK = []


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name, info, flush=True)
    OK.append(bool(cond))


def scene_tensors(n=10, seed=3):
    rng = np.random.default_rng(seed)
    pos = torch.tensor(rng.uniform(-6, 6, (2, n, 2)), dtype=torch.float32, device=dev)
    vel = torch.tensor(rng.uniform(-3, 3, (2, n, 2)), dtype=torch.float32, device=dev)
    return pos, vel, torch.ones(2, n, device=dev), torch.ones(2, n, device=dev)


def randomise(m, std=0.3):
    torch.manual_seed(0)
    for mod in (m.wmlp, m.cmlp):
        torch.nn.init.normal_(mod[-1].weight, std=std)
        torch.nn.init.normal_(mod[-1].bias, std=std)
    with torch.no_grad():
        m.gvec.copy_(torch.tensor([0.9, 0.1]))


def test_identity():
    E = cs.build("E", dev, CK)
    B = cs.build("Bf", dev, CK)
    pos, vel, mass, mask = scene_tensors()
    Pe, Ve = sf.rollout(E, pos, vel, mass, mask, 6)
    for tag in ("untrained", "random heads"):
        if tag == "random heads":
            randomise(B)
        B.set_scene(None, None)
        P, V = sf.rollout(B, pos, vel, mass, mask, 6)
        check(f"Bf {tag}, no geometry == Exp E bitwise", torch.equal(P, Pe) and torch.equal(V, Ve), f"{float((P - Pe).abs().max()):.1e}")
    geom = sc.rasterize([sc.hbox((-7, -7), (7, 7)), sc.circle((1.0, 1.0), 1.5)], 128, 64.0).to(dev)[None]
    B.set_scene(geom, None)
    P, V = sf.rollout(B, pos, vel, mass, mask, 6)
    check("Bf random heads with geometry differs from E", float((P - Pe).abs().max()) > 1e-3, f"{float((P - Pe).abs().max()):.3g}")
    B.set_scene(None, None)


def test_freeze():
    B = cs.build("Bf", dev, CK)
    B.train()
    rng = np.random.default_rng(5)
    scenes = [cs.with_geom(cs.gen_scene(rng, "box15")) for _ in range(4)]
    data = []
    for s in scenes:
        P, V = cs.truth(s, 24, dev)
        data.append((P + sf.CENTER, V, s["mass"]))
    tensors = sf.to_tensors(data, dev)
    geom, rad = cs._geom_stack(scenes, dev), cs._rad_pad(scenes, tensors[0].shape[2]).to(dev)
    for only, names in (("gvec,wmlp", ("gvec", "wmlp")), ("gvec,wmlp,cmlp", ("gvec", "wmlp", "cmlp"))):
        B = cs.build("Bf", dev, CK)
        B.train()
        torch.manual_seed(1)
        for mod in (B.wmlp, B.cmlp):
            torch.nn.init.normal_(mod[-1].weight, std=0.05)
        before = {k: v.detach().clone() for k, v in B.state_dict().items()}
        args = types.SimpleNamespace(lr=1e-3, device=str(dev), seed=4738, opt="", time_budget=0, iters=8, k_start=4, k_end=6, steps=24, batch=2, log_every=1000, loss_cap=100.0, train_only=only,
                                     ema=0.0, lr_warmup=0, snap_iters="", w_vel=0.1, long_windows=0, batch_hook=lambda idx: B.set_scene(geom[idx], rad[idx]))
        path = os.path.join(tempfile.mkdtemp(), "t.pt")
        tsf.train(B, tensors, args, path, lambda s: None)
        moved = [k for k, v in B.state_dict().items() if not torch.equal(v, before[k])]
        bad = [k for k in moved if not any(k.startswith(n) for n in names)]
        check(f"train-only {only}: only {names} parameters move", len(bad) == 0 and any(k.startswith("wmlp") for k in moved) and any(k.startswith("gvec") for k in moved), f"moved {len(moved)}, outside {bad[:3]}")
        if "cmlp" in names:
            check("train-only with cmlp moves cmlp", any(k.startswith("cmlp") for k in moved))
        else:
            check("train-only without cmlp leaves cmlp at zero", not any(k.startswith("cmlp") for k in moved))
    B.set_scene(None, None)


def test_checkpoint_roundtrip():
    B = cs.build("Bf", dev, CK)
    randomise(B)
    path = os.path.join(tempfile.mkdtemp(), "b.pt")
    torch.save({"model": B.state_dict()}, path)
    B2 = cs.build("Bf", dev, path)
    pos, vel, mass, mask = scene_tensors()
    geom = sc.rasterize([sc.hbox((-7, -7), (7, 7))], 128, 64.0).to(dev)[None]
    B.set_scene(geom, None)
    B2.set_scene(geom, None)
    Pa, _ = sf.rollout(B, pos, vel, mass, mask, 4)
    Pb, _ = sf.rollout(B2, pos, vel, mass, mask, 4)
    check("Bf checkpoint round trip reproduces rollout", torch.equal(Pa, Pb))


def main():
    test_identity()
    test_freeze()
    test_checkpoint_roundtrip()
    print("ALL PASS" if all(OK) else f"FAILURES {OK.count(False)}")
    sys.exit(0 if all(OK) else 1)


main()
