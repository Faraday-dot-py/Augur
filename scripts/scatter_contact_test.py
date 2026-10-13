"""Unit tests for scripts/scatter_contact.py (run on Polaris: python scripts/scatter_contact_test.py [E ckpt])."""
import sys

import numpy as np
import torch

from scripts import scatter_contact as sc
from scripts import scatter_contact_scenes as cs
from scripts import scatter_field as sf
from scripts import scatter_scenes as ss

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CK = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt"
OK = []


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name, info, flush=True)
    OK.append(bool(cond))


def test_truth_parity():
    rng = np.random.default_rng(1)
    pos, vel = rng.uniform(-6, 6, (12, 2)), rng.uniform(-2.3, 2.3, (12, 2))
    Pa, Va = ss.bounce_truth(pos, vel, 30, dev)
    Pb, Vb = sc.contact_truth([sc.hbox((-7, -7), (7, 7))], pos, vel, np.full(12, ss.BOX_R), 30, dev)
    check("truth parity vs bounce_truth (box15, R .75)", np.abs(Pa - Pb).max() < 1e-8, f"max {np.abs(Pa - Pb).max():.2e}")


def test_sdf():
    p = torch.tensor([[0.0, 0.0], [6.0, 0.0], [5.0, 6.5], [8.0, 0.0]], dtype=torch.float64)
    d = sc.prim_sdf(sc.hbox((-7, -7), (7, 7)), p)
    check("hbox sdf", torch.allclose(d, torch.tensor([7.0, 1.0, 0.5, -1.0], dtype=torch.float64)), str(d.tolist()))
    c = sc.circle((1.0, 1.0), 2.0)
    check("circle sdf", abs(float(sc.prim_sdf(c, torch.tensor([4.0, 1.0], dtype=torch.float64))) - 1.0) < 1e-12)
    check("hdisc sdf", abs(float(sc.prim_sdf(sc.hdisc((0, 0), 5.0), torch.tensor([3.0, 0.0], dtype=torch.float64))) - 2.0) < 1e-12)
    tri = sc.poly([(0, 0), (4, 0), (0, 3)])
    pts = torch.tensor([[1.0, 1.0], [-1.0, 0.0], [5.0, 0.0], [3.0, 3.0], [2.0, -1.0]], dtype=torch.float64)
    d = sc.prim_sdf(tri, pts)
    want = [-1.0, 1.0, 1.0, 1.8, 1.0]
    check("triangle sdf (inside negative, edge/vertex distances)", np.allclose(d.numpy(), want, atol=1e-9), str(d.tolist()))
    rng = np.random.default_rng(0)
    for pr in (c, tri, sc.rect_poly((0, 0), 3, 2, 0.4), sc.hdisc((0, 0), 5.0), sc.hbox((-7, -7), (7, 7))):
        q = torch.tensor(rng.uniform(-6.5, 6.5, (200, 2)), dtype=torch.float64)
        q = q[sc.prim_sdf(pr, q).abs() > 0.05]
        (d, n), = [sc.prim_terms(pr, q)[0]] if pr["kind"] != "hbox" else [(sc.prim_sdf(pr, q), None)]
        if n is None:
            continue
        e = 1e-6
        gx = (sc.prim_sdf(pr, q + torch.tensor([e, 0], dtype=torch.float64)) - sc.prim_sdf(pr, q - torch.tensor([e, 0], dtype=torch.float64))) / (2 * e)
        gy = (sc.prim_sdf(pr, q + torch.tensor([0, e], dtype=torch.float64)) - sc.prim_sdf(pr, q - torch.tensor([0, e], dtype=torch.float64))) / (2 * e)
        err = (torch.stack([gx, gy], -1) - n).abs().max()
        check(f"normal == grad sdf ({pr['kind']})", float(err) < 1e-4, f"{float(err):.1e}")


def test_gather():
    prims = [sc.hbox((-7, -7), (7, 7)), sc.circle((2.0, 1.0), 1.5)]
    geom = sc.rasterize(prims, 128, 64.0).to(dev)[None]
    m = sc.ContactScatterField(extent=64.0, grid=128, kernel=True, pp=2.0, split=True, potential=True, verlet=True, net="unet").to(dev)
    q = torch.tensor([[[-3.1, 0.7], [6.2, -2.2], [40.0, 0.0], [0.0, -50.0]]], device=dev)
    g = m._gather_geom(geom, q)[0, :, 0].cpu().double()
    want = sc.scene_sdf(prims, q[0].double().cpu())
    check("grid_sample sdf gather matches analytic (interior, linear pieces)", np.allclose(g[:2].numpy(), want[:2].numpy(), atol=0.05), f"{g[:2].tolist()} vs {want[:2].tolist()}")
    check("gather clamps (border) outside the grid", bool(torch.isfinite(g).all()) and g[2] < 0 and g[3] < 0, str(g.tolist()))
    xs = (np.arange(128) + 0.5) * 0.5 - 32
    node = torch.tensor([[[xs[40], xs[77]]]], device=dev, dtype=torch.float32)
    gn = float(m._gather_geom(geom, node)[0, 0, 0])
    check("gather at node centre returns node value (x along width)", abs(gn - float(geom[0, 0, 77, 40])) < 1e-5, f"{gn} vs {float(geom[0, 0, 77, 40])}")


def models():
    cfg = sc.sf  # noqa
    out = {}
    for name in ("B", "A"):
        out[name] = cs.build(name, dev, CK)
    out["E"] = cs.build("E", dev, CK)
    return out


def test_identity(ms):
    rng = np.random.default_rng(3)
    pos = torch.tensor(rng.uniform(-6, 6, (2, 10, 2)), dtype=torch.float32, device=dev)
    vel = torch.tensor(rng.uniform(-2, 2, (2, 10, 2)), dtype=torch.float32, device=dev)
    mass, mask = torch.ones(2, 10, device=dev), torch.ones(2, 10, device=dev)
    geom = sc.rasterize([sc.hbox((-7, -7), (7, 7))], 128, 64.0).to(dev)[None]
    ref = None
    for name in ("E", "B", "A"):
        m = ms[name]
        if hasattr(m, "set_scene"):
            m.set_scene(geom, None)
        P, V = sf.rollout(m, pos, vel, mass, mask, 4)
        if ref is None:
            ref = (P, V)
        else:
            check(f"zero-init {name} == Exp E", float((P - ref[0]).abs().max()) == 0.0 and float((V - ref[1]).abs().max()) == 0.0, f"{float((P - ref[0]).abs().max()):.1e}")


def test_grad(ms):
    for name, head in (("B", "wmlp"), ("A", "wkmlp")):
        m = ms[name]
        m.train()
        for mod in (getattr(m, head), m.cmlp):
            torch.nn.init.normal_(mod[-1].weight, std=0.1)
        geom = sc.rasterize([sc.hbox((-7, -7), (7, 7)), sc.circle((0.0, 0.0), 1.5)], 128, 64.0).to(dev)[None]
        m.set_scene(geom, None)
        pos = torch.tensor([[[5.8, 0.0], [4.6, 0.6], [1.0, 2.5], [0.2, -3.0]]], device=dev, requires_grad=True)
        vel = torch.randn(1, 4, 2, device=dev)
        mass, mask = torch.ones(1, 4, device=dev), torch.ones(1, 4, device=dev)
        field = m.init_field(1, dev)
        p, v, _, _, dv = sf._step_m(m, pos, vel, mass, mask, field)
        (p.sum() + (v ** 2).sum()).backward()
        gs = [float(x.grad.abs().sum()) for x in (getattr(m, head)[0].weight, m.cmlp[0].weight, m.gvec)]
        check(f"{name} grads finite and nonzero (wall head, pair head, gvec)", all(np.isfinite(gs)) and gs[0] > 0 and gs[1] > 0 and gs[2] > 0, str(gs))
        m.zero_grad()
        m.eval()


def test_knn():
    g = torch.Generator().manual_seed(0)
    pos = ((torch.rand(2, 600, 2, generator=g) - 0.5) * 40).to(dev)
    mask = torch.ones(2, 600, device=dev)
    mask[1, 500:] = 0
    ds1, r1, nb1 = sc.knn_dense(pos, mask, 8)
    ds2, r2, nb2 = sc.knn_cell(pos, mask, 8, 4.0, 64.0)
    sel = (r1 < 4.0) & (mask.reshape(-1)[:, None] > 0)
    check("knn cell list == dense where r_k < cell", bool(torch.allclose(r1[sel], r2[sel], atol=1e-4)) and bool((nb1[sel] == nb2[sel]).all()), f"{int(sel.sum())} entries")


def main():
    test_truth_parity()
    test_sdf()
    test_gather()
    test_knn()
    ms = models()
    test_identity(ms)
    test_grad(ms)
    print("ALL PASS" if all(OK) else f"FAILURES {OK.count(False)}")
    sys.exit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()
