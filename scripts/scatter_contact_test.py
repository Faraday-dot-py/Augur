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
    hp = sc.hpoly(sc.rect_poly((0, 0), 4, 4, 0.0)["pts"])
    q = torch.tensor([[0.0, 0.0], [3.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    (d, n), = sc.prim_terms(hp, q)
    check("hpoly sdf (free inside) and inward normal", np.allclose(d.numpy(), [2.0, -1.0, 1.0], atol=1e-9) and np.allclose(n[2].numpy(), [-1.0, 0.0], atol=1e-9), f"{d.tolist()} {n[2].tolist()}")
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
    for name in ("B", "A", "C"):
        out[name] = cs.build(name, dev, CK)
    out["E"] = cs.build("E", dev, CK)
    return out


def test_identity(ms):
    rng = np.random.default_rng(3)
    pos = torch.tensor(rng.uniform(-6, 6, (2, 10, 2)), dtype=torch.float32, device=dev)
    vel = torch.tensor(rng.uniform(-2, 2, (2, 10, 2)), dtype=torch.float32, device=dev)
    mass, mask = torch.ones(2, 10, device=dev), torch.ones(2, 10, device=dev)
    geom = sc.rasterize([sc.hbox((-7, -7), (7, 7)), sc.circle((1.0, 1.0), 1.5)], 128, 64.0).to(dev)[None]
    ref = None
    for name in ("E", "B", "A", "C"):
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


def test_chan(ms):
    m = ms["C"]
    m.train()
    geom = sc.rasterize([sc.hbox((-7, -7), (7, 7)), sc.circle((0.0, 0.0), 1.5)], 128, 64.0).to(dev)[None]
    f = sc.chan_feats(geom, m.h)
    check("chan feats: 4 channels, occupancy in [0,1], |sdf|/8 <= 1, finite", f.shape[1] == 4 and float(f[:, 0].min()) >= 0 and float(f[:, 0].max()) <= 1 and float(f[:, 1].abs().max()) <= 1 and bool(torch.isfinite(f).all()))
    check("chan model: no wall MLP, extra input weights zero-init", not hasattr(m, "wmlp") and float(m.net.gin.weight.abs().sum()) == 0.0 and float(m.net.gin.bias.abs().sum()) == 0.0)
    pos = torch.tensor([[[5.9, 0.3]]], device=dev)
    vel = torch.tensor([[[8.0, 0.0]]], device=dev)
    mass, mask = torch.ones(1, 1, device=dev), torch.ones(1, 1, device=dev)
    field = m.init_field(1, dev)[0]
    torch.manual_seed(0)
    torch.nn.init.normal_(m.net.gin.weight, std=1.0)
    m.set_scene(None, None)
    a_free, _ = m._force(pos, vel, mass, mask, field)
    m.set_scene(geom, None)
    a_wall, _ = m._force(pos, vel, mass, mask, field)
    check("single ball: momfix zeroes the field force without geometry, geometry force survives momfix", float(a_free.abs().max()) < 1e-5 and float(a_wall.abs().max()) > 1e-3, f"{float(a_free.abs().max()):.2e} {float(a_wall.abs().max()):.2e}")
    with torch.no_grad():
        m.net.gin.weight.mul_(100.0)
    pos = torch.tensor(np.random.default_rng(5).uniform(-6, 6, (1, 8, 2)), dtype=torch.float32, device=dev)
    vel = torch.zeros(1, 8, 2, device=dev)
    mass, mask = torch.ones(1, 8, device=dev), torch.ones(1, 8, device=dev)
    a_wall, _ = m._force(pos, vel, mass, mask, field)
    m.set_scene(None, None)
    a_free, _ = m._force(pos, vel, mass, mask, field)
    dmax = float((a_wall - a_free).abs().max())
    check("geometry force bounded by FMAX per component (weights x100)", 1e-3 < dmax <= sc.FMAX * (1 + 1e-6), f"{dmax:.3f}")
    m.set_scene(geom, None)
    torch.nn.init.normal_(m.net.gin.weight, std=0.5)
    pos = torch.tensor([[[5.8, 0.0], [4.6, 0.6], [1.0, 2.5], [0.2, -3.0]]], device=dev, requires_grad=True)
    p, v, _, _, dv = sf._step_m(m, pos, torch.randn(1, 4, 2, device=dev), torch.ones(1, 4, device=dev), torch.ones(1, 4, device=dev), m.init_field(1, dev))
    (p.sum() + (v ** 2).sum()).backward()
    g = float(m.net.gin.weight.grad.abs().sum())
    check("gin gradient finite and nonzero", np.isfinite(g) and g > 0, f"{g:.3g}")
    m.zero_grad()
    m.eval()
    m.set_scene(None, None)


def test_scenes():
    from scripts import scatter_contact_rand as rc

    rng = np.random.default_rng(11)
    got = {}
    for fam in rc.SPECS:
        sc_ = [rc.gen_scene(rng, fam) for _ in range(40)]
        got[fam] = sc_
        bad = 0
        for s in sc_:
            sd = sc.scene_sdf(s["prims"], torch.tensor(s["pos"], dtype=torch.float64)).numpy()
            d = np.linalg.norm(s["pos"][:, None] - s["pos"][None], axis=-1) + 10 * np.eye(len(s["pos"]))
            bad += int((sd < rc.R0 + 0.1 - 1e-9).any() or (d < 2 * rc.R0 + 0.1 - 1e-9).any() or not 2 <= len(s["pos"]) <= 15)
        check(f"scenes {fam}: balls in free space, no overlap, 2-15 balls", bad == 0, f"{bad} bad")
    obs = lambda s: [p for p in s["prims"] if "size" in p and p["shape"] != "floor"]
    cnt = {n: sum(len(obs(s)) == n for s in got["c_train"]) for n in range(5)}
    check("c_train: obstacle counts within 0-4 and all counts occur", max(len(obs(s)) for s in got["c_train"]) <= 4 and min(cnt.values()) > 0, str(cnt))
    ok = all(0.8 <= p["size"] <= 3.2 and p["ctr"][1] <= 3.5 and p["convex"] for s in got["c_train"] for p in obs(s))
    check("c_train: sizes .8-3.2, centres y<=3.5, convex only", ok)
    check("c_train: containers box/disc/open only", all(s["cont"] in ("box", "disc", "open") for s in got["c_train"]) and len({s["cont"] for s in got["c_train"]}) == 3)
    check("c_big: sizes >= 3.8 (above train max 3.2)", all(p["size"] >= 3.8 for s in got["c_big"] for p in obs(s)))
    check("c_small: sizes <= .6 (below train min .8)", all(p["size"] <= 0.6 for s in got["c_small"] for p in obs(s)))
    check("c_pos: obstacle centres y in [4.5,10] (train max 3.5)", all(4.5 <= p["ctr"][1] <= 10 for s in got["c_pos"] for p in obs(s)))
    check("c_count: 6-8 obstacles (train max 4)", all(6 <= len(obs(s)) <= 8 for s in got["c_count"]), str(sorted({len(obs(s)) for s in got["c_count"]})))
    check("c_nonconvex: every scene has a non-convex obstacle, all three classes occur", all(any(not p["convex"] for p in obs(s)) for s in got["c_nonconvex"]) and {p["shape"] for s in got["c_nonconvex"] for p in obs(s)} >= set(rc.NONCONVEX))
    check("c_cont: container is an hpoly (never in train)", all(s["prims"][0]["kind"] == "hpoly" for s in got["c_cont"]))
    check("train never uses hpoly or non-convex", not any(p["kind"] == "hpoly" or not p.get("convex", True) for s in got["c_train"] for p in s["prims"]))


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
    test_chan(ms)
    test_scenes()
    print("ALL PASS" if all(OK) else f"FAILURES {OK.count(False)}")
    sys.exit(0 if all(OK) else 1)


main()
